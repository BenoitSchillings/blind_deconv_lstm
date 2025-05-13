import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
import matplotlib.pyplot as plt # For visualization

# --- Hyperparameters ---
LEARNING_RATE = 1e-4
EPOCHS = 50 # Increase for real training
BATCH_SIZE = 8  # Adjust based on GPU memory
NUM_ITERATIONS = 8 # Number of refinement steps (T)
KERNEL_SIZE = 15   # Example kernel dimension (N x N, assumed odd)
IMG_CHANNELS = 1  # Grayscale images
HIDDEN_DIM_LSTM = 64 # Channels in ConvLSTM hidden state
IMG_SIZE = 64      # Example image dimension (H x W)
NUM_TRAIN_SAMPLES = 1024 # Number of training images
NUM_VAL_SAMPLES = 128   # Number of validation images

# Loss Weights (CRITICAL - requires careful tuning!)
W_IMG = 1.0       # Weight for image reconstruction loss
W_KER = 0.5       # Weight for kernel reconstruction loss
W_SPARSE = 0.01   # Weight for kernel sparsity (L1)
W_FIDELITY = 1.0  # Weight for Poisson fidelity loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# For mixed precision training (can speed up training and save memory)
USE_AMP = torch.cuda.is_available()

# --- ConvLSTM Cell Implementation ---
# (A common implementation pattern, PyTorch doesn't have a built-in one)
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim, # i, f, o, g gates
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1) # Concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        # Needs FLOAT tensor type
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device, dtype=torch.float),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device, dtype=torch.float))

# --- Differentiable Convolution ---
def apply_conv(image, kernel):
    """Applies convolution using kernel"""
    batch_size, channels, height, width = image.shape
    k_batch, k_channels, k_h, k_w = kernel.shape # k_channels should be 1

    if k_channels != 1 or channels != 1:
         # Basic implementation assumes single channel image and kernel for simplicity
         # For multi-channel, would need grouped convolution or per-channel application
         raise NotImplementedError("Basic apply_conv assumes single channel image/kernel")

    # Ensure kernel is contiguous and shape is (out_ch=1, in_ch=1, H, W)
    kernel = kernel.contiguous().view(batch_size, 1, k_h, k_w)

    # Calculate padding for 'same' size output (assuming odd kernel size)
    padding = k_h // 2

    # Group convolution: each batch element is convolved with its corresponding kernel
    # Reshape image: (1, B*C, H, W)
    # Reshape kernel: (B*C, 1, kH, kW)
    image_grouped = image.view(1, batch_size * channels, height, width)
    kernel_grouped = kernel.view(batch_size * channels, 1, k_h, k_w)

    # Perform grouped convolution
    output = F.conv2d(image_grouped, kernel_grouped, padding=padding, groups=batch_size*channels)

    # Reshape output back to (B, C, H, W)
    output = output.view(batch_size, channels, height, width)

    return output

# --- Model: Iterative Deconvolution Network ---
class IterativeDeconvNet(nn.Module):
    def __init__(self, num_iterations=NUM_ITERATIONS, img_channels=IMG_CHANNELS,
                 kernel_size_out=KERNEL_SIZE, hidden_dim=HIDDEN_DIM_LSTM,
                 lstm_kernel_size=3):
        super().__init__()
        self.T = num_iterations
        self.img_channels = img_channels
        self.kernel_size_out = kernel_size_out
        self.hidden_dim = hidden_dim

        # Simple fixed initialization for kernel (e.g., small Gaussian)
        # Create a Gaussian kernel and unsqueeze for batch/channel dims
        sigma = 1.0
        center = kernel_size_out // 2
        x, y = torch.meshgrid(torch.arange(kernel_size_out), torch.arange(kernel_size_out), indexing='ij')
        dist_sq = (x - center)**2 + (y - center)**2
        gaussian = torch.exp(-dist_sq / (2 * sigma**2))
        gaussian = gaussian / gaussian.sum() # Normalize
        self.register_buffer('init_k', gaussian.view(1, 1, kernel_size_out, kernel_size_out))

        # Feature encoder for the blurry image y (optional, can simplify input to LSTM)
        # Example: Reduce spatial dims slightly, increase channels
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(img_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(2), # Optional downsampling
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # Adjust hidden_dim calculation if maxpooling is used

        # ConvLSTM Cell
        # Input: x_hat_t (img_channels), k_hat_t (1 channel), features_y (hidden_dim)
        lstm_input_dim = self.img_channels + 1 + self.hidden_dim
        self.conv_lstm = ConvLSTMCell(lstm_input_dim, hidden_dim, kernel_size=lstm_kernel_size)

        # Decoders from LSTM hidden state h_t
        # Decoder for image x_hat_t
        self.decoder_x = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Upsample(scale_factor=2), # If feature_encoder downsampled
            nn.Conv2d(hidden_dim, img_channels, kernel_size=1),
            nn.Sigmoid() # Assuming image pixels are normalized [0, 1]
        )

        # Decoder for kernel k_hat_t
        # Output a kernel_size_out x kernel_size_out map
        self.decoder_k = nn.Sequential(
             nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
             nn.ReLU(),
             nn.Conv2d(hidden_dim // 2, 1, kernel_size=1)
             # No activation here, will apply ReLU + Normalization later
        )

    def forward(self, y):
        batch_size, _, H, W = y.shape
        image_size = (H, W)

        # Encode blurry image features (once)
        features_y = self.feature_encoder(y)

        # Initialize LSTM state
        h_t, c_t = self.conv_lstm.init_hidden(batch_size, features_y.shape[2:]) # Use feature map size

        # Initialize estimates
        x_hat_t = torch.sigmoid(y) # Use input image (passed through sigmoid) as init guess for x
        k_hat_t = self.init_k.repeat(batch_size, 1, 1, 1) # Fixed Gaussian init

        outputs_x = []
        outputs_k = []

        for t in range(self.T):
            # Resize kernel to match feature map size if needed (using interpolation)
            k_hat_t_resized = F.interpolate(k_hat_t, size=features_y.shape[2:], mode='bilinear', align_corners=False)
            # Resize x_hat if needed (though likely same size as y features)
            x_hat_t_resized = F.interpolate(x_hat_t, size=features_y.shape[2:], mode='bilinear', align_corners=False)

            # Prepare LSTM input
            lstm_input = torch.cat([x_hat_t_resized, k_hat_t_resized, features_y], dim=1)

            # LSTM step
            h_t, c_t = self.conv_lstm(lstm_input, (h_t, c_t))

            # Decode estimates for this step
            x_hat_t = self.decoder_x(h_t) # Output is H x W
            k_hat_raw = self.decoder_k(h_t) # Output is small, e.g. H_feat x W_feat

            # Upsample kernel output to desired KERNEL_SIZE x KERNEL_SIZE
            k_hat_raw = F.interpolate(k_hat_raw, size=(self.kernel_size_out, self.kernel_size_out), mode='bilinear', align_corners=False)

            # Post-process kernel: Non-negativity + Sum-to-one
            k_hat_t = torch.relu(k_hat_raw) # Non-negativity
            k_sum = torch.sum(k_hat_t, dim=[2, 3], keepdim=True) # Sum over spatial dims
            k_hat_t = k_hat_t / (k_sum + 1e-8) # Normalize (add epsilon for stability)

            outputs_x.append(x_hat_t)
            outputs_k.append(k_hat_t)

        # Return list of estimates for each step
        return outputs_x, outputs_k


# --- Dummy Dataset ---
class BlindDeconvDataset(Dataset):
    def __init__(self, num_samples, img_size, kernel_size, noise_level=1.0, is_val=False):
        self.num_samples = num_samples
        self.img_size = img_size
        self.kernel_size = kernel_size
        self.noise_level = noise_level # Scaling factor related to Poisson intensity
        self.is_val = is_val # Use fixed seed for validation

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.is_val:
            seed = idx # Fixed seed for validation
        else:
            seed = None # Random seed for training

        rng = np.random.RandomState(seed)

        # 1. Generate clean image (simple random pattern for demo)
        clean_image = rng.rand(self.img_size, self.img_size).astype(np.float32) * 0.8 + 0.1 # Range [0.1, 0.9]

        # 2. Generate sparse kernel
        kernel = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float32)
        # Example: few random non-zero positive entries
        num_sparse_entries = rng.randint(3, (self.kernel_size*self.kernel_size)//8 + 3)
        for _ in range(num_sparse_entries):
            r, c = rng.randint(0, self.kernel_size, size=2)
            kernel[r, c] = rng.rand() * 0.5 + 0.5 # Positive values
        # Ensure non-negativity (already done) and sum-to-one
        if kernel.sum() < 1e-6: # Avoid division by zero if all entries were zero
             center = self.kernel_size // 2
             kernel[center, center] = 1.0
        kernel /= kernel.sum()

        # 3. Apply convolution (using numpy for data generation)
        from scipy.signal import convolve2d
        ideal_blurred = convolve2d(clean_image, kernel, mode='same', boundary='wrap') # Use wrap for simplicity

        # 4. Add Poisson noise
        # Scale intensity - higher mean = lower relative noise
        mean_intensity = ideal_blurred * self.noise_level
        noisy_image = rng.poisson(mean_intensity).astype(np.float32)
        # Normalize noisy image back roughly to initial range for network input stability
        noisy_image /= (self.noise_level + 1e-6) # Approximate normalization

        # Clamp noisy image to avoid extreme values after normalization
        noisy_image = np.clip(noisy_image, 0, 1)

        # Convert to PyTorch tensors (add channel dimension)
        clean_image_t = torch.from_numpy(clean_image).unsqueeze(0)
        noisy_image_t = torch.from_numpy(noisy_image).unsqueeze(0)
        kernel_t = torch.from_numpy(kernel).unsqueeze(0)

        return noisy_image_t, clean_image_t, kernel_t


# --- Loss Functions ---
criterion_image = nn.MSELoss() # Or nn.L1Loss()
criterion_kernel = nn.MSELoss() # Or nn.L1Loss()
criterion_sparse = nn.L1Loss() # L1 norm for sparsity
# For PoissonNLLLoss: input should be the *mean* (k*x), target is the noisy observation y
# log_input=False means input is expected rate, not log-rate. Ensure rate > 0.
criterion_fidelity = nn.PoissonNLLLoss(log_input=False, reduction='mean', eps=1e-8)

# --- Training Loop ---
def train_one_epoch(model, dataloader, optimizer, scaler, device):
    model.train()
    total_loss_epoch = 0.0

    for batch_idx, (y_batch, x_true_batch, k_true_batch) in enumerate(dataloader):
        y_batch = y_batch.to(device)
        x_true_batch = x_true_batch.to(device)
        k_true_batch = k_true_batch.to(device)

        optimizer.zero_grad()

        # Use AMP context manager
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            list_x_hat, list_k_hat = model(y_batch)

            total_loss_batch = 0
            # Calculate loss for each iteration step
            for t in range(model.T):
                x_hat_t = list_x_hat[t]
                k_hat_t = list_k_hat[t]

                # Re-blur estimate for fidelity loss
                # Ensure non-negative input for PoissonNLLLoss
                reblurred_t = apply_conv(x_hat_t, k_hat_t).clamp(min=1e-8) # Add epsilon for numerical stability

                loss_img_t = criterion_image(x_hat_t, x_true_batch)
                loss_ker_t = criterion_kernel(k_hat_t, k_true_batch)
                loss_sparse_t = criterion_sparse(k_hat_t, torch.zeros_like(k_hat_t)) # L1 norm vs zero
                loss_fid_t = criterion_fidelity(reblurred_t, y_batch)

                step_loss = (W_IMG * loss_img_t + W_KER * loss_ker_t +
                             W_SPARSE * loss_sparse_t + W_FIDELITY * loss_fid_t)

                total_loss_batch += step_loss

            # Average loss over iterations
            total_loss_batch = total_loss_batch / model.T

        # Scaler scales losses for gradient computation
        scaler.scale(total_loss_batch).backward()

        # Scaler used to unscale gradients before optimizer steps
        scaler.step(optimizer)

        # Updates the scale for next iteration
        scaler.update()

        total_loss_epoch += total_loss_batch.item()

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Avg Loss: {total_loss_batch.item():.4f}")

    return total_loss_epoch / len(dataloader)

# --- Validation Loop ---
def validate(model, dataloader, device):
    model.eval()
    total_val_loss = 0.0
    total_psnr = 0.0
    num_batches = 0

    with torch.no_grad():
        for y_batch, x_true_batch, k_true_batch in dataloader:
            y_batch = y_batch.to(device)
            x_true_batch = x_true_batch.to(device)
            k_true_batch = k_true_batch.to(device)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                list_x_hat, list_k_hat = model(y_batch)

                # Evaluate based on the *final* iteration
                x_hat_final = list_x_hat[-1]
                k_hat_final = list_k_hat[-1]
                reblurred_final = apply_conv(x_hat_final, k_hat_final).clamp(min=1e-8)

                loss_img = criterion_image(x_hat_final, x_true_batch)
                loss_ker = criterion_kernel(k_hat_final, k_true_batch)
                loss_sparse = criterion_sparse(k_hat_final, torch.zeros_like(k_hat_final))
                loss_fid = criterion_fidelity(reblurred_final, y_batch)

                val_loss = (W_IMG * loss_img + W_KER * loss_ker +
                            W_SPARSE * loss_sparse + W_FIDELITY * loss_fid)

            total_val_loss += val_loss.item()

            # Calculate PSNR (example metric)
            mse = F.mse_loss(x_hat_final, x_true_batch)
            if mse > 0:
                psnr = 20 * math.log10(1.0 / math.sqrt(mse)) # Assuming max pixel value is 1.0
                total_psnr += psnr
            num_batches += 1

    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0
    avg_psnr = total_psnr / num_batches if num_batches > 0 else 0
    return avg_val_loss, avg_psnr


# --- Main Execution ---
if __name__ == "__main__":
    # --- Data Loading ---
    train_dataset = BlindDeconvDataset(NUM_TRAIN_SAMPLES, IMG_SIZE, KERNEL_SIZE, is_val=False)
    val_dataset = BlindDeconvDataset(NUM_VAL_SAMPLES, IMG_SIZE, KERNEL_SIZE, is_val=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val samples.")

    # --- Model & Optimizer ---
    model = IterativeDeconvNet(
        num_iterations=NUM_ITERATIONS,
        img_channels=IMG_CHANNELS,
        kernel_size_out=KERNEL_SIZE,
        hidden_dim=HIDDEN_DIM_LSTM
    ).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP) # AMP scaler
    print(f"Model created. Parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # --- Training ---
    best_val_psnr = -1.0
    print("Starting training...")
    for epoch in range(EPOCHS):
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, scaler, DEVICE)
        avg_val_loss, avg_val_psnr = validate(model, val_loader, DEVICE)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val PSNR: {avg_val_psnr:.2f} dB")

        # Save best model based on validation PSNR
        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            save_path = "best_deconv_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved to {save_path} (PSNR: {best_val_psnr:.2f} dB)")

    print("Training finished.")

    # --- Example Inference & Visualization ---
    print("\nRunning inference on one validation sample...")
    model.load_state_dict(torch.load("best_deconv_model.pth", map_location=DEVICE))
    model.eval()

    y_inf, x_true_inf, k_true_inf = val_dataset[0] # Get first validation sample
    y_inf = y_inf.unsqueeze(0).to(DEVICE) # Add batch dimension

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=USE_AMP):
         list_x_hat_inf, list_k_hat_inf = model(y_inf)

    # Visualize results from the final iteration
    x_hat_final_inf = list_x_hat_inf[-1].squeeze().cpu().numpy()
    k_hat_final_inf = list_k_hat_inf[-1].squeeze().cpu().numpy()
    y_inf_np = y_inf.squeeze().cpu().numpy()
    x_true_inf_np = x_true_inf.squeeze().cpu().numpy()
    k_true_inf_np = k_true_inf.squeeze().cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Example Deconvolution Result (Final Iteration)')

    im_opts = {'cmap': 'gray', 'vmin': 0, 'vmax': 1}
    ker_opts = {'cmap': 'viridis'}

    axes[0, 0].imshow(y_inf_np, **im_opts)
    axes[0, 0].set_title('Input Blurry (y)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(x_hat_final_inf, **im_opts)
    axes[0, 1].set_title('Estimated Sharp (x_hat)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(x_true_inf_np, **im_opts)
    axes[0, 2].set_title('Ground Truth Sharp (x)')
    axes[0, 2].axis('off')

    axes[1, 0].axis('off') # Empty space

    axes[1, 1].imshow(k_hat_final_inf, **ker_opts)
    axes[1, 1].set_title('Estimated Kernel (k_hat)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(k_true_inf_np, **ker_opts)
    axes[1, 2].set_title('Ground Truth Kernel (k)')
    axes[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.savefig("deconvolution_result.png")
    print("Saved visualization to deconvolution_result.png")
    # plt.show() # Uncomment to display plot
