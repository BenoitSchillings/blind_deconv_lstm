import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
from glob import glob # To find files easily
from astropy.io import fits # To read FITS files
from scipy.signal import convolve2d # For data generation convolution
import matplotlib.pyplot as plt # For visualization
import random

# --- Hyperparameters ---
LEARNING_RATE = 1e-4
EPOCHS = 50 # Increase for real training
BATCH_SIZE = 4  # Adjust based on GPU memory (256x256 images need more mem)
NUM_ITERATIONS = 8 # Number of refinement steps (T)
KERNEL_SIZE = 15   # Kernel dimension (N x N, fixed)
IMG_CHANNELS = 1  # Grayscale images (assuming FITS are single plane)
HIDDEN_DIM_LSTM = 64 # Channels in ConvLSTM hidden state
IMG_SIZE = 256     # Image dimension (H x W)
# Noise level: Represents a scaling factor for the mean in Poisson distribution
# Adjust based on your typical signal levels
POISSON_NOISE_FACTOR = 50.0

# Data paths (MODIFY THESE)
IMAGE_DIR = "./training_images/"
KERNEL_DIR = "./kernels/"
OUTPUT_DIR = "./output/" # To save models and plots

# Train/Validation Split
VAL_SPLIT_RATIO = 0.1 # Use 10% of data for validation

# Loss Weights (CRITICAL - requires careful tuning!)
W_IMG = 1.0       # Weight for image reconstruction loss
W_KER = 0.5       # Weight for kernel reconstruction loss
W_SPARSE = 0.01   # Weight for kernel sparsity (L1)
W_FIDELITY = 1.0  # Weight for Poisson fidelity loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# For mixed precision training (can speed up training and save memory)
USE_AMP = torch.cuda.is_available()

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- ConvLSTM Cell Implementation (Unchanged) ---
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
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device, dtype=torch.float),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device, dtype=torch.float))

# --- Differentiable Convolution (Unchanged) ---
def apply_conv(image, kernel):
    batch_size, channels, height, width = image.shape
    k_batch, k_channels, k_h, k_w = kernel.shape

    if k_channels != 1 or channels != 1:
         raise NotImplementedError("Basic apply_conv assumes single channel image/kernel")

    kernel = kernel.contiguous().view(batch_size, 1, k_h, k_w)
    padding = k_h // 2
    image_grouped = image.view(1, batch_size * channels, height, width)
    kernel_grouped = kernel.view(batch_size * channels, 1, k_h, k_w)
    output = F.conv2d(image_grouped, kernel_grouped, padding=padding, groups=batch_size*channels)
    output = output.view(batch_size, channels, height, width)
    return output

# --- Model: Iterative Deconvolution Network (Unchanged structure, only input/output sizes differ) ---
class IterativeDeconvNet(nn.Module):
    def __init__(self, num_iterations=NUM_ITERATIONS, img_channels=IMG_CHANNELS,
                 kernel_size_out=KERNEL_SIZE, hidden_dim=HIDDEN_DIM_LSTM,
                 lstm_kernel_size=3):
        super().__init__()
        self.T = num_iterations
        self.img_channels = img_channels
        self.kernel_size_out = kernel_size_out
        self.hidden_dim = hidden_dim

        # Kernel initialization moved to Dataset/forward pass if needed

        # Feature encoder for the blurry image y
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(img_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # ConvLSTM Cell
        lstm_input_dim = self.img_channels + 1 + self.hidden_dim
        self.conv_lstm = ConvLSTMCell(lstm_input_dim, hidden_dim, kernel_size=lstm_kernel_size)

        # Decoders from LSTM hidden state h_t
        self.decoder_x = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, img_channels, kernel_size=1),
            nn.Sigmoid() # Output image in [0, 1]
        )
        self.decoder_k = nn.Sequential(
             nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
             nn.ReLU(),
             # Use AdaptiveAvgPool2d to reduce spatial dimensions to 1x1 before linear layer
             nn.AdaptiveAvgPool2d((1, 1)),
             nn.Flatten(),
             nn.Linear(hidden_dim // 2, kernel_size_out * kernel_size_out) # Output N*N values
        )

    def forward(self, y):
        batch_size, _, H, W = y.shape
        image_size = (H, W) # Will be 256x256

        # Encode blurry image features (once)
        features_y = self.feature_encoder(y)
        features_size = features_y.shape[2:] # Get size after encoding

        # Initialize LSTM state
        h_t, c_t = self.conv_lstm.init_hidden(batch_size, features_size)

        # Initialize estimates
        x_hat_t = torch.sigmoid(y) # Use input image (passed through sigmoid) as init guess for x
        # Initialize kernel (e.g., delta function in center)
        init_k_np = np.zeros((self.kernel_size_out, self.kernel_size_out), dtype=np.float32)
        init_k_np[self.kernel_size_out // 2, self.kernel_size_out // 2] = 1.0
        k_hat_t = torch.from_numpy(init_k_np).view(1, 1, self.kernel_size_out, self.kernel_size_out).repeat(batch_size, 1, 1, 1).to(y.device)

        outputs_x = []
        outputs_k = []

        for t in range(self.T):
            # Resize kernel/image to match feature map size for LSTM input
            k_hat_t_resized = F.interpolate(k_hat_t, size=features_size, mode='bilinear', align_corners=False)
            x_hat_t_resized = F.interpolate(x_hat_t, size=features_size, mode='bilinear', align_corners=False)

            # Prepare LSTM input
            lstm_input = torch.cat([x_hat_t_resized, k_hat_t_resized, features_y], dim=1)

            # LSTM step
            h_t, c_t = self.conv_lstm(lstm_input, (h_t, c_t))

            # Decode estimates for this step
            x_hat_t = self.decoder_x(h_t) # Output is H x W (256x256)

            # Decode kernel - outputs N*N values, needs reshaping
            k_hat_flat = self.decoder_k(h_t)
            k_hat_raw = k_hat_flat.view(batch_size, 1, self.kernel_size_out, self.kernel_size_out)

            # Post-process kernel: Non-negativity + Sum-to-one
            k_hat_t = torch.relu(k_hat_raw) # Non-negativity
            k_sum = torch.sum(k_hat_t, dim=[2, 3], keepdim=True) # Sum over spatial dims
            k_hat_t = k_hat_t / (k_sum + 1e-8) # Normalize (add epsilon for stability)

            outputs_x.append(x_hat_t)
            outputs_k.append(k_hat_t)

        # Return list of estimates for each step
        return outputs_x, outputs_k

# --- Dataset using FITS files ---
class FitsBlindDeconvDataset(Dataset):
    def __init__(self, image_dir, kernel_dir, file_indices, noise_factor=POISSON_NOISE_FACTOR, kernel_size=KERNEL_SIZE, is_val=False):
        """
        Args:
            image_dir (str): Directory containing clean FITS images.
            kernel_dir (str): Directory containing FITS kernels.
            file_indices (list): List of integer indices to use for images in this dataset split.
            noise_factor (float): Scaling factor for Poisson noise generation.
            kernel_size (int): Dimension of the kernel (e.g., 15).
            is_val (bool): If True, use fixed random seed for kernel selection.
        """
        self.image_dir = image_dir
        self.kernel_dir = kernel_dir
        self.file_indices = file_indices
        self.noise_factor = noise_factor
        self.kernel_size = kernel_size
        self.is_val = is_val

        self.image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith('.fits')])
        self.kernel_files = sorted([os.path.join(kernel_dir, f) for f in os.listdir(kernel_dir) if f.lower().endswith('.fits')])

        if not self.image_files:
            raise FileNotFoundError(f"No FITS files found in image directory: {image_dir}")
        if not self.kernel_files:
            raise FileNotFoundError(f"No FITS files found in kernel directory: {kernel_dir}")

        self.indexed_image_files = [self.image_files[i] for i in file_indices]

        # Seed for consistent validation kernel selection
        if self.is_val:
            self.rng = np.random.RandomState(42) # Fixed seed for validation set
        else:
            self.rng = np.random.RandomState() # Default random seed

    def __len__(self):
        return len(self.indexed_image_files)

    def __getitem__(self, idx):
        # 1. Load clean image
        image_path = self.indexed_image_files[idx]
        try:
            # Use memmap=False for safety if files are modified, getdata reads primary HDU
            with fits.open(image_path, memmap=False) as hdul:
                clean_image = hdul[0].data # Assuming data is in the primary HDU
                if clean_image is None:
                     raise ValueError(f"Primary HDU data is None in {image_path}")
                # Convert to float32 and handle potential NaNs/Infs
                clean_image = np.nan_to_num(clean_image.astype(np.float32))
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return dummy data or raise error
            return torch.zeros(1, IMG_SIZE, IMG_SIZE), torch.zeros(1, IMG_SIZE, IMG_SIZE), torch.zeros(1, KERNEL_SIZE, KERNEL_SIZE)

        # --- Image Normalization (EXAMPLE - ADJUST AS NEEDED) ---
        # Simple min-max scaling to [0, 1] per image
        min_val = np.min(clean_image)
        max_val = np.max(clean_image)
        if max_val > min_val:
            clean_image = (clean_image - min_val) / (max_val - min_val)
        else:
            clean_image = np.zeros_like(clean_image) # Handle constant image
        clean_image = np.clip(clean_image, 0, 1) # Ensure range
        # -----------------------------------------------------

        # 2. Load random kernel
        # Seed kernel selection for validation consistency
        if self.is_val:
            kernel_idx = self.rng.randint(0, len(self.kernel_files))
        else:
            kernel_idx = random.randint(0, len(self.kernel_files) - 1) # Use python's random for training

        kernel_path = self.kernel_files[kernel_idx]
        try:
            with fits.open(kernel_path, memmap=False) as hdul:
                kernel = hdul[0].data
                if kernel is None:
                     raise ValueError(f"Primary HDU data is None in {kernel_path}")
                # Ensure kernel size matches (optional, maybe resize/crop)
                if kernel.shape != (self.kernel_size, self.kernel_size):
                    # Example: center crop or pad if needed - requires more logic
                    print(f"Warning: Kernel {kernel_path} shape {kernel.shape} differs from expected ({self.kernel_size},{self.kernel_size}). Skipping sample.")
                    # Return dummy data or implement robust resizing/padding
                    return torch.zeros(1, IMG_SIZE, IMG_SIZE), torch.zeros(1, IMG_SIZE, IMG_SIZE), torch.zeros(1, KERNEL_SIZE, KERNEL_SIZE)
                kernel = np.nan_to_num(kernel.astype(np.float32))
        except Exception as e:
            print(f"Error loading kernel {kernel_path}: {e}")
            return torch.zeros(1, IMG_SIZE, IMG_SIZE), torch.zeros(1, IMG_SIZE, IMG_SIZE), torch.zeros(1, KERNEL_SIZE, KERNEL_SIZE)


        # --- Kernel Normalization ---
        kernel = np.maximum(0, kernel) # Enforce non-negativity
        kernel_sum = kernel.sum()
        if kernel_sum < 1e-8:
             # Avoid division by zero, maybe replace with delta?
             center = self.kernel_size // 2
             kernel = np.zeros_like(kernel)
             kernel[center, center] = 1.0
        else:
             kernel /= kernel_sum # Enforce sum-to-one
        # --------------------------

        # 3. Apply convolution
        # Assuming image shape (H, W), kernel shape (kH, kW)
        ideal_blurred = convolve2d(clean_image, kernel, mode='same', boundary='wrap') # Or 'reflect'
        ideal_blurred = np.maximum(0, ideal_blurred) # Result should be non-negative rate

        # 4. Add Poisson noise
        mean_photons = ideal_blurred * self.noise_factor # Scale to mean counts
        noisy_photons = self.rng.poisson(mean_photons).astype(np.float32)

        # Normalize noisy image for network input (dividing by noise factor)
        # This makes the network target independent of noise_factor scaling
        noisy_image = noisy_photons / (self.noise_factor + 1e-8)
        noisy_image = np.clip(noisy_image, 0, np.max(noisy_image)) # Simple clip, maybe adaptive scaling needed

        # --- Network Input Normalization (EXAMPLE) ---
        # Scale noisy image to approx [0, 1] based on its max - adjust if needed!
        max_noisy = np.max(noisy_image)
        if max_noisy > 1e-6:
            noisy_image = noisy_image / max_noisy
        noisy_image = np.clip(noisy_image, 0, 1)
        # --------------------------------------------

        # Convert to PyTorch tensors (add channel dimension)
        # Ensure IMG_CHANNELS matches the dimension added (usually 1 for grayscale)
        clean_image_t = torch.from_numpy(clean_image).unsqueeze(0)
        noisy_image_t = torch.from_numpy(noisy_image).unsqueeze(0)
        kernel_t = torch.from_numpy(kernel).unsqueeze(0)

        return noisy_image_t, clean_image_t, kernel_t


# --- Loss Functions (Unchanged) ---
criterion_image = nn.MSELoss() # Or nn.L1Loss()
criterion_kernel = nn.MSELoss() # Or nn.L1Loss()
criterion_sparse = nn.L1Loss() # L1 norm for sparsity
# For PoissonNLLLoss: input should be the *mean* (k*x), target is the noisy observation y
# IMPORTANT: Target (y_batch) and Input (reblurred_t) should ideally represent
# expected photon counts for PoissonNLLLoss. If they are normalized [0,1],
# the loss is mathematically less correct but might still work as a surrogate.
# Here we use the normalized versions for simplicity. Add epsilon for stability.
criterion_fidelity = nn.PoissonNLLLoss(log_input=False, reduction='mean', eps=1e-8)


# --- Training Loop (Unchanged structure) ---
def train_one_epoch(model, dataloader, optimizer, scaler, device):
    model.train()
    total_loss_epoch = 0.0

    for batch_idx, (y_batch, x_true_batch, k_true_batch) in enumerate(dataloader):
        y_batch = y_batch.to(device)
        x_true_batch = x_true_batch.to(device)
        k_true_batch = k_true_batch.to(device)

        optimizer.zero_grad(set_to_none=True) # More efficient zeroing

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            list_x_hat, list_k_hat = model(y_batch)

            total_loss_batch = 0
            for t in range(model.T):
                x_hat_t = list_x_hat[t]
                k_hat_t = list_k_hat[t]

                # Re-blur estimate - input needs to be non-negative
                reblurred_t = apply_conv(x_hat_t, k_hat_t).clamp(min=1e-8)

                loss_img_t = criterion_image(x_hat_t, x_true_batch)
                loss_ker_t = criterion_kernel(k_hat_t, k_true_batch)
                # L1 sparsity encourages kernel values towards 0
                loss_sparse_t = criterion_sparse(k_hat_t, torch.zeros_like(k_hat_t))
                # Using normalized y_batch and reblurred_t. Ensure y_batch is also >= 0
                loss_fid_t = criterion_fidelity(reblurred_t, y_batch.clamp(min=0))

                step_loss = (W_IMG * loss_img_t + W_KER * loss_ker_t +
                             W_SPARSE * loss_sparse_t + W_FIDELITY * loss_fid_t)

                total_loss_batch += step_loss

            total_loss_batch = total_loss_batch / model.T

        scaler.scale(total_loss_batch).backward()
        # Optional: Gradient clipping can help stabilize training
        # scaler.unscale_(optimizer) # Unscales the gradients of optimizer's assigned params in-place
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss_epoch += total_loss_batch.item()

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Avg Loss: {total_loss_batch.item():.4f}")

    return total_loss_epoch / len(dataloader)

# --- Validation Loop (Unchanged structure) ---
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
                x_hat_final = list_x_hat[-1]
                k_hat_final = list_k_hat[-1]
                reblurred_final = apply_conv(x_hat_final, k_hat_final).clamp(min=1e-8)

                loss_img = criterion_image(x_hat_final, x_true_batch)
                loss_ker = criterion_kernel(k_hat_final, k_true_batch)
                loss_sparse = criterion_sparse(k_hat_final, torch.zeros_like(k_hat_final))
                loss_fid = criterion_fidelity(reblurred_final, y_batch.clamp(min=0))

                val_loss = (W_IMG * loss_img + W_KER * loss_ker +
                            W_SPARSE * loss_sparse + W_FIDELITY * loss_fid)

            total_val_loss += val_loss.item()

            mse = F.mse_loss(x_hat_final, x_true_batch)
            if mse.item() > 1e-10: # Avoid log(0)
                 # Assuming max intensity is 1.0 due to normalization
                psnr = 20 * math.log10(1.0 / math.sqrt(mse.item()))
                total_psnr += psnr
            # Handle case where MSE is zero or near-zero? (perfect reconstruction)
            elif mse.item() <= 1e-10:
                 total_psnr += 100 # Assign a high PSNR value for perfect reconstruction

            num_batches += 1

    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0
    avg_psnr = total_psnr / num_batches if num_batches > 0 else 0
    return avg_val_loss, avg_psnr


# --- Main Execution ---
if __name__ == "__main__":
    # --- Data Loading Setup ---
    all_image_files = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.fits')])
    num_total_images = len(all_image_files)
    if num_total_images == 0:
        raise FileNotFoundError(f"No FITS images found in {IMAGE_DIR}")

    indices = list(range(num_total_images))
    random.shuffle(indices) # Shuffle indices for random split

    split_point = int(num_total_images * (1 - VAL_SPLIT_RATIO))
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    print(f"Total images found: {num_total_images}")
    print(f"Using {len(train_indices)} for training, {len(val_indices)} for validation.")

    train_dataset = FitsBlindDeconvDataset(IMAGE_DIR, KERNEL_DIR, train_indices, is_val=False)
    val_dataset = FitsBlindDeconvDataset(IMAGE_DIR, KERNEL_DIR, val_indices, is_val=True)

    # Consider persistent_workers=True if data loading is a bottleneck
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print("Dataloaders created.")

    # --- Model & Optimizer ---
    model = IterativeDeconvNet(
        num_iterations=NUM_ITERATIONS,
        img_channels=IMG_CHANNELS,
        kernel_size_out=KERNEL_SIZE,
        hidden_dim=HIDDEN_DIM_LSTM
    ).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # Added weight decay
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    print(f"Model created. Parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # --- Training ---
    best_val_psnr = -float('inf') # Initialize correctly
    print("Starting training...")
    for epoch in range(EPOCHS):
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, scaler, DEVICE)
        avg_val_loss, avg_val_psnr = validate(model, val_loader, DEVICE)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val PSNR: {avg_val_psnr:.2f} dB")

        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            save_path = os.path.join(OUTPUT_DIR, "best_deconv_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved to {save_path} (PSNR: {best_val_psnr:.2f} dB)")

    print("Training finished.")

    # --- Example Inference & Visualization ---
    print("\nRunning inference on one validation sample...")
    best_model_path = os.path.join(OUTPUT_DIR, "best_deconv_model.pth")
    if os.path.exists(best_model_path):
        try:
             model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
             model.eval()
             print(f"Loaded best model from {best_model_path}")

             # Get a sample from validation loader
             y_inf_batch, x_true_inf_batch, k_true_inf_batch = next(iter(val_loader))
             y_inf = y_inf_batch[0:1].to(DEVICE) # Take first item, keep batch dim
             x_true_inf = x_true_inf_batch[0] # Get first item tensor
             k_true_inf = k_true_inf_batch[0] # Get first item tensor

             with torch.no_grad(), torch.cuda.amp.autocast(enabled=USE_AMP):
                  list_x_hat_inf, list_k_hat_inf = model(y_inf)

             x_hat_final_inf = list_x_hat_inf[-1].squeeze().cpu().numpy()
             k_hat_final_inf = list_k_hat_inf[-1].squeeze().cpu().numpy()
             y_inf_np = y_inf.squeeze().cpu().numpy()
             x_true_inf_np = x_true_inf.squeeze().cpu().numpy()
             k_true_inf_np = k_true_inf.squeeze().cpu().numpy()

             fig, axes = plt.subplots(2, 3, figsize=(12, 8))
             fig.suptitle('Example Deconvolution Result (Final Iteration)')
             im_opts = {'cmap': 'gray', 'vmin': 0, 'vmax': 1} # Assumes [0,1] normalization
             ker_opts = {'cmap': 'viridis'}

             axes[0, 0].imshow(y_inf_np, **im_opts)
             axes[0, 0].set_title(f'Input Blurry (y) - Max: {np.max(y_inf_np):.2f}')
             axes[0, 0].axis('off')
             axes[0, 1].imshow(x_hat_final_inf, **im_opts)
             axes[0, 1].set_title('Estimated Sharp (x_hat)')
             axes[0, 1].axis('off')
             axes[0, 2].imshow(x_true_inf_np, **im_opts)
             axes[0, 2].set_title('Ground Truth Sharp (x)')
             axes[0, 2].axis('off')
             axes[1, 0].axis('off')
             axes[1, 1].imshow(k_hat_final_inf, **ker_opts)
             axes[1, 1].set_title(f'Estimated Kernel (k_hat) Sum: {np.sum(k_hat_final_inf):.2f}')
             axes[1, 1].axis('off')
             axes[1, 2].imshow(k_true_inf_np, **ker_opts)
             axes[1, 2].set_title(f'Ground Truth Kernel (k) Sum: {np.sum(k_true_inf_np):.2f}')
             axes[1, 2].axis('off')

             plt.tight_layout(rect=[0, 0.03, 1, 0.95])
             plot_path = os.path.join(OUTPUT_DIR, "deconvolution_result.png")
             plt.savefig(plot_path)
             print(f"Saved visualization to {plot_path}")
             # plt.show()

        except Exception as e:
            print(f"Error during inference/visualization: {e}")

    else:
        print("Could not find saved model file for inference.")
