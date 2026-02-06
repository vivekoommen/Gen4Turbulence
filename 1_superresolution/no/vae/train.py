# ---------------------------------------------------------------------------------------------
# Author: Aniruddha Bora
# Date: 09/01/2025
# ---------------------------------------------------------------------------------------------

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model import VQVAE
import torch.nn as nn

import os
import sys
import time
import matplotlib.pyplot as plt

DTYPE = torch.float32

CMAP = "gray"

def compute_power(true, pred, inp):
    BS, nt, nx, ny = true.shape
    
    # Compute the Fourier transforms and amplitude squared for both true and pred
    fourier_true = torch.fft.fftn(true, dim=(-2, -1))
    fourier_pred = torch.fft.fftn(pred, dim=(-2, -1))
    fourier_inp  = torch.fft.fftn(inp , dim=(-2, -1))

    # Get the squared amplitudes
    amplitudes_true = torch.abs(fourier_true) #** 2
    amplitudes_pred = torch.abs(fourier_pred) #** 2
    amplitudes_inp = torch.abs(fourier_inp) #** 2

    # Create the k-frequency grids
    kfreq_y = torch.fft.fftfreq(ny) * ny
    kfreq_x = torch.fft.fftfreq(nx) * nx
    kfreq2D_x, kfreq2D_y = torch.meshgrid(kfreq_x, kfreq_y, indexing='ij')
    
    # Compute the wavenumber grid
    knrm = torch.sqrt(kfreq2D_x ** 2 + kfreq2D_y ** 2).to(true.device)
    
    # Define the bins for the wavenumber
    kbins = torch.arange(0.5, nx // 2 + 1, 1.0, device=true.device)
    
    # Digitize knrm to bin indices
    knrm_flat = knrm.flatten()
    bin_indices = torch.bucketize(knrm_flat, kbins)

    # Reshape and flatten the amplitudes
    amplitudes_true_flat = amplitudes_true.view(BS, nt, nx * ny)
    amplitudes_pred_flat = amplitudes_pred.view(BS, nt, nx * ny)
    amplitudes_inp_flat  = amplitudes_inp.view(BS, nt, nx * ny)

    # Initialize Abins
    Abins_true = torch.zeros((BS, nt, len(kbins) - 1), device=true.device)
    Abins_pred = torch.zeros((BS, nt, len(kbins) - 1), device=pred.device)
    Abins_inp  = torch.zeros((BS, nt, len(kbins) - 1), device= inp.device)

    # Vectorized binning: sum up the values in each bin
    for bin_idx in range(1, len(kbins)):
        mask = (bin_indices == bin_idx).unsqueeze(0).unsqueeze(0)  # Create a mask for each bin
        Abins_true[:, :, bin_idx - 1] = (amplitudes_true_flat * mask).sum(dim=-1) / mask.sum(dim=-1)
        Abins_pred[:, :, bin_idx - 1] = (amplitudes_pred_flat * mask).sum(dim=-1) / mask.sum(dim=-1)
        Abins_inp[:,  :, bin_idx - 1] = (amplitudes_inp_flat  * mask).sum(dim=-1) / mask.sum(dim=-1)

    # Scale the binned amplitudes
    scaling_factor = torch.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
    Abins_true *= scaling_factor
    Abins_pred *= scaling_factor
    Abins_inp  *= scaling_factor

    return Abins_true, Abins_pred, Abins_inp

def plot_power_spectrum(power_inp, power_true, power_pred, inp, true, pred, epoch, err):
    f = 2
    fig, axes = plt.subplots(1, 4, figsize=(4*f, 1*f))
    
    sample_id=8
    t_id = 0
    for i in range(1):
        x = torch.arange(true.shape[-2]//2)
        axes[0].loglog(x, power_true[sample_id,t_id], label='true', c='black')
        axes[0].loglog(x, power_inp[sample_id,t_id], label='NO', c='blue')
        axes[0].loglog(x, power_pred[sample_id,t_id], label='adv. NO', c='red')
        axes[0].set_xlabel(r'$k$')
        if i==0:
            axes[0].legend()
        if i==0:
            axes[0].set_ylabel(r'$P(k)$')
    

    inp_sample = inp[sample_id, t_id]
    true_sample = true[sample_id, t_id]
    pred_sample = pred[sample_id, t_id]
    vmin, vmax = true_sample.min(), true_sample.max()
    im1 = axes[1].imshow(true_sample, vmin=vmin, vmax=vmax, cmap=CMAP)
    axes[1].set_title("True")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    im = axes[2].imshow(inp_sample, vmin=vmin, vmax=vmax, cmap=CMAP)
    axes[2].set_title("NO")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    im = axes[3].imshow(pred_sample, vmin=vmin, vmax=vmax, cmap=CMAP)
    axes[3].set_title("NO+VAE")
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    fig.colorbar(im1, ax=axes[3])
    plt.tight_layout()


    fig.suptitle(f"Epoch: {epoch}, MSE: {err:.2e}", fontsize=22, y=1.2)
    plt.savefig(f"power_spectrum/{epoch}.png", dpi=150, bbox_inches='tight')
    plt.close()

def error_metric(inp, pred,true, epoch, Par={}, is_plot=True):
    power_inp, power_true, power_pred = compute_power(inp, true, pred)
    err = torch.mean( (torch.log(power_true)-torch.log(power_pred) )**2 )
    f_err = torch.norm(true-pred, p=2)/torch.norm(true, p=2)
    ref_err = torch.norm(true-inp, p=2)/torch.norm(true, p=2)
    if is_plot:
        plot_power_spectrum(power_inp.detach().cpu().numpy(), power_true.detach().cpu().numpy(), power_pred.detach().cpu().numpy(), inp.detach().cpu().numpy(), true.detach().cpu().numpy(), pred.detach().cpu().numpy(), epoch, err)
    return err, f_err, ref_err

######################## VO input #############################

# Custom Dataset
class SuperResDataset(Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = torch.tensor(input_data, dtype=torch.float32)
        self.target_data = torch.tensor(target_data, dtype=torch.float32)

        print(f"input_data : {self.input_data.shape}")
        print(f"target_data: {self.target_data.shape}")

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx]


# Training Function
def train(model, train_loader, val_loader, optimizer, num_epochs, device, save_path="best_flow_model.pth"):
    criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()
    model.to(device)

    best_loss = float('inf')

    for epoch in range(num_epochs):
        begin_time = time.time()
        model.train()
        total_train_loss = 0

        for x_lr, x_hr in train_loader:
            x_lr, x_hr = x_lr.to(device), x_hr.to(device)
            optimizer.zero_grad()
            outputs, vq_loss = model(x_lr)
            recon_loss = recon_criterion(outputs, x_hr)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation Step
        model.eval()
        total_val_loss = 0
        spec_err = 0.0
        field_err = 0.0
        ref_err = 0.0
        plot_flag = True
        with torch.no_grad():
            for x_lr, x_hr in val_loader:
                x_lr, x_hr = x_lr.to(device), x_hr.to(device)
                outputs, vq_loss = model(x_lr)
                recon_loss = recon_criterion(outputs, x_hr)
                loss = recon_loss + vq_loss
                
                total_val_loss += loss.item()

                s_err, f_err, r_err = error_metric(x_lr, outputs, x_hr, epoch+1, is_plot=plot_flag)
                spec_err += s_err.item()
                field_err += f_err.item()
                ref_err += r_err.item()
                plot_flag = False

        avg_val_loss = total_val_loss / len(val_loader)
        spec_err /= len(val_loader)
        field_err /= len(val_loader)
        ref_err /= len(val_loader)

        if spec_err < best_loss:
            best_loss = spec_err
            best_model_id = epoch
            torch.save(model.state_dict(), f'Params/best_model.pt')

        elapsed_time = time.time() - begin_time
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4e} | Val Loss: {avg_val_loss:.4e} | spec err: {spec_err:.4e} | field err: {field_err:.4e} | ref err: {ref_err:.4e}, best model: {best_model_id}, epoch time: {elapsed_time:.2f}")
        

def load_data():
    train_inputs = np.load('../TRAIN_PRED.npy',allow_pickle=True)  # Shape: (BS, Nt, Nx, Ny)
    B,T,X,Y = train_inputs.shape
    train_inputs = train_inputs.reshape(-1,1,X,Y)

    train_targets = np.load('../TRAIN_TRUE.npy',allow_pickle=True)  # Shape: (BS, Nt, Nx, Ny)
    train_targets = train_targets.reshape(-1,1,X,Y)

    val_inputs = np.load('../VAL_PRED.npy',allow_pickle=True)  # Shape: (BS, Nt, Nx, Ny)
    val_inputs = val_inputs.reshape(-1,1,X,Y)

    val_targets = np.load('../VAL_TRUE.npy',allow_pickle=True) 
    val_targets = val_targets.reshape(-1,1,X,Y)
    
    return train_inputs, train_targets, val_inputs, val_targets

# Main Training Loop
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_inputs, train_targets, val_inputs, val_targets = load_data()

    print(f"train_inputs: {train_inputs.shape}")
    print(f"train_targets: {train_targets.shape}")
    print(f"val_inputs: {val_inputs.shape}")
    print(f"val_targets: {val_targets.shape}")

    os.makedirs("Params", exist_ok=True)
    os.makedirs("power_spectrum", exist_ok=True)

    
    # Create DataLoaders
    batch_size = 20

    print(f"Train Dataset prep")
    train_dataset = SuperResDataset(train_inputs, train_targets)
    print(f"Val Dataset prep")
    val_dataset = SuperResDataset(val_inputs, val_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # Initialize model and optimizer
    model = VQVAE(in_channels=1, hidden_channels=21, embedding_dim=64, num_embeddings=256, commitment_cost=0.25)
    total_params = sum(param.numel() for param in model.parameters())

    print(f"Total number of parameters: {total_params}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    # If you want *all* parameters including non-trainable:
    num_all_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters (trainable + non-trainable): {num_all_params}")

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    train(model, train_loader, val_loader, optimizer, num_epochs=100000, device=device)

    print("Training completed! Best model saved as 'best_flow_model.pth'.")
