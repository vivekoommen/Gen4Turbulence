# ---------------------------------------------------------------------------------------------
# Author: Vivek Oommen
# Date: 09/01/2025
# ---------------------------------------------------------------------------------------------


import os
import sys

import math
import time
import datetime
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tcunet import Unet3D
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchinfo import summary
import torchprofile
import matplotlib.pyplot as plt
import scipy.stats as stats

import pickle

torch.manual_seed(23)

scaler = GradScaler()

DTYPE = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class CustomLoss(nn.Module):
    def __init__(self, Par):
        super(CustomLoss, self).__init__()
        self.Par = Par

    def forward(self, y_pred, y_true):
        y_true = (y_true - self.Par["out_shift"])/self.Par["out_scale"]
        y_pred = (y_pred - self.Par["out_shift"])/self.Par["out_scale"]
        loss = torch.norm(y_true-y_pred, p=2)/torch.norm(y_true, p=2)
        return loss

class YourDataset_train(Dataset):
    def __init__(self, x, t, y, transform=None):
        self.x = x
        self.t = t
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        t_sample = self.t[idx]
        y_sample = self.y[idx]

        if self.transform:
            x_sample, t_sample, y_sample = self.transform(x_sample, t_sample, y_sample)

        return x_sample, t_sample, y_sample
    
class YourDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        y_sample = self.y[idx]

        if self.transform:
            x_sample, y_sample = self.transform(x_sample, y_sample)

        return x_sample, y_sample


def preprocess_train(traj, Par):
    nsamples = traj.shape[0]
    nt = traj.shape[1]
    temp = nt - Par['lb'] - Par['lf'] + 1
    x_idx = np.arange(temp).reshape(-1,1)
    x_idx = np.tile(x_idx, (1, Par['lf'])).reshape(-1,1)

    x_idx_ls = []
    for i in range(Par["lb"]):
        x_idx_ls.append(x_idx+i)
    x_idx = np.concatenate(x_idx_ls, axis=1)

    t_idx = np.arange(Par['lf']).reshape(1,-1)

    t_idx = np.tile(t_idx, (temp,1)).reshape(-1,)

    y_idx = np.arange(nt)
    y_idx = sliding_window_view(y_idx[Par['lb']:], window_shape=Par['lf']).reshape(-1,)

    print(f"x_idx: {x_idx.shape}")
    print(f"t_idx: {t_idx.shape}")
    print(f"y_idx: {y_idx.shape}")

    return x_idx, t_idx, y_idx

def preprocess(traj, Par):
    nsamples = traj.shape[0]
    nt = traj.shape[1]
    temp = nt - Par['lb'] - Par['LF'] + 1
    x_idx = np.arange(temp).reshape(-1,1)

    x_idx_ls = []
    for i in range(Par["lb"]):
        x_idx_ls.append(x_idx+i)
    x_idx = np.concatenate(x_idx_ls, axis=1)

    t_idx = np.arange(Par['lf']).reshape(-1,)

    y_idx = np.arange(nt)
    y_idx = sliding_window_view(y_idx[Par['lb']:], window_shape=Par['LF'])#.reshape(-1,)

    print(f"x_idx: {x_idx.shape}")
    print(f"t_idx: {t_idx.shape}")
    print(f"y_idx: {y_idx.shape}")

    return x_idx, t_idx, y_idx


def combined_scheduler(optimizer, total_epochs, warmup_epochs, last_epoch=-1):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def rollout(model, x,t,NT, Par, batch_size):
    # x - [bs, lb, nf, nx,ny]
    # t - [lf,]
    # NT - length of target time-series


    y_pred_ls = []

    bs = batch_size
    end= bs
    while True:
        start = end-bs
        out_ls = []
        
        temp_x1 = x[start:end] #[BS, lb, nf, nx,ny]
        out_ls = [temp_x1.to(device)]
        traj = torch.cat(out_ls, dim=1)

        while traj.shape[1] < NT:
            # model.eval()
            with torch.no_grad():
                temp_x = torch.repeat_interleave(temp_x1, Par['lf'], dim=0) #[BS*lf, lb, nf, nx,ny]
                temp_t = t.repeat(traj.shape[0]) #[BS*lf, ]
                if True:
                    out = model(temp_x.to(device), temp_t.to(device)).reshape(-1,Par['lf'], Par['nf'],Par['nx'],Par['ny'], Par['nz']) #[BS, lf, nf, nx,ny]
                out_ls.append(out)
                traj = torch.cat(out_ls, dim=1)
                temp_x1 = traj[:,-Par['lb']:] #[BS, lb, nf, nx,ny]
                
        pred = torch.cat(out_ls, dim=1)[:, Par['lb']:NT] #[BS, lf, nf, nx, ny]
        y_pred_ls.append(pred)

        end = end+bs
        if end-bs > x.shape[0]+1:
            break
    
    y_pred = torch.cat(y_pred_ls, dim=0)

    return y_pred

def compute_tke_spectrum_timeseries_torch(u_all, lx, ly, lz, smooth=False, device='cpu'):
    """
    Compute TKE spectrum over time without loops using PyTorch.
    
    Parameters:
    -----------
    u_all : torch.Tensor
        Tensor of shape [3, ntime, nx, ny, nz] with velocity components.
    lx, ly, lz : float
        Domain sizes.
    smooth : bool
        Apply optional smoothing.
    device : str
        Device to use.

    Returns:
    --------
    knyquist : float
    wave_numbers : torch.Tensor [maxbin]
    tke_spectrum : torch.Tensor [ntime, maxbin]
    """
    u_all = torch.tensor(u_all, dtype=DTYPE, device=device) # [3, ntime, nx, ny, nz]
    nf, ntime, nx, ny, nz = u_all.shape
    assert nf == 3, "First dimension must be 3 (velocity components)"

    ntot = nx * ny * nz

    # FFT
    uh_all = torch.fft.fftn(u_all, dim=(-3, -2, -1)) / ntot  # [3, ntime, nx, ny, nz]

    # Compute energy: sum of squares of real and imaginary parts
    energy = 0.5 * (uh_all.real**2 + uh_all.imag**2).sum(dim=0)  # [ntime, nx, ny, nz]

    # Wavenumber grids
    kx = torch.fft.fftfreq(nx, d=lx / nx, device=device)
    ky = torch.fft.fftfreq(ny, d=ly / ny, device=device)
    kz = torch.fft.fftfreq(nz, d=lz / nz, device=device)
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
    K_mag = torch.sqrt(KX**2 + KY**2 + KZ**2)  # [nx, ny, nz]

    # Flatten for binning
    k_bins = torch.round(K_mag.flatten() / K_mag.max() * (nx // 2)).to(torch.int64)  # [npts]
    max_bin = k_bins.max().item() + 1

    wave_numbers = torch.arange(max_bin, device=device) * (2 * np.pi / ((lx + ly + lz)/3))

    # Flatten energy: [ntime, npts]
    energy_flat = energy.view(ntime, -1)  # [ntime, npts]

    # Allocate spectrum
    tke_spectrum = torch.zeros(ntime, max_bin, device=device)

    # Vectorized binning using broadcasting + scatter_add
    for k in range(max_bin):
        mask = (k_bins == k)
        if mask.any():
            tke_spectrum[:, k] = energy_flat[:, mask].sum(dim=1)

    if smooth:
        kernel = torch.ones(5, device=device) / 5
        kernel = kernel[None, None, :]
        tke_spectrum = torch.nn.functional.conv1d(
            tke_spectrum[:, None, :], kernel, padding=2
        ).squeeze(1)

    knyquist = (2 * np.pi / ((lx + ly + lz)/3)) * min(nx, ny, nz) / 2

    return knyquist, wave_numbers.detach().cpu().numpy(), tke_spectrum.detach().cpu().numpy()  # [ntime, maxbin]


def make_images(true, pred, epoch):
    # T,P - bs, nf, nx, ny, nz
    true = true.permute(0,2,1,3,4,5)
    pred = pred.permute(0,2,1,3,4,5)

    sample_id = 0
    f_id = 0
    t_id = 4
    z_id = 64

    CMAP = "viridis"
    
    knyquist, wave_numbers, spectrum_true = compute_tke_spectrum_timeseries_torch( true[sample_id, :3], 2*np.pi, 2*np.pi, 2*np.pi, device=device  )
    knyquist, wave_numbers, spectrum_pred = compute_tke_spectrum_timeseries_torch( pred[sample_id, :3], 2*np.pi, 2*np.pi, 2*np.pi, device=device  )
    

    T = true[sample_id, f_id, t_id, :, :, z_id].detach().cpu().numpy()
    P = pred[sample_id, f_id, t_id, :, :, z_id].detach().cpu().numpy()

    VMIN = np.min(T)
    VMAX = np.max(T)
    # VMIN = -1
    # VMAX = +1

    fig, axes = plt.subplots(1,3, figsize=(18,5))
    im = axes[0].imshow(T, cmap=CMAP, vmin=VMIN, vmax=VMAX)
    axes[0].set_title("True")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].imshow(P, cmap=CMAP, vmin=VMIN, vmax=VMAX)
    axes[1].set_title("Pred")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].loglog(wave_numbers, spectrum_true[t_id], label="True", color="black", lw=2.5)
    axes[2].loglog(wave_numbers, spectrum_pred[t_id], label="Pred", color="blue", lw=1.5)
    axes[2].loglog(wave_numbers, wave_numbers**(-5/3), label="slope = -5/3", color="magenta", ls="dotted")
    axes[2].legend()
    axes[2].set_xlabel("k", fontsize = 18)
    axes[2].set_ylabel("E(k)", fontsize=18)

    plt.tight_layout()

    fig.suptitle(f"Epoch: {epoch}", fontsize=22, y=1.2)
    plt.savefig(f"images/{epoch}.png", dpi=150, bbox_inches='tight')
    plt.close()


res = 128
begin_time = time.time()

traj = np.load("../data/data.npy") #[nf, nt, nx, ny, nz]
traj = traj.transpose(1,0,2,3,4)[4:]
traj = np.expand_dims(traj, axis=0) #[B, nt, nf, nx, ny, nz]
print(f"traj: {traj.shape}")
print(f"Data Loading Time: {time.time() - begin_time:.1f}s")

traj_train = traj[:, :160]
traj_val   = traj[:, 160:180]
traj_test  = traj[:, 180:]

Par = {}
Par['nx'] = traj_train.shape[-3]
Par['ny'] = traj_train.shape[-2]
Par['nz'] = traj_train.shape[-1]
Par['nf'] = traj_train.shape[2]
Par['d_emb'] = 128

Par['lb'] = 4
Par['lf'] = 4
Par['LF'] = 10
Par['channels'] = Par['nf']*Par['lb']
Par['num_epochs'] = 200 # NO converged within 200 epochs (= 15400 iterations of weight updates)

time_cond = np.linspace(0, 1, Par['lf'])
if Par['lf']==1:
    time_cond = np.linspace(0, 1, Par['lf']) + 1


begin_time = time.time()
print('\nTrain Dataset')
x_idx_train, t_idx_train, y_idx_train = preprocess_train(traj_train, Par)
print('\nValidation Dataset')
x_idx_val, t_idx_val, y_idx_val  = preprocess(traj_val, Par)
print('\nTest Dataset')
x_idx_test, t_idx_test, y_idx_test  = preprocess(traj_test, Par)
print(f"Data Preprocess Time: {time.time() - begin_time:.1f}s")

# sys.exit()

t_min = np.min(time_cond)
t_max = np.max(time_cond)
if Par['lf']==1:
    t_min=0
    t_max=1

MEAN = np.load('../data/MEAN.npy').reshape(1,-1,1,1,1)
STD  = np.load('../data/STD.npy').reshape(1,-1,1,1,1)
MIN  = np.load('../data/MIN.npy').reshape(1,-1,1,1,1)
MAX  = np.load('../data/MAX.npy').reshape(1,-1,1,1,1)
print(f"MEAN: {MEAN.shape}\nSTD: {STD.shape}\nMIN: {MIN.shape}\nMAX: {MAX.shape}")

Par['inp_shift'] = torch.tensor(MEAN, dtype=DTYPE, device=device)
Par['inp_scale'] = torch.tensor(STD, dtype=DTYPE, device=device)
Par['out_shift'] = torch.tensor(MEAN, dtype=DTYPE, device=device)
Par['out_scale'] = torch.tensor(STD, dtype=DTYPE, device=device)
Par['t_shift']   = torch.tensor(t_min, dtype=DTYPE, device=device)
Par['t_scale']   = torch.tensor(t_max - t_min, dtype=DTYPE, device=device)


with open('Par.pkl', 'wb') as f:
    pickle.dump(Par, f)


# Create custom datasets
traj_train_tensor = torch.tensor(traj_train, dtype=DTYPE)
traj_val_tensor = torch.tensor(traj_val, dtype=DTYPE)
traj_test_tensor = torch.tensor(traj_test, dtype=DTYPE)
time_cond_tensor = torch.tensor(time_cond, dtype=DTYPE)


train_dataset = YourDataset_train(x_idx_train, t_idx_train, y_idx_train)
val_dataset = YourDataset(x_idx_val, y_idx_val)
test_dataset = YourDataset(x_idx_test, y_idx_test)


# Define data loaders
train_batch_size = 8 #100
val_batch_size   = 1 #100
test_batch_size  = 1 #100
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size)

# Initialize your Unet2D model
model = Unet3D(dim=16, Par=Par, dim_mults=(1, 2, 4, 8), channels=Par['channels']).to(device).to(torch.float32)

# Define loss function and optimizer
criterion = CustomLoss(Par)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

# Learning rate scheduler (Cosine Annealing)
scheduler = combined_scheduler(optimizer, Par['num_epochs'] * len(train_loader), int(0.1 * Par['num_epochs']) * len(train_loader))


# Training loop
num_epochs = Par['num_epochs']
best_val_loss = float('inf')
best_model_id = 0

os.makedirs('models', exist_ok=True)
os.makedirs('images', exist_ok=True)

t0 = time.time()
for epoch in range(num_epochs):
    begin_time = time.time()
    model.train()
    train_loss = 0.0

    for x_idx, t_idx, y_idx in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        skip_flag = np.random.choice([True, False], 1, p=[0.2,0.8])
        if skip_flag:
            scheduler.step()
            continue

        x = traj_train_tensor[0, x_idx].to(device)
        t = time_cond_tensor[t_idx].to(device)
        y_true = traj_train_tensor[0, y_idx].to(device)

        optimizer.zero_grad()
        y_pred = model(x, t)
        loss   = criterion(y_pred, y_true.to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Update learning rate
        scheduler.step()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_idx, y_idx in val_loader:
            x = traj_val_tensor[0, x_idx]        #[BS, lb, nf, nx, ny]
            t = time_cond_tensor[t_idx_val]      #[lf, ]
            y_true = traj_val_tensor[0, y_idx]   #[BS,lf, nf, nx, ny]
            y_pred = rollout(model, x,t,Par['lb']+Par['LF'], Par, val_batch_size)
            if True:
                loss   = criterion(y_pred, y_true.to(device))
            val_loss += loss.item()

    val_loss /= len(val_loader)

    make_images(y_true, y_pred, epoch)

    # Save the model if validation loss is the lowest so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_id = epoch+1
        torch.save(model.state_dict(), f'models/model_{best_model_id}.pt')
    
    time_stamp = str('[')+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+str(']')
    elapsed_time = time.time() - begin_time
    print(time_stamp + f' - Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}, best model: {best_model_id}, LR: {scheduler.get_last_lr()[0]:.4e}, epoch time: {elapsed_time:.2f}')

print('Training finished.')
print(f"Training Time: {time.time() - t0:.1f}s")

# Testing loop
model.eval()
test_loss = 0.0
with torch.no_grad():
    for x_idx, y_idx in test_loader:
        x = traj_test_tensor[0, x_idx]        #[BS, lb, nf, nx, ny]
        t = time_cond_tensor[t_idx_test]      #[lf, ]
        y_true = traj_test_tensor[0, y_idx]   #[BS,lf, nf, nx, ny]
        y_pred = rollout(model, x,t,Par['lb']+Par['LF'], Par, val_batch_size)
        if True:
            loss   = criterion(y_pred, y_true.to(device))
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4e}')
