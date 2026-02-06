# ---------------------------------------------------------------------------------------------
# Author: Vivek Oommen
# Date: 09/01/2025
# ---------------------------------------------------------------------------------------------

import os
import sys
import random

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


# Define your custom loss function here
class CustomLoss(nn.Module):
    def __init__(self, Par):
        super(CustomLoss, self).__init__()
        self.Par = Par

    def forward(self, y_pred, y_true):
        y_true = (y_true - self.Par["out_shift"])/self.Par["out_scale"]
        y_pred = (y_pred - self.Par["out_shift"])/self.Par["out_scale"]
        loss = torch.norm(y_true-y_pred, p=2)/torch.norm(y_true, p=2)
        return loss

class YourDataset(Dataset):
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
    

def preprocess(traj, Par):
    nsamples = traj.shape[0]
    nt = traj.shape[1]
    temp = nt - Par['LB'] + 1
    x_idx = np.arange(temp).reshape(-1,1)
    x_idx = np.tile(x_idx, (1, Par['lf'])).reshape(-1,1)

    x_idx_ls = []
    for i in range(Par["LB"]):
        x_idx_ls.append(x_idx+i)
    x_idx = np.concatenate(x_idx_ls, axis=1)

    t_idx = np.arange(Par['lf']).reshape(1,-1)

    t_idx = np.tile(t_idx, (temp,1)).reshape(-1,)

    y_idx = np.arange(nt)
    y_idx = sliding_window_view(y_idx, window_shape=Par['LB']).reshape(-1,)


    x_idx = x_idx[:, 0]

    print(f"x_idx: {x_idx.shape}")
    print(f"{x_idx[:10]}")
    print(f"t_idx: {t_idx.shape}")
    print(f"{t_idx[:10]}")
    print(f"y_idx: {y_idx.shape}")
    print(f"{y_idx[:10]}")

    return x_idx, t_idx, y_idx




def combined_scheduler(optimizer, total_epochs, warmup_epochs, last_epoch=-1):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

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


def make_images(x, true, pred, epoch, make_plot=False):
    # T,P - bs, nf, nt, nx, ny, nz
    x    = x.unsqueeze(2)
    true = true.unsqueeze(2)
    pred = pred.unsqueeze(2)

    sample_id = random.randint(0, x.shape[0]-1)
    f_id = 0
    t_id = 0
    z_id = 16

    CMAP = "viridis"

    lx,ly,lz = 8,8,2
    
    knyquist, wave_numbers, spectrum_true = compute_tke_spectrum_timeseries_torch( true[sample_id, :3], lx,ly,lz, device=device  )
    knyquist, wave_numbers, spectrum_pred = compute_tke_spectrum_timeseries_torch( pred[sample_id, :3], lx,ly,lz, device=device  )
    
    
    s_err = np.mean( (np.log(spectrum_true[:,:])-np.log(spectrum_pred[:,:]) )**2 )

    X = x[sample_id, f_id, t_id, :, :, z_id].detach().cpu().numpy()
    T = true[sample_id, f_id, t_id, :, :, z_id].detach().cpu().numpy()
    P = pred[sample_id, f_id, t_id, :, :, z_id].detach().cpu().numpy()

    VMIN = np.min(T)
    VMAX = np.max(T)
    # VMIN = -1
    # VMAX = +1

    if make_plot:
        fig, axes = plt.subplots(1,4, figsize=(24,5))
        im = axes[0].imshow(X, cmap=CMAP)
        axes[0].set_title("Input")
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

        im = axes[1].imshow(T, cmap=CMAP, vmin=VMIN, vmax=VMAX)
        axes[1].set_title("True")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(P, cmap=CMAP, vmin=VMIN, vmax=VMAX)
        axes[2].set_title("Pred")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        axes[3].loglog(wave_numbers, spectrum_true[t_id], label="True", color="black", lw=2.5)
        axes[3].loglog(wave_numbers, spectrum_pred[t_id], label="Pred", color="blue", lw=1.5)
        axes[3].loglog(wave_numbers, wave_numbers**(-5/3), label="slope = -5/3", color="magenta", ls="dotted")
        axes[3].legend()
        axes[3].set_xlabel("k", fontsize = 18)
        axes[3].set_ylabel("E(k)", fontsize=18)

        

        plt.tight_layout()


        fig.suptitle(f"Epoch: {epoch}", fontsize=22, y=1.2)
        plt.savefig(f"images/{epoch}.png", dpi=150, bbox_inches='tight')
        plt.close()

    return s_err


def random_point_inpainting_prep(x: torch.Tensor, y: torch.Tensor,
                                  min_percent: float = 0.0, max_percent: float = 1.0):
    """
    x: torch.Tensor of shape [bs, nf, nx, ny, nz]
    y: torch.Tensor of shape [bs, nf, nx, ny, nz]

    Returns:
      x_in: [bs, nf+1, nx, ny, nz] - randomly masked x concatenated with mask
      y   : unchanged target
    """
    bs, nf, nx, ny, nz = x.shape
    device, dtype = x.device, x.dtype
    total_voxels = nx * ny * nz

    # Create empty mask container
    mask = torch.ones((bs, 1, nx, ny, nz), device=device, dtype=dtype)

    for b in range(bs):
        # Sample a random masking percentage for this sample
        masking_percent = random.uniform(min_percent, max_percent)
        num_masked_voxels = int(total_voxels * masking_percent)

        # Generate random indices to mask
        indices = torch.randperm(total_voxels, device=device)[:num_masked_voxels]

        # Convert flat indices to 3D coordinates
        x_idx = indices // (ny * nz)
        y_idx = (indices % (ny * nz)) // nz
        z_idx = indices % nz

        # Apply masking
        mask[b, 0, x_idx, y_idx, z_idx] = 0

    mask = mask.repeat(1,4,1,1,1)
    # Apply the mask to x
    x_masked = x * mask  # broadcast along channel dim

    # Concatenate mask as extra channel
    x_in = torch.cat([x_masked, mask], dim=1)  # [bs, nf+1, nx, ny, nz]

    return x_in, y

def random_point_inpainting_prep_type2(x: torch.Tensor, y: torch.Tensor,
                                  min_percent: float = 0.0, max_percent: float = 1.0):
    """
    x: torch.Tensor of shape [bs, nf, nx, ny, nz]
    y: torch.Tensor of shape [bs, nf, nx, ny, nz]

    Returns:
      x_in: [bs, nf+1, nx, ny, nz] - randomly masked x concatenated with mask
      y   : unchanged target
    """
    bs, nf, nx, ny, nz = x.shape
    device, dtype = x.device, x.dtype
    total_voxels = nx * ny * nz

    masking_percent = random.uniform(min_percent, max_percent)
    rand_field = torch.rand((bs, 4, nx, ny, nz), device=device)
    mask = (rand_field > masking_percent).float()

    # Apply the mask to x
    x_masked = x * mask  # broadcast along channel dim

    # Concatenate mask as extra channel
    x_in = torch.cat([x_masked, mask], dim=1)  # [bs, nf+1, nx, ny, nz]

    return x_in, y

def sr_prep(x: torch.Tensor, y: torch.Tensor, mask_size=0.125):
    """
    x: torch.Tensor of shape [bs, nf, nx, ny, nz]
    y: torch.Tensor of shape [bs, nf, nx, ny, nz]  (same content as x[:,0])
    
    Returns:
      x_in: [bs, nf+1, nx, ny, nz]  (masked x with mask concatenated)
      y   : unchanged target tensor
    """
    bs, nf, nx, ny, nz = x.shape
    device, dtype = x.device, x.dtype

    if random.random() < 0.5:
        xnew, ynew = random_point_inpainting_prep(x[:], y[:], min_percent=0.0, max_percent=1.0)
    else:
        xnew, ynew = random_point_inpainting_prep_type2(x[:], y[:], min_percent=0.0, max_percent=1.0)

    return xnew, ynew






# Load your data into NumPy arrays (x_train, t_train, y_train, x_val, t_val, y_val, x_test, t_test, y_test)
#########################
begin_time = time.time()

temp = np.load(f"../data/u.npy")[:, :, :256, :256] #[nf, nt, nx, ny, nz]
traj_i = temp
traj_i = traj_i.transpose(1,0,2,3,4) #[nt, nf, nx, ny, nz]
traj_i = np.expand_dims(traj_i, axis=0) #[1, nt, nf, nx, ny, nz]

traj_o = temp 
traj_o = traj_o.transpose(1,0,2,3,4) #[nt, nf, nx, ny, nz]
traj_o = np.expand_dims(traj_o, axis=0) #[1, nt, nf, nx, ny, nz]

print(f"traj_i: {traj_i.shape}")
print(f"traj_o: {traj_o.shape}")

print(f"Data Loading Time: {time.time() - begin_time:.1f}s")

# sys.exit()

traj_i_train = traj_i[:, :150]
traj_i_val   = traj_i[:, 150:168]
traj_i_test  = traj_i[:, 168:]

traj_o_train = traj_o[:, :150]
traj_o_val   = traj_o[:, 150:168]
traj_o_test  = traj_o[:, 168:]


Par = {}
# Par['nt'] = 100 
Par['nx'] = traj_o_train.shape[-3]
Par['ny'] = traj_o_train.shape[-2]
Par['nz'] = traj_o_train.shape[-1]
Par['nf'] = traj_o_train.shape[2]
Par['d_emb'] = 128

Par['lb'] = 1
Par['LB'] = 1
Par['lf'] = 1
Par['channels'] = (Par['nf'])

Par['num_epochs'] = 500 #(9500 iterations of weight update)

time_cond = np.linspace(0, 1, Par['lf'])
if Par['lf']==1:
    time_cond = np.linspace(0, 1, Par['lf']) + 1


begin_time = time.time()
print('\nTrain Dataset')
x_idx_train, t_idx_train, y_idx_train = preprocess(traj_i_train, Par)
print('\nValidation Dataset')
x_idx_val, t_idx_val, y_idx_val  = preprocess(traj_i_val, Par)
print('\nTest Dataset')
x_idx_test, t_idx_test, y_idx_test  = preprocess(traj_i_test, Par)
print(f"Data Preprocess Time: {time.time() - begin_time:.1f}s")

# sys.exit()

t_min = np.min(time_cond)
t_max = np.max(time_cond)
if Par['lf']==1:
    t_min=0
    t_max=1

inp_MEAN = np.load('../data/MEAN_u.npy').reshape(1,-1,1,1,1)
inp_MEAN = np.concatenate([inp_MEAN, np.array([0]*4).reshape(1,-1,1,1,1) ], axis=1)

inp_STD  = np.load('../data/STD_u.npy').reshape(1,-1,1,1,1)
inp_STD  = np.concatenate([inp_STD, np.array([1]*4).reshape(1,-1,1,1,1) ], axis=1)

inp_MIN  = np.load('../data/MIN_u.npy').reshape(1,-1,1,1,1)
inp_MIN = np.concatenate([inp_MIN, np.array([0]*4).reshape(1,-1,1,1,1) ], axis=1)

inp_MAX  = np.load('../data/MAX_u.npy').reshape(1,-1,1,1,1)
inp_MAX  = np.concatenate([inp_MAX, np.array([1]*4).reshape(1,-1,1,1,1) ], axis=1)
print(f"inp MEAN: {inp_MEAN.shape}\ninp STD: {inp_STD.shape}\ninp MIN: {inp_MIN.shape}\ninp MAX: {inp_MAX.shape}")



out_MEAN = np.load('../data/MEAN_u.npy').reshape(1,-1,1,1,1)
out_STD  = np.load('../data/STD_u.npy').reshape(1,-1,1,1,1)
out_MIN  = np.load('../data/MIN_u.npy').reshape(1,-1,1,1,1)
out_MAX  = np.load('../data/MAX_u.npy').reshape(1,-1,1,1,1)
print(f"out MEAN: {out_MEAN.shape}\nout STD: {out_STD.shape}\nout MIN: {out_MIN.shape}\nout MAX: {out_MAX.shape}")

Par['inp_shift'] = torch.tensor(inp_MEAN, dtype=DTYPE, device=device)
Par['inp_scale'] = torch.tensor(inp_STD, dtype=DTYPE, device=device)
Par['out_shift'] = torch.tensor(out_MEAN, dtype=DTYPE, device=device)
Par['out_scale'] = torch.tensor(out_STD, dtype=DTYPE, device=device)
Par['t_shift']   = torch.tensor(t_min, dtype=DTYPE, device=device)
Par['t_scale']   = torch.tensor(t_max - t_min, dtype=DTYPE, device=device)


with open('Par.pkl', 'wb') as f:
    pickle.dump(Par, f)


traj_i_train_tensor = torch.tensor(traj_i_train, dtype=DTYPE)
traj_i_val_tensor = torch.tensor(traj_i_val, dtype=DTYPE)
traj_i_test_tensor = torch.tensor(traj_i_test, dtype=DTYPE)

traj_o_train_tensor = torch.tensor(traj_o_train, dtype=DTYPE)
traj_o_val_tensor = torch.tensor(traj_o_val, dtype=DTYPE)
traj_o_test_tensor = torch.tensor(traj_o_test, dtype=DTYPE)

time_cond_tensor = torch.tensor(time_cond, dtype=DTYPE)


train_dataset = YourDataset(x_idx_train, t_idx_train, y_idx_train)
val_dataset = YourDataset(x_idx_val, t_idx_val, y_idx_val)
test_dataset = YourDataset(x_idx_test, t_idx_test, y_idx_test)


# Define data loaders
train_batch_size = 8 #100
val_batch_size   = 6 #100
test_batch_size  = 6 #100
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size)

# Initialize your Unet2D model
model = Unet3D(dim=16, Par=Par, dim_mults=(1, 2, 4, 8), channels=2*Par['channels']).to(device).to(torch.float32)
summary(model, input_size=((1, 2*Par["channels"], Par["nx"], Par["ny"], Par["nz"]), (1,)) )

# Define loss function and optimizer
criterion = CustomLoss(Par)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

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
        skip_flag = np.random.choice([True, False], 1, p=[0.05,0.95])
        if skip_flag:
            scheduler.step()
            continue

        x = traj_i_train_tensor[0, x_idx].to(device)
        t = time_cond_tensor[t_idx].to(device)
        y_true = traj_o_train_tensor[0, y_idx].to(device)



        x, y_true = sr_prep(x, y_true)

        optimizer.zero_grad()
        y_pred = model(x, t)
        loss   = criterion(y_pred, y_true)
        # backward + step
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
        for x_idx, t_idx, y_idx in val_loader:
            x = traj_i_val_tensor[0, x_idx].to(device)        #[BS, lb, nf, nx, ny]
            t = time_cond_tensor[t_idx].to(device)      #[lf, ]
            y_true = traj_o_val_tensor[0, y_idx].to(device)   #[BS,lf, nf, nx, ny]

            x, y_true = sr_prep(x, y_true)


            y_pred = model(x,t) 
            if True:
                loss   = criterion(y_pred, y_true)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    _ = make_images(x, y_true, y_pred, epoch+1, make_plot=True)

    # Save the model if validation loss is the lowest so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_id = epoch+1
        torch.save(model.state_dict(), f'models/best_model.pt')
    
    time_stamp = str('[')+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+str(']')
    elapsed_time = time.time() - begin_time
    print(time_stamp + f' - Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}, best model: {best_model_id}, LR: {scheduler.get_last_lr()[0]:.4e}, epoch time: {elapsed_time:.2f}')

print('Training finished.')
print(f"Training Time: {time.time() - t0:.1f}s")

# Testing loop
model.eval()
test_loss = 0.0
with torch.no_grad():
    for x_idx, t_idx, y_idx in test_loader:
        x = traj_i_test_tensor[0, x_idx].to(device)        #[BS, lb, nf, nx, ny]
        t = time_cond_tensor[t_idx].to(device)      #[lf, ]
        y_true = traj_o_test_tensor[0, y_idx].to(device)   #[BS,lf, nf, nx, ny]
        y_pred = model(x,t) 
        if True:
            loss   = criterion(y_pred, y_true)
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4e}')

