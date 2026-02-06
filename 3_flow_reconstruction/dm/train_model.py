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
from torch.optim.lr_scheduler import CosineAnnealingLR
# from YourDataset import YourDataset  # Import your custom dataset here
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchinfo import summary
import torchprofile
import matplotlib.pyplot as plt

from utils.architecture import Unet
from utils.diffusion import ElucidatedDiffusion

torch.manual_seed(23)
import pickle

DTYPE = torch.float32

scaler = GradScaler()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def error_metric(pred,true, Par):
    return torch.norm(true-pred, p=2)/torch.norm(true, p=2)

class MyDataset(Dataset):
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



def reg_inpainting_prep(x: torch.Tensor, y: torch.Tensor, mask_size=0.125):
    """
    x: torch.Tensor of shape [bs, nf, nx, ny, nz]
    y: torch.Tensor of shape [bs, nf, nx, ny, nz]  (same content as x[:,0])
    
    Returns:
      x_in: [bs, nf+1, nx, ny, nz]  (masked x with mask concatenated)
      y    : unchanged target tensor
    """
    bs, nf, nx, ny, nz = x.shape
    device, dtype = x.device, x.dtype

    # 1) Build mask: shape [bs,1,1,nx,ny,nz], same mask for all batch entries
    #    Start all ones, then zero out 1–4 random patches.
    mask = torch.ones((1, 1, nx, ny, nz), device=device, dtype=dtype)
    num_patches = random.choice([1, 2, 3, 4])
    for _ in range(num_patches):
        # random patch size up to 25% of each dim (but at least 1)
        lx = random.randint(1, max(1, int(mask_size * nx)))
        ly = random.randint(1, max(1, int(mask_size * ny)))
        lz = random.randint(1, max(1, int(mask_size * nz)))
        # random start positions
        x0 = random.randint(0, nx - lx)
        y0 = random.randint(0, ny - ly)
        z0 = random.randint(0, nz - lz)
        # zero out that cuboid
        mask[..., x0:x0+lx, y0:y0+ly, z0:z0+lz] = 0

    # repeat across batch
    mask = mask.repeat(bs, 1, 1, 1, 1)  # [bs,1,nx,ny,nz]

    # 2) apply mask to x
    x_masked = x * mask

    # 3) concatenate mask as an extra feature channel along dim=2
    #    so channel count goes from nf to nf+1
    x_in = torch.cat([x_masked, mask], dim=1)  #  [bs,nf+1,nx,ny,nz]

    # return the prepared input and the original target
    return x_in, y


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
      y    : unchanged target tensor
    """
    bs, nf, nx, ny, nz = x.shape
    device, dtype = x.device, x.dtype

    if random.random() < 0.5:
        xnew, ynew = random_point_inpainting_prep(x[:], y[:], min_percent=0.0, max_percent=1.0)
    else:
        xnew, ynew = random_point_inpainting_prep_type2(x[:], y[:], min_percent=0.0, max_percent=1.0)

    return xnew, ynew


def sr_prep_test(x: torch.Tensor, y: torch.Tensor, mask_size=0.25):
    """
    x: torch.Tensor of shape [bs, 1, nf, nx, ny, nz]
    y: torch.Tensor of shape [bs,    nf, nx, ny, nz]  (same content as x[:,0])
    
    Returns:
      x_in: [bs, 1, nf+1, nx, ny, nz]  (masked x with mask concatenated)
      y    : unchanged target tensor
    """
    bs, nf, nx, ny, nz = x.shape
    device, dtype = x.device, x.dtype

    # 1) Build mask: shape [bs,1,1,nx,ny,nz], same mask for all batch entries
    #    Start all ones, then zero out 1–4 random patches.
    mask = torch.ones((1, 1, nx, ny, nz), device=device, dtype=dtype)
    num_patches = 1 #random.choice([1, 2, 3, 4])
    for _ in range(num_patches):
        # random patch size up to 25% of each dim (but at least 1)
        lx = 64 #random.randint(1, max(1, int(mask_size * nx)))
        ly = 64 #random.randint(1, max(1, int(mask_size * ny)))
        lz = 8 #random.randint(1, max(1, int(mask_size * nz)))
        # random start positions
        x0 = 96 #random.randint(0, nx - lx)
        y0 = 96 #random.randint(0, ny - ly)
        z0 = 12 #random.randint(0, nz - lz)
        # zero out that cuboid
        mask[..., x0:x0+lx, y0:y0+ly, z0:z0+lz] = 0

    # repeat across batch
    mask = mask.repeat(bs, 1, 1, 1, 1)  # [bs,1,1,nx,ny,nz]

    # 2) apply mask to x
    x_masked = x * mask 

    # 3) concatenate mask as an extra feature channel along dim=2
    #    so channel count goes from nf → nf+1
    x_in = torch.cat([x_masked, mask], dim=1)  # → [bs,1,nf+1,nx,ny,nz]

    # return the prepared input and the original target
    return x_in, y




res = 128
begin_time = time.time()

temp = np.load(f"../data/u.npy")[:, :, :256, :256] #[nf, nt, nx, ny, nz]
traj_i = temp
traj_i = traj_i.transpose(1,0,2,3,4) #[nt, nf, nx, ny, nz]
traj_i = np.expand_dims(traj_i, axis=0) #[1, nt, nf, nx, ny, nz]

traj_o = np.load(f"../data/u.npy")[:, :, :256, :256] #np.load(f"../data_11k_31gb/u.npy")[:, :, :256, :256] #[nf, nt, nx, ny, nz]
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




# inp_min = np.min(x_train)
# inp_max = np.max(x_train)
# out_min = np.min(y_train)
# out_max = np.max(y_train)

inp_min  = np.load('../data/MIN_u.npy').reshape(1,1,-1,1,1,1)
# inp_min = np.concatenate([inp_MIN, np.array([0]).reshape(1,1,-1,1,1,1) ], axis=2)

inp_max  = np.load('../data/MAX_u.npy').reshape(1,1,-1,1,1,1)
# inp_max  = np.concatenate([inp_MAX, np.array([1]).reshape(1,1,-1,1,1,1) ], axis=2)

out_min  = np.load('../data/MIN_u.npy').reshape(1,1,-1,1,1,1)
out_max  = np.load('../data/MAX_u.npy').reshape(1,1,-1,1,1,1)


Par = {"inp_shift" : torch.tensor(inp_min, dtype=DTYPE, device=device),
       "inp_scale" : torch.tensor(inp_max - inp_min, dtype=DTYPE, device=device),
       "out_shift" : torch.tensor(out_min, dtype=DTYPE, device=device),
       "out_scale" : torch.tensor(out_max - out_min, dtype=DTYPE, device=device),
       "nx"        : traj_i_train.shape[-3],
       "ny"        : traj_i_train.shape[-2],
       "nz"        : traj_i_train.shape[-1],
       "nf"        : traj_i_train.shape[2],
       "lb"        : 1,
       "lf"        : 1,
       'LB'        : 1,
       "num_epochs": 100000
       }



# Normalizing the data to [0,1]
shift = Par['inp_shift'].detach().cpu().numpy()
scale = Par['inp_scale'].detach().cpu().numpy()
traj_i_train = (traj_i_train - shift)/scale
traj_i_val = (traj_i_val - shift)/scale
traj_i_test = (traj_i_test - shift)/scale

shift = Par['out_shift'].detach().cpu().numpy()
scale = Par['out_scale'].detach().cpu().numpy()
traj_o_train = (traj_o_train - shift)/scale
traj_o_val = (traj_o_val - shift)/scale
traj_o_test = (traj_o_test - shift)/scale

Par["sigma_data"] = np.std(traj_o_train)

print(f'STD data: {Par["sigma_data"]}')

# Traj splitting
begin_time = time.time()

print('\nTrain Dataset')
x_idx_train, t_idx_train, y_idx_train = preprocess(traj_i_train, Par)
print('\nValidation Dataset')
x_idx_val, t_idx_val, y_idx_val  = preprocess(traj_i_val, Par)
print('\nTest Dataset')
x_idx_test, t_idx_test, y_idx_test  = preprocess(traj_i_test, Par)
print(f"Data Preprocess Time: {time.time() - begin_time:.1f}s")


Par.update({"channels"       : traj_i_train.shape[2],
            "self_condition" : True
            })

print(f'Par:\n{Par}')
with open('Par.pkl', 'wb') as f:
    pickle.dump(Par, f)

traj_i_train_tensor = torch.tensor(traj_i_train, dtype=DTYPE)
traj_i_val_tensor = torch.tensor(traj_i_val, dtype=DTYPE)
traj_i_test_tensor = torch.tensor(traj_i_test, dtype=DTYPE)

traj_o_train_tensor = torch.tensor(traj_o_train, dtype=DTYPE)
traj_o_val_tensor = torch.tensor(traj_o_val, dtype=DTYPE)
traj_o_test_tensor = torch.tensor(traj_o_test, dtype=DTYPE)

train_dataset = MyDataset(x_idx_train, t_idx_train, y_idx_train)
val_dataset = MyDataset(x_idx_val, t_idx_val, y_idx_val)
test_dataset = MyDataset(x_idx_test, t_idx_test, y_idx_test)


# Define data loaders
train_batch_size = 6 #100
val_batch_size   = 6 #100
test_batch_size  = 6 #100
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size)

# Define Network Architecture
net = Unet(
    dim = 16,
    dim_mults = (1, 2, 4, 8),
    channels = Par["channels"],
    self_condition = Par["self_condition"],
    flash_attn = True
).to(device).to(torch.float32)

dummy_x = torch.tensor(np.random.uniform(size=(1, Par['nf'], Par['nx'], Par['ny'], Par['nz'])), dtype=DTYPE, device=device )
dummy_t = torch.tensor(np.array([1,]), dtype=DTYPE, device=device)
dummy_cond = torch.tensor(np.random.uniform(size=(1, Par['nf']*2, Par['nx'], Par['ny'], Par['nz'])), dtype=DTYPE, device=device )
print(f'dummy_x   : {dummy_x.shape}')
print(f'dummy_t   : {dummy_t.shape}')
print(f'dummy_cond: {dummy_cond.shape}')

dummy_out = net(dummy_x, dummy_t, dummy_cond)
print(f'dummy_out: {dummy_out.shape}')


summary(net, input_size=( (dummy_x.shape), (dummy_t.shape), dummy_cond.shape ) )

# sys.exit()

model = ElucidatedDiffusion(net,
                                channels = Par["channels"],
                                image_size_h=Par["nx"],
                                image_size_w=Par["ny"],
                                image_size_d=Par["nz"],
                                sigma_data=Par["sigma_data"])

dummy_input = (dummy_x, dummy_cond)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)

# Learning rate scheduler (Cosine Annealing)
scheduler = CosineAnnealingLR(optimizer, T_max= Par['num_epochs'] * len(train_loader) )  # Adjust T_max as needed

# Training loop
num_epochs = Par['num_epochs']
best_loss = float('inf')
best_model_id = 0

os.makedirs('models', exist_ok=True)
os.makedirs('images', exist_ok=True)

t0 = time.time()
for epoch in range(num_epochs):
    begin_time = time.time()
    model.train()
    train_loss = 0.0

    train_time = time.time()
    for  x_idx, _, y_idx  in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):

        l_fidel = traj_i_train_tensor[0, x_idx].to(device)
        h_fidel = traj_o_train_tensor[0, y_idx].to(device)

        l_fidel, h_true = sr_prep(l_fidel, h_fidel)

        optimizer.zero_grad()
        loss = model(h_fidel.to(device), l_fidel.to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()


    train_loss /= len(train_loader)
    train_time = time.time()-train_time

    # Validation
    if epoch != 0 and epoch % 10 == 0:
    # if True:
        val_time = time.time()
        model.eval()
        val_loss = 0.0
        spec_err = 0.0
        with torch.no_grad():
            for x_idx, _, y_idx in val_loader:
                l_fidel = traj_i_val_tensor[0, x_idx].to(device)
                h_fidel = traj_o_val_tensor[0, y_idx].to(device)

                l_fidel, h_true = sr_prep(l_fidel, h_fidel)

                pred  = model.sample(l_fidel.to(device))
                loss  = error_metric(pred, h_fidel.to(device), Par)
                s_err = make_images(l_fidel, h_fidel, pred, epoch+1)
                val_loss += loss.item()
                spec_err += s_err

        val_loss /= len(val_loader)
        spec_err /= len(val_loader)

        _ = make_images(l_fidel, h_fidel, pred, epoch+1, make_plot=True)

            # Save the model if validation loss is the lowest so far
        if spec_err < best_loss:
            best_loss = spec_err
            best_model_id = epoch+1
            torch.save(model.state_dict(), f'models/best_model.pt')
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'models/model_{epoch}.pt')

        val_time = time.time() - val_time
        
        time_stamp = str('[')+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+str(']')
        elapsed_time = time.time() - begin_time
        print(time_stamp + f' - Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}, spec err: {spec_err:.4e}, best model: {best_model_id}, LR: {scheduler.get_last_lr()[0]:.4e}, train time: {train_time:.2f}, val time: {val_time:.2f}')

    else:
        time_stamp = str('[')+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+str(']')
        print(time_stamp + f' - Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4e}, LR: {scheduler.get_last_lr()[0]:.4e}, train time: {train_time:.2f}')


print('Training finished.')
print(f"Training Time: {time.time() - t0:.1f}s")

