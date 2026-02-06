# ---------------------------------------------------------------------------------------------
# Author: Vivek Oommen
# Date: 09/01/2025
# ---------------------------------------------------------------------------------------------

import argparse
import os
import sys
import random

sys.path.append(os.path.abspath("..")) 

import datetime
import time

from tcunet import Unet3D

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data import Dataset

from torchinfo import summary
import torchprofile

from model import *
from utils import *
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


from torchvision.utils import save_image

import matplotlib.pyplot as plt

DTYPE = torch.float32


CMAP = "magma"



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


def get_slices(P, T):
    #P,T - [BS*lf, nf, X, Y, Z]
    _, nf, nx, ny, nz = T.shape
  
    size_x = 32
    size_y = 32
    size_z = 16

    # ymin, ymax = 0, 127

    idx = np.random.choice(nx-size_x, 1)[0]
    idy = np.random.choice(ny-size_y, 1)[0]
    # idy = np.random.choice(np.arange(ymin, ymax-size_y), 1)[0]
    idz = np.random.choice(nz-size_z, 1)[0]

    x1, x2 = idx, idx+size_x
    y1, y2 = idy, idy+size_y
    z1, z2 = idz, idz+size_z

    slice_T = T[:,:, y1:y2, x1:x2, z1:z2]
    slice_P = P[:,:, y1:y2, x1:x2, z1:z2]


    return slice_P, slice_T


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

    # bs1 = 1 #Since my batch_size = 6, 16% of the samples does unconditional generation

    # x1, y1 = random_point_inpainting_prep(x[:bs1], y[:bs1], min_percent=1.0, max_percent=1.0) #unconditional generation
    # x2, y2 = random_point_inpainting_prep(x[bs1:], y[bs1:], min_percent=0.0, max_percent=1.0)

    # xnew = torch.cat([x1,x2], dim=0)
    # ynew = torch.cat([y1,y2], dim=0)

    if random.random() < 0.5:
        xnew, ynew = random_point_inpainting_prep(x[:], y[:], min_percent=0.0, max_percent=1.0)
    else:
        xnew, ynew = random_point_inpainting_prep_type2(x[:], y[:], min_percent=0.0, max_percent=1.0)

    return xnew, ynew




os.makedirs('images', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--crop_size', default=128, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=1, type=int, help='super resolution upscale factor')
parser.add_argument('--warmup_batches', default=0, type=int, help='number of batches with pixel-wise loss only')
parser.add_argument('--n_batches', default=14000000, type=int, help='number of batches of training')
parser.add_argument('--residual_blocks', default=23, type=int, help='number of residual blocks in the generator')
parser.add_argument('--batch', default=0, type=int, help='batch to start training from')
parser.add_argument('--lr', default=0.0002, type=float, help='adam: learning rate')
parser.add_argument('--sample_interval', default=100, type=int, help='interval between saving image samples')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #d2l.try_gpu()
print(device)

##############################################
begin_time = time.time()

temp = np.load(f"../../data/u.npy")[:, :, :256, :256] #[nf, nt, nx, ny, nz]
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

Par['num_epochs'] = 500 #50

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

t_min = np.min(time_cond)
t_max = np.max(time_cond)
if Par['lf']==1:
    t_min=0
    t_max=1

inp_MEAN = np.load('../../data/MEAN_u.npy').reshape(1,-1,1,1,1)
inp_MEAN = np.concatenate([inp_MEAN, np.array([0]*4).reshape(1,-1,1,1,1) ], axis=1)

inp_STD  = np.load('../../data/STD_u.npy').reshape(1,-1,1,1,1)
inp_STD  = np.concatenate([inp_STD, np.array([1]*4).reshape(1,-1,1,1,1) ], axis=1)

inp_MIN  = np.load('../../data/MIN_u.npy').reshape(1,-1,1,1,1)
inp_MIN = np.concatenate([inp_MIN, np.array([0]*4).reshape(1,-1,1,1,1) ], axis=1)

inp_MAX  = np.load('../../data/MAX_u.npy').reshape(1,-1,1,1,1)
inp_MAX  = np.concatenate([inp_MAX, np.array([1]*4).reshape(1,-1,1,1,1) ], axis=1)
print(f"inp MEAN: {inp_MEAN.shape}\ninp STD: {inp_STD.shape}\ninp MIN: {inp_MIN.shape}\ninp MAX: {inp_MAX.shape}")



out_MEAN = np.load('../../data/MEAN_u.npy').reshape(1,-1,1,1,1)
out_STD  = np.load('../../data/STD_u.npy').reshape(1,-1,1,1,1)
out_MIN  = np.load('../../data/MIN_u.npy').reshape(1,-1,1,1,1)
out_MAX  = np.load('../../data/MAX_u.npy').reshape(1,-1,1,1,1)
print(f"out MEAN: {out_MEAN.shape}\nout STD: {out_STD.shape}\nout MIN: {out_MIN.shape}\nout MAX: {out_MAX.shape}")

Par['inp_shift'] = torch.tensor(inp_MEAN, dtype=DTYPE, device=device)
Par['inp_scale'] = torch.tensor(inp_STD, dtype=DTYPE, device=device)
Par['out_shift'] = torch.tensor(out_MEAN, dtype=DTYPE, device=device)
Par['out_scale'] = torch.tensor(out_STD, dtype=DTYPE, device=device)
Par['t_shift']   = torch.tensor(t_min, dtype=DTYPE, device=device)
Par['t_scale']   = torch.tensor(t_max - t_min, dtype=DTYPE, device=device)

# Create custom datasets
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



hr_shape = (opt.crop_size, opt.crop_size)

generator = Unet3D(dim=16, Par=Par, dim_mults=(1, 2, 4, 8), channels=2*Par['channels']).to(device).to(torch.float32)
summary(generator, input_size=((1, 2*Par["channels"], Par["nx"], Par["ny"], Par["nz"]), (1,)) ) 
print('Loading model ...')
path_model = '../models/best_model.pt'
generator.load_state_dict(torch.load(path_model))


# Adjust the dimensions as per your model's input size
dummy_x = torch.tensor(torch.randn(1, 2*Par['channels'], Par['nx'],Par['ny'], Par['nz']),   dtype=DTYPE, device=device)
dummy_t = time_cond_tensor[0:1].to(device)
dummy_input = (dummy_x, dummy_t)
# Profile the model
flops = 2*torchprofile.profile_macs(generator, dummy_input)
print(f"FLOPs: {flops:.2e}")


discriminator = UNetDiscriminatorSN(Par["nf"]).to(device).to(DTYPE)
print(summary(discriminator, input_size=(1,Par["nf"],Par['nx'],Par['ny'],Par['nz'])) )

feature_extractor = FeatureExtractor(Par, weight_path="pretrain/resnet_10_23dataset.pth").to(device).to(DTYPE)

# Losses  
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)

criterion_NO = CustomLoss(Par)


if opt.batch != 0:
    generator.load_state_dict(torch.load('saved_models/generator_%d.pth' % opt.batch))
    discriminator.load_state_dict(torch.load('saved_models/discriminator_%d.pth' % opt.batch))

# initialize optimzier 
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

# initialize ema  
ema_G = EMA(generator, 0.999)
ema_D = EMA(discriminator, 0.999)
ema_G.register()
ema_D.register()

best_loss = float('inf')

# -----------------
# Training 
# -----------------

batch = opt.batch
while batch < opt.n_batches:
    begin_time = time.time()
    for i, (x_idx, t_idx, y_idx) in enumerate(train_loader):

        batches_done = batch + i

        skip_flag = np.random.choice([True, False], 1, p=[0.5,0.5])
        if skip_flag:
            continue

        data = traj_i_train_tensor[0, x_idx].to(device)
        tt = time_cond_tensor[t_idx].to(device)
        target = traj_o_train_tensor[0, y_idx].to(device)

        data, target = sr_prep(data, target)

        imgs_lr = data 
        imgs_hr = target 

        # ---------------------
        # Training Generator
        # ---------------------

        optimizer_G.zero_grad()

        gen_hr = generator(imgs_lr, tt)
        

        loss_pixel = criterion_pixel(gen_hr, imgs_hr)

        gen_hr, imgs_hr = get_slices(gen_hr, imgs_hr)

        valid = torch.ones((imgs_hr.size(0), 1, *imgs_hr.shape[-3:]), requires_grad=False).to(device)
        fake = torch.zeros((imgs_hr.size(0), 1, *imgs_hr.shape[-3:]), requires_grad=False).to(device)

        if batches_done < opt.warmup_batches:
            loss_pixel.backward()
            optimizer_G.step()
            ema_G.update()
            print(
                '[Iteration %d/%d] [Batch %d/%d] [G pixel: %f]' % 
                (batches_done, opt.n_batches, i, len(train_loader), loss_pixel.item())
            )
            continue
        elif batches_done == opt.warmup_batches:
            optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4)

        pred_real = discriminator(imgs_hr).detach()
        pred_fake = discriminator(gen_hr)

        loss_GAN = (
            criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid) + 
            criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), fake)
        ) / 2

        

        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        real_features = [real_f.detach() for real_f in real_features]
        loss_content = sum(criterion_content(gen_f, real_f) * w for gen_f, real_f, w in zip(gen_features, real_features, [0.1, 0.1, 1]))


        loss_G = loss_content + 0.1 * loss_GAN + loss_pixel
        

        loss_G.backward()
        optimizer_G.step()
        ema_G.update()

        # ---------------------
        # Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        pred_real = discriminator(imgs_hr)
        pred_fake = discriminator(gen_hr.detach())

        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()
        ema_D.update()

        # -------------------------
        # Log Progress
        # -------------------------

        print(
            '[Iteration %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]' % 
            (
                batches_done, 
                opt.n_batches, 
                i, 
                len(train_loader), 
                loss_D.item(), 
                loss_G.item(), 
                loss_content.item(), 
                loss_GAN.item(), 
                loss_pixel.item()
            )
        )


    batch = batches_done + 1

    val_loss = 0.0
    spec_err = 0.0
    
    with torch.no_grad():
        for x_idx, t_idx, y_idx in val_loader:
            x = traj_i_val_tensor[0, x_idx].to(device)        #[BS, lb, nf, nx, ny]
            t = time_cond_tensor[t_idx].to(device)      #[lf, ]
            y_true = traj_o_val_tensor[0, y_idx].to(device)   #[BS,lf, nf, nx, ny]

            x, y_true = sr_prep(x, y_true)


            y_pred = generator(x,t) 
            if True:
                loss   = criterion_NO(y_pred, y_true)
            s_err = make_images(x, y_true, y_pred, batch)
            val_loss += loss
            spec_err += s_err

    val_loss /= len(val_loader)
    spec_err /= len(val_loader)
    
    ema_G.apply_shadow()
    ema_D.apply_shadow()

    if val_loss < best_loss:
        best_loss = val_loss
        best_model_id = batch
        torch.save(generator.state_dict(), f'saved_models/best_generator_{best_model_id}.pt')

    torch.save(generator.state_dict(), 'saved_models/generator_%d.pt' % batch)

    ema_G.restore()
    ema_D.restore()

    time_stamp = str('[')+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+str(']')
    elapsed_time = time.time() - begin_time

    _ = make_images(x, y_true, y_pred, batch, make_plot=True)

    print(time_stamp + f' - Batch {batch}/{opt.n_batches}, Val Loss: {val_loss:.4e}, spec err: {spec_err:.4e}, best model: {best_model_id}, epoch time: {elapsed_time:.2f}'
          )