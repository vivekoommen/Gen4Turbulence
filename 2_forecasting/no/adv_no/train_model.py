# ---------------------------------------------------------------------------------------------
# Author: Vivek Oommen
# Date: 09/01/2025
# ---------------------------------------------------------------------------------------------

import argparse
import os
import sys
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




def get_slices(P, T):
    #P,T - [BS*lf, nf, X, Y, Z]
    _, nf, nx, ny, nz = T.shape
  
    size_x = 32
    size_y = 32
    size_z = 32


    idx = np.random.choice(nx-size_x, 1)[0]
    idy = np.random.choice(ny-size_y, 1)[0]
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
    u_all = torch.tensor(u_all, dtype=DTYPE, device=device)  # [3, ntime, nx, ny, nz]
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


def make_images(ubar, true, pred, epoch, make_plot=False):
    # T,P - bs, nf, nt, nx, ny, nz
    true = true.permute(0,2,1,3,4,5)
    pred = pred.permute(0,2,1,3,4,5)

    sample_id = 0
    f_id = 0
    t_id = 8
    z_id = 64

    CMAP = "viridis"

    TRUE = true
    PRED = pred.detach().cpu()
    
    knyquist, wave_numbers, spectrum_true = compute_tke_spectrum_timeseries_torch( TRUE[sample_id, :3], 2*np.pi, 2*np.pi, 2*np.pi, device=device  )
    knyquist, wave_numbers, spectrum_pred = compute_tke_spectrum_timeseries_torch( PRED[sample_id, :3], 2*np.pi, 2*np.pi, 2*np.pi, device=device  )

    
    s_err = np.mean( (np.log(spectrum_true[:,:96])-np.log(spectrum_pred[:,:96]) )**2 )

    T = true[sample_id, f_id, t_id, :, :, z_id].detach().cpu().numpy()
    P = pred[sample_id, f_id, t_id, :, :, z_id].detach().cpu().numpy()

    VMIN = np.min(T)
    VMAX = np.max(T)
    # VMIN = -1
    # VMAX = +1

    if make_plot:
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

    return s_err


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
res = 128
begin_time = time.time()

traj = np.load("../../data/data.npy") #[nf, nt, nx, ny, nz]
traj = traj.transpose(1,0,2,3,4)[4:]
traj = np.expand_dims(traj, axis=0) #[B, nt, nf, nx, ny, nz]
print(f"traj: {traj.shape}")
print(f"Data Loading Time: {time.time() - begin_time:.1f}s")

traj_train = traj[:, :160]
traj_val   = traj[:, 160:]
traj_test  = traj[:, 160:]

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

Par['num_epochs'] = 200 #50

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


t_min = np.min(time_cond)
t_max = np.max(time_cond)
if Par['lf']==1:
    t_min=0
    t_max=1

MEAN = np.load('../../data/MEAN.npy').reshape(1,-1,1,1,1)
STD  = np.load('../../data/STD.npy').reshape(1,-1,1,1,1)
MIN  = np.load('../../data/MIN.npy').reshape(1,-1,1,1,1)
MAX  = np.load('../../data/MAX.npy').reshape(1,-1,1,1,1)
print(f"MEAN: {MEAN.shape}\nSTD: {STD.shape}\nMIN: {MIN.shape}\nMAX: {MAX.shape}")

Par['inp_shift'] = torch.tensor(MEAN, dtype=DTYPE, device=device)
Par['inp_scale'] = torch.tensor(STD, dtype=DTYPE, device=device)
Par['out_shift'] = torch.tensor(MEAN, dtype=DTYPE, device=device)
Par['out_scale'] = torch.tensor(STD, dtype=DTYPE, device=device)
Par['t_shift']   = torch.tensor(t_min, dtype=DTYPE, device=device)
Par['t_scale']   = torch.tensor(t_max - t_min, dtype=DTYPE, device=device)


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


hr_shape = (opt.crop_size, opt.crop_size)

# generator = MnM(Par).to(device).to(DTYPE)
generator = Unet3D(dim=16, Par=Par, dim_mults=(1, 2, 4, 8), channels=Par['channels']).to(device).to(torch.float32)
print(summary(generator, input_size=((1,)+traj_train_tensor[0,x_idx_train[0:1]].shape[1:], (1,)) ) )
print('Loading model ...')
path_model = '../models/best_model.pt'
generator.load_state_dict(torch.load(path_model))


# Adjust the dimensions as per your model's input size
dummy_x = torch.tensor(torch.randn(1, Par['nf'], Par['lb'], Par['nx'],Par['ny'], Par['nz']),   dtype=DTYPE, device=device)
dummy_t = time_cond_tensor[0:1].to(device)
dummy_input = (dummy_x, dummy_t)
# Profile the model
flops = 2*torchprofile.profile_macs(generator, dummy_input)
print(f"FLOPs: {flops:.2e}")


discriminator = UNetDiscriminatorSN(Par["nf"]).to(device).to(DTYPE)
print(summary(discriminator, input_size=(1,Par["nf"],128,128,128)) )

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

        data = traj_train_tensor[0, x_idx].to(device)
        tt = time_cond_tensor[t_idx].to(device)
        target = traj_train_tensor[0, y_idx].to(device)

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
        for x_idx, y_idx in val_loader:
            x = traj_val_tensor[0, x_idx]        #[BS, lb, nf, nx, ny]
            t = time_cond_tensor[t_idx_val]      #[lf, ]
            y_true = traj_val_tensor[0, y_idx]   #[BS,lf, nf, nx, ny]
            y_pred = rollout(generator, x,t,Par['lb']+Par['LF'], Par, val_batch_size)



            loss = criterion_NO(y_pred, y_true.to(device)).item() 
            s_err = make_images(x, y_true, y_pred, batch)
            val_loss += loss
            spec_err += s_err

    val_loss /= len(val_loader)
    spec_err /= len(val_loader)
    
    ema_G.apply_shadow()
    ema_D.apply_shadow()

    if spec_err < best_loss:
        best_loss = spec_err
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