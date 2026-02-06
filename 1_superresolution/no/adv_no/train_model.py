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

from tcunet import Unet2D

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

    sample_id=-2
    t_id = 0
    for i in range(1):
        x = torch.arange(true.shape[-2]//2)
        axes[0].loglog(x, power_true[sample_id,t_id], label='true', c='black')
        axes[0].loglog(x, power_inp[sample_id,t_id], label='NO', c='blue')
        axes[0].loglog(x, power_pred[sample_id,t_id], label='adv. NO', c='red')
        # axes[i].set_title(f"t: {0}")
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
    axes[3].set_title("adv NO")
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    fig.colorbar(im1, ax=axes[3])
    plt.tight_layout()


    fig.suptitle(f"Epoch: {epoch}, MSE: {err:.2e}", fontsize=22, y=1.2)
    plt.savefig(f"power_spectrum/{epoch}.png", dpi=150, bbox_inches='tight')
    plt.close()

def error_metric(inp, pred,true, epoch, Par, is_plot=True):
    power_inp, power_true, power_pred = compute_power(inp, true, pred)
    err = torch.mean( (torch.log(power_true)-torch.log(power_pred) )**2 )
    f_err = torch.norm(true-pred, p=2)/torch.norm(true, p=2)
    ref_err = torch.norm(true-inp, p=2)/torch.norm(true, p=2)
    if is_plot:
        plot_power_spectrum(power_inp.detach().cpu().numpy(), power_true.detach().cpu().numpy(), power_pred.detach().cpu().numpy(), inp.detach().cpu().numpy(), true.detach().cpu().numpy(), pred.detach().cpu().numpy(), epoch, err)
    return err, f_err, ref_err


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


def preprocess(traj_i, traj_o, Par):
    x = sliding_window_view(traj_i[:,:,:,:], window_shape=Par['lf'], axis=1 ).transpose(0,1,4,2,3).reshape(-1,Par['lf'],Par['nx'], Par['ny'])[:, [0,-1]] # BS, 2, nx, ny
    y = sliding_window_view(traj_o[:,:,:,:], window_shape=Par['lf'], axis=1 ).transpose(0,1,4,2,3).reshape(-1,Par['lf'],Par['nx'], Par['ny'])            # BS, lf, nx, ny
    t = np.linspace(0,1,Par['lf']).reshape(-1,1)

    nt = y.shape[1]
    n_samples = y.shape[0]

    t = np.tile(t, [n_samples,1]).reshape(-1,)                 
    x = np.repeat(x,nt, axis=0)             
    y = y.reshape(y.shape[0]*y.shape[1],1,y.shape[2],y.shape[3]) 


    print('x: ', x.shape)
    print('y: ', y.shape)
    print('t: ', t.shape)
    print()
    return x,y,t


os.makedirs('power_spectrum', exist_ok=True)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(device)

##############################################
begin_time = time.time()
traj_i = np.load(f"../../data/lr8_data.npy").astype(np.float32)/256 
traj_i = np.expand_dims(traj_i, axis=0) 
traj_o = np.load(f"../../data/hr_data.npy").astype(np.float32)/256 
traj_o = np.expand_dims(traj_o, axis=0) 

print(f"traj_i: {traj_i.shape}")
print(f"traj_o: {traj_o.shape}")

print(f"Data Loading Time: {time.time() - begin_time:.1f}s")


nsamples = traj_i.shape[1]
idx1 = int(0.8*nsamples)
idx2 = int(0.9*nsamples)

print(idx1, idx2)

traj_i_train = traj_i[:, :idx1]
traj_i_val   = traj_i[:, idx1:idx2]
traj_i_test  = traj_i[:, idx2:]

traj_o_train = traj_o[:, :idx1]
traj_o_val   = traj_o[:, idx1:idx2]
traj_o_test  = traj_o[:, idx2:]

Par = {}
Par['nx'] = traj_i_train.shape[2]
Par['ny'] = traj_i_train.shape[3]
Par['nf'] = 1
Par['d_emb'] = 128

Par['lb'] = 2
Par['lf'] = 4+1

begin_time = time.time()
print('\nTrain Dataset')
x_train, y_train, t_train = preprocess(traj_i_train, traj_o_train, Par)
print('\nValidation Dataset')
x_val, y_val, t_val  = preprocess(traj_i_val, traj_o_val, Par)
print('\nTest Dataset')
x_test, y_test, t_test  = preprocess(traj_i_test, traj_o_test, Par)
print(f"Data Preprocess Time: {time.time() - begin_time:.1f}s")

t_min = np.min(t_train)
t_max = np.max(t_train)

Par['inp_scale'] = np.max(x_train) - np.min(x_train)
Par['inp_shift'] = np.min(x_train)
Par['out_scale'] = np.max(y_train) - np.min(y_train)
Par['out_shift'] = np.min(y_train)
Par['t_shift']   = t_min
Par['t_scale']   = t_max - t_min

# Create custom datasets
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
t_train_tensor = torch.tensor(t_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

x_val_tensor   = torch.tensor(x_val,   dtype=torch.float32)
t_val_tensor   = torch.tensor(t_val,   dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val,   dtype=torch.float32)

x_test_tensor  = torch.tensor(x_test,  dtype=torch.float32)
t_test_tensor  = torch.tensor(t_test,  dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test,  dtype=torch.float32)

train_dataset = YourDataset(x_train_tensor, t_train_tensor, y_train_tensor)
val_dataset = YourDataset(x_val_tensor, t_val_tensor, y_val_tensor)
test_dataset = YourDataset(x_test_tensor, t_test_tensor, y_test_tensor)

# Define data loaders
train_batch_size = 20
val_batch_size   = 20
test_batch_size  = 20
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size)


hr_shape = (opt.crop_size, opt.crop_size)

generator = Unet2D(dim=16, Par=Par, dim_mults=(1, 2, 4, 8)).to(device).to(torch.float32)
summary(generator, input_size=((1,)+x_train.shape[1:], (1,)) )

# Adjust the dimensions as per your model's input size
dummy_x = x_train_tensor[0:1].to(device)
dummy_t = t_train_tensor[0:1].to(device)
dummy_input = (dummy_x, dummy_t)

# Profile the model
flops = 2*torchprofile.profile_macs(generator, dummy_input)
print(f"FLOPs: {flops:.2e}")
print('Loading model ...')
path_model = '../models/best_model.pt'
generator.load_state_dict(torch.load(path_model))

discriminator = UNetDiscriminatorSN(Par["nf"]).to(device).to(DTYPE)
print(summary(discriminator, input_size=(1,1,128,128)) )

feature_extractor = FeatureExtractor().to(device).to(DTYPE)

# set feature extractor to inference mode  
feature_extractor.eval()

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

best_val_loss = float('inf')

# -----------------
# Training 
# -----------------

batch = opt.batch
while batch < opt.n_batches:
    begin_time = time.time()
    for i, (data, tt, target) in enumerate(train_loader):
        batches_done = batch + i

        tt = tt.to(device)
        imgs_lr = data.to(device) 
        imgs_hr = target.to(device) #[BS, nf, lf, nx, ny]

        # ---------------------
        # Training Generator
        # ---------------------

        optimizer_G.zero_grad()

        gen_hr = generator(imgs_lr, tt)

        loss_pixel = criterion_pixel(gen_hr, imgs_hr)

        valid = torch.ones((imgs_hr.size(0), 1, *imgs_hr.shape[-2:]), requires_grad=False).to(device)
        fake = torch.zeros((imgs_hr.size(0), 1, *imgs_hr.shape[-2:]), requires_grad=False).to(device)

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
        loss_content = sum(criterion_content(gen_f, real_f) * w for gen_f, real_f, w in zip(gen_features, real_features, [0.1, 0.1, 1, 1, 1]))

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

    batch = batches_done + 1

    val_loss = 0.0
    spec_err = 0.0
    field_err = 0.0
    ref_err = 0.0
    plot_flag = True
    with torch.no_grad():
        for x, tt, y_true in val_loader:
            y_pred = generator(x.to(device), tt.to(device))
            loss = criterion_NO(y_pred, y_true.to(device) ).item() 
            s_err, f_err, r_err = error_metric(y_pred, y_pred, y_true.to(device), batch, Par, plot_flag)
            val_loss += loss
            spec_err += s_err.item()
            field_err += f_err.item()
            ref_err += r_err.item()
            plot_flag = False

    val_loss /= len(val_loader)
    spec_err /= len(val_loader)
    field_err /= len(val_loader)
    ref_err /= len(val_loader)
    

    ema_G.apply_shadow()
    ema_D.apply_shadow()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_id = batch
        torch.save(generator.state_dict(), f'saved_models/best_generator_{best_model_id}.pt')

    torch.save(generator.state_dict(), 'saved_models/generator_%d.pth' % batch)

    ema_G.restore()
    ema_D.restore()

    time_stamp = str('[')+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+str(']')
    elapsed_time = time.time() - begin_time

    print(time_stamp + f' - Batch {batch}/{opt.n_batches}, Val Loss: {val_loss:.4e}, spec err: {spec_err:.4e}, field err: {field_err:.4e}, ref err: {ref_err:.4e}, best model: {best_model_id}, epoch time: {elapsed_time:.2f}'
          )