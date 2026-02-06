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
from tcunet import Unet2D
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchinfo import summary
import torchprofile
import matplotlib.pyplot as plt

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

def combined_scheduler(optimizer, total_epochs, warmup_epochs, last_epoch=-1):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def make_images(true, pred, epoch):
    # T,P - bs, nt, nx, ny
    sample_id = -2
    t_id = 0

    CMAP = "gray"
    VMIN = 0
    VMAX = 255


    T = true[sample_id, t_id].detach().cpu().numpy()*256
    P = pred[sample_id, t_id].detach().cpu().numpy()*256

    fig, axes = plt.subplots(1,2, figsize=(20,5))
    axes[0].imshow(T, cmap=CMAP, vmin=VMIN, vmax=VMAX)
    axes[0].set_title("True")
    axes[1].imshow(P, cmap=CMAP, vmin=VMIN, vmax=VMAX)
    axes[1].set_title("Pred")

    plt.tight_layout()


    fig.suptitle(f"Epoch: {epoch}", fontsize=22, y=1.2)
    plt.savefig(f"images/{epoch}.png", dpi=150, bbox_inches='tight')
    plt.close()



#########################
begin_time = time.time()

# Please replace with the path to full dataset. 
# The following datasets are subsampled datasets for debugging purposes. 

traj_i = np.load(f"../data/lr8_data.npy").astype(np.float32)/256 
traj_i = np.expand_dims(traj_i, axis=0) 
traj_o = np.load(f"../data/hr_data.npy").astype(np.float32)/256
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
# Par['nt'] = 100 
Par['nx'] = traj_i_train.shape[2]
Par['ny'] = traj_i_train.shape[3]
Par['nf'] = 1
Par['d_emb'] = 128
Par['lb'] = 2
Par['num_epochs'] = 50 # NO converged within 50 epochs (= 9950 iterations of weight updates)
Par['lf'] = 4+1 # For downsampling in time by a factor of 4 

begin_time = time.time()
print('\nTrain Dataset')
x_train, y_train, t_train = preprocess(traj_i_train, traj_o_train, Par)
print('\nValidation Dataset')
x_val, y_val, t_val  = preprocess(traj_i_val, traj_o_val, Par)
print('\nTest Dataset')
x_test, y_test, t_test  = preprocess(traj_i_test, traj_o_test, Par)
print(f"Data Preprocess Time: {time.time() - begin_time:.1f}s")

# sys.exit()

t_min = np.min(t_train)
t_max = np.max(t_train)

Par['inp_scale'] = np.max(x_train) - np.min(x_train)
Par['inp_shift'] = np.min(x_train)
Par['out_scale'] = np.max(y_train) - np.min(y_train)
Par['out_shift'] = np.min(y_train)
Par['t_shift']   = t_min
Par['t_scale']   = t_max - t_min

with open('Par.pkl', 'wb') as f:
    pickle.dump(Par, f)


# Create datasets
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

# Initialize your Unet2D model
model = Unet2D(dim=16, Par=Par, dim_mults=(1, 2, 4, 8)).to(device).to(torch.float32)
summary(model, input_size=((1,)+x_train.shape[1:], (1,)) )

# Adjust the dimensions as per your model's input size
dummy_x = x_train_tensor[0:1].to(device)
dummy_t = t_train_tensor[0:1].to(device)
dummy_input = (dummy_x, dummy_t)

# Profile the model
flops = 2*torchprofile.profile_macs(model, dummy_input)
print(f"FLOPs: {flops:.2e}")

# Define loss function and optimizer
criterion = CustomLoss(Par)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

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

    for x, t, y_true in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        optimizer.zero_grad()
        with autocast():
            y_pred = model(x.to(device), t.to(device))
            loss   = criterion(y_pred, y_true.to(device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

        # Update learning rate
        scheduler.step()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, t, y_true in val_loader:
            with autocast():
                y_pred = model(x.to(device), t.to(device))
                loss   = criterion(y_pred, y_true.to(device))
            val_loss += loss.item()

    val_loss /= len(val_loader)

    # Save the model if validation loss is the lowest so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_id = epoch+1
        torch.save(model.state_dict(), f'models/model_{best_model_id}.pt')

    make_images(y_true, y_pred, epoch)
    
    time_stamp = str('[')+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+str(']')
    elapsed_time = time.time() - begin_time
    print(time_stamp + f' - Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}, best model: {best_model_id}, LR: {scheduler.get_last_lr()[0]:.4e}, epoch time: {elapsed_time:.2f}')

print('Training finished.')
print(f"Training Time: {time.time() - t0:.1f}s")

# Testing loop
model.eval()
test_loss = 0.0
with torch.no_grad():
    for x, t, y_true in test_loader:
        with autocast():
            y_pred = model(x.to(device), t.to(device))
            loss = criterion(y_pred, y_true.to(device))
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4e}')

