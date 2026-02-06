import math
from functools import partial

import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange


# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'trilinear'),
        nn.Conv3d(dim, default(dim_out, dim), 3, padding = 1, padding_mode="circular")
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) (d p3)-> b (c p1 p2 p3) h w d', p1 = 2, p2 = 2, p3 = 2),
        nn.Conv3d(dim * 8, default(dim_out, dim), 1, padding_mode="circular")
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        # self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.proj = nn.Conv3d(dim, dim_out, 3, padding = 1, padding_mode="circular")
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1, padding_mode="circular") if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.heads = heads
        if heads is not None:
            self.scale = dim_head ** -0.5
            hidden_dim = dim_head * heads
            self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias = False)

            self.to_out = nn.Sequential(
                nn.Conv3d(hidden_dim, dim, 1),
                LayerNorm(dim)
            )

    def forward(self, x):
        if self.heads is None:
            return x

        b, c, h, w, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z-> b h c (x y z)', h = self.heads), qkv)

        # q = q.softmax(dim = -2)
        # k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y z) -> b (h c) x y z', h = self.heads, x = h, y = w)
        return self.to_out(out)


class SpectralDerivative(nn.Module):
    def __init__(self, par):
        super(SpectralDerivative, self).__init__()
        self.par = par

    def _get_direction_dim(self, direction: str):
        directions_dict = {'x': 2, 'y': 3, 'z': 4}
        dir_lower = direction.lower()
        if dir_lower not in directions_dict:
            raise ValueError(f"Unknown direction '{direction}'. Must be 'x', 'y', or 'z'.")
        return directions_dict[dir_lower]

    def compute_derivative(self, field: torch.Tensor, direction: str = 'x', h: float = 1.0) -> torch.Tensor:

        if not isinstance(h, float):
            raise AssertionError("h must be provided as a float (grid spacing).")

        # 1) figure out which dimension index corresponds to 'direction'
        dim = self._get_direction_dim(direction)

        # 2) number of grid points along that axis
        N = field.size(dim)
        device = field.device
        dtype = field.dtype

        # 3) build the 1D wave‐number array k = 2π * fftfreq(N, d=h).
        k_1d = torch.fft.fftfreq(n=N, d=h, device=device, dtype=dtype)  # shape (N,)
        k_1d = 2.0 * math.pi * k_1d                                         # shape (N,)

        # 4) reshape k_1d so that it can broadcast along all other dims except `dim`
        shape = [1] * field.ndim
        shape[dim] = N
        k = k_1d.view(shape)  # now shape = [1,...,1, N,1,1] with N at index `dim`

        # 5) FFT along the chosen axis
        field_fft = torch.fft.fft(field, dim=dim)  # complex64/complex128 tensor, same shape

        # 6) multiply by (i*k) in Fourier space to get the derivative’s Fourier coefficients
        derivative_fft = field_fft * (1j * k)

        # 7) inverse FFT back to real‐space and take the real part
        deriv = torch.fft.ifft(derivative_fft, dim=dim).real

        return deriv

    def compute_divergence(self, V: torch.Tensor, h_: float = 1.0) -> torch.Tensor:

        Vx = V[:, 0:1, ...]  # shape [BS, 1, Nx, Ny, Nz]
        Vy = V[:, 1:2, ...]  # shape [BS, 1, Nx, Ny, Nz]
        Vz = V[:, 2:3, ...]  # shape [BS, 1, Nx, Ny, Nz]

        # Compute partials via spectral method:
        Vx_x = self.compute_derivative(Vx, direction='x', h=h_)
        Vy_y = self.compute_derivative(Vy, direction='y', h=h_)
        Vz_z = self.compute_derivative(Vz, direction='z', h=h_)

        div = Vx_x + Vy_y + Vz_z
        return div


    def compute_curl(self, f: torch.Tensor, h_: float = 1.0) -> torch.Tensor:

        fx = f[:, 0:1, ...]  # shape [BS, 1, Nx, Ny, Nz]
        fy = f[:, 1:2, ...]  # shape [BS, 1, Nx, Ny, Nz]
        fz = f[:, 2:3, ...]  # shape [BS, 1, Nx, Ny, Nz]

        # Compute partials via spectral method:
        fz_y = self.compute_derivative(fz, direction='y', h=h_)
        fy_z = self.compute_derivative(fy, direction='z', h=h_)
        
        fz_x = self.compute_derivative(fz, direction='x', h=h_)
        fx_z = self.compute_derivative(fx, direction='z', h=h_)

        fy_x = self.compute_derivative(fy, direction='x', h=h_)
        fx_y = self.compute_derivative(fx, direction='y', h=h_)

        Px = fz_y - fy_z
        Py = -(fz_x - fx_z)
        Pz = fy_x - fx_y

        P = torch.cat([Px, Py, Pz], dim=1)
        return P
    

class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        Par,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        attention_heads = 4,
        dim_head = 32
    ):
        super().__init__()
        self.Par = Par

        self.sd_block = SpectralDerivative(Par)

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv3d(input_channels, init_dim, 7, padding = 3, padding_mode="circular")

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                nn.Identity(),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv3d(dim_in, dim_out, 3, padding = 1, padding_mode="circular")
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = nn.Identity() 
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                nn.Identity(),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv3d(dim_out, dim_in, 3, padding = 1, padding_mode="circular")
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = self.Par["nf"] 

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv3d(dim, self.out_dim, 1, padding_mode="circular")
    
    def get_grid(self, shape, device='cuda'):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def forward(self, x, time, x_self_cond = None, use_grid=True):

        x = (x - self.Par['inp_shift'])/self.Par['inp_scale']
        x = x.reshape(-1, self.Par["lb"]*self.Par["nf"], self.Par["nx"], self.Par["ny"], self.Par["nz"])
        time = (time - self.Par['t_shift'])/self.Par['t_scale']

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time) if time is not None else None

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        out = self.final_conv(x)

        f1_f2_f3 = out[:, :3]
        P = out[:, 3:4]
 
        uvw = f1_f2_f3 

        out = torch.cat([uvw, P], dim=1)

        out = out.unsqueeze(1)
        out = out*self.Par['out_scale'] + self.Par['out_shift']
        out = out.reshape(-1, self.Par["nf"], self.Par["nx"], self.Par["ny"], self.Par["nz"])
        return out


if __name__ == "__main__":
    model = Unet3D(dim=16, dim_mults=(1, 2, 4, 8))
    pred = model(torch.rand((16, 3, 64, 64, 64)), time=torch.rand((16, )))
    print('OK')