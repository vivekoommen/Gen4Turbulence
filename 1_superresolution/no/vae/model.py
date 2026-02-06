# ---------------------------------------------------------------------------------------------
# Author: Aniruddha Bora
# Date: 09/01/2025
# ---------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------
# Channel Attention Block
# ----------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.relu(self.conv1(y))
        y = self.sigmoid(self.conv2(y))
        return x * y

# ----------------------------------
# Residual Block with Channel Attention
# ----------------------------------
class ResidualBlockCA(nn.Module):
    def __init__(self, channels):
        super(ResidualBlockCA, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.ca    = ChannelAttention(channels)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.ca(out)
        out += residual
        out = self.relu(out)
        return out

# ----------------------------------
# Self-Attention Block (SAGAN style)
# ----------------------------------
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        
    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, H*W, C//8)
        proj_key   = self.key_conv(x).view(B, -1, H * W)                      # (B, C//8, H*W)
        energy = torch.bmm(proj_query, proj_key)                               # (B, H*W, H*W)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(B, -1, H * W)                     # (B, C, H*W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))                # (B, C, H*W)
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out

# ----------------------------------
# Transformer Bottleneck Block
# ----------------------------------
class TransformerBottleneck(nn.Module):
    def __init__(self, embedding_dim, num_layers=2, num_heads=4, dropout=0.1):
        super(TransformerBottleneck, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x: (B, embedding_dim, H, W) -> flatten spatial dims
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        x_transformed = self.transformer(x_flat)         # (B, H*W, C)
        x_out = x_transformed.permute(0, 2, 1).view(B, C, H, W)
        return x_out

# ----------------------------------
# Vector Quantizer (VQ) Module
# ----------------------------------
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0/num_embeddings, 1.0/num_embeddings)
    
    def forward(self, inputs):
        # inputs: (B, D, H, W)
        input_shape = inputs.shape
        flat_input = inputs.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        
        # Add a small epsilon for numerical stability.
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embeddings.weight.t()) + 1e-8)
        
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        quantized = torch.matmul(encodings, self.embeddings.weight)
        quantized = quantized.view(input_shape[0], input_shape[2], input_shape[3], self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss


# ----------------------------------
# Encoder: 3 Downsampling Stages with Transformer Bottleneck
# ----------------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels, base_channels, embedding_dim, num_residuals=2):
        super(Encoder, self).__init__()
        # Stage 1: Downsample by 2
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.rb1 = nn.Sequential(*[ResidualBlockCA(base_channels) for _ in range(num_residuals)])
        # Stage 2: Downsample by 2
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.rb2 = nn.Sequential(*[ResidualBlockCA(base_channels * 2) for _ in range(num_residuals)])
        # Stage 3: Downsample by 2
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.rb3 = nn.Sequential(*[ResidualBlockCA(base_channels * 4) for _ in range(num_residuals)])
        # Self-attention (convolutional) before transformer bottleneck
        self.attn = SelfAttention(base_channels * 4)
        # Transformer bottleneck: enhances long-range dependencies
        self.transformer = TransformerBottleneck(embedding_dim=base_channels * 4, num_layers=2, num_heads=4, dropout=0.1)
        # Final projection to latent space (embedding dimension)
        self.conv4 = nn.Conv2d(base_channels * 4, embedding_dim, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        skip1 = self.rb1(self.conv1(x))         # (B, base_channels, H/2, W/2)
        skip2 = self.rb2(self.conv2(skip1))       # (B, base_channels*2, H/4, W/4)
        x3 = self.rb3(self.conv3(skip2))          # (B, base_channels*4, H/8, W/8)
        x3 = self.attn(x3)
        x3 = self.transformer(x3)
        z = self.conv4(x3)                        # (B, embedding_dim, H/8, W/8)
        return (skip1, skip2), z

# ----------------------------------
# Decoder: 3 Upsampling Stages with Skip Connections & Transformer Block
# ----------------------------------
class Decoder(nn.Module):
    def __init__(self, embedding_dim, base_channels, out_channels, num_residuals=2):
        super(Decoder, self).__init__()
        # Project latent to hidden features
        self.conv1 = nn.Conv2d(embedding_dim, base_channels * 4, kernel_size=3, stride=1, padding=1)
        self.rb1 = nn.Sequential(*[ResidualBlockCA(base_channels * 4) for _ in range(num_residuals)])
        # Transformer block in decoder
        self.transformer = TransformerBottleneck(embedding_dim=base_channels * 4, num_layers=1, num_heads=4, dropout=0.1)
        # Upsample 1: H/8 -> H/4
        self.deconv1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.rb2 = nn.Sequential(*[ResidualBlockCA(base_channels * 2) for _ in range(num_residuals)])
        # Merge with skip2 from encoder (B, base_channels*2, H/4, W/4)
        self.merge_conv2 = nn.Conv2d(base_channels * 2 * 2, base_channels * 2, kernel_size=3, stride=1, padding=1)
        # Upsample 2: H/4 -> H/2
        self.deconv2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.rb3 = nn.Sequential(*[ResidualBlockCA(base_channels) for _ in range(num_residuals)])
        # Merge with skip1 from encoder (B, base_channels, H/2, W/2)
        self.merge_conv1 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, stride=1, padding=1)
        # Upsample 3: H/2 -> H
        self.deconv3 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.rb4 = nn.Sequential(*[ResidualBlockCA(base_channels) for _ in range(num_residuals)])
        # Final reconstruction layer
        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, z, skips):
        skip1, skip2 = skips
        x = self.conv1(z)            # (B, base_channels*4, H/8, W/8)
        x = self.rb1(x)
        x = self.transformer(x)      # Transformer in decoder
        x = self.deconv1(x)          # Upsample to H/4
        x = self.rb2(x)
        x = torch.cat([x, skip2], dim=1)  # Merge skip2
        x = self.merge_conv2(x)
        x = self.deconv2(x)          # Upsample to H/2
        x = self.rb3(x)
        x = torch.cat([x, skip1], dim=1)  # Merge skip1
        x = self.merge_conv1(x)
        x = self.deconv3(x)          # Upsample to H
        x = self.rb4(x)
        x = self.conv_out(x)
        return x

# ----------------------------------
# VQ-VAE: Combining Encoder, Vector Quantizer, and Decoder
# ----------------------------------
class VQVAE(nn.Module):
    def __init__(self, in_channels=10, hidden_channels=128, embedding_dim=256,
                 num_embeddings=512, commitment_cost=0.30):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels, hidden_channels, embedding_dim)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, hidden_channels, in_channels)
    
    def forward(self, x):
        skips, z = self.encoder(x)
        quantized, vq_loss = self.vq(z)
        x_recon = self.decoder(quantized, skips)
        return x_recon, vq_loss

