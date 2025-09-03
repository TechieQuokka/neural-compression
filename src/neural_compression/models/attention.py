import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with positional encoding"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_position_encoding: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_position_encoding = use_position_encoding
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        if use_position_encoding:
            self.position_encoding = PositionalEncoding2D(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Reshape to sequence format: (B, H*W, C)
        x_seq = x.flatten(2).transpose(1, 2)
        
        # Add positional encoding
        if self.use_position_encoding:
            x_seq = self.position_encoding(x_seq, H, W)
        
        # Multi-head attention
        residual = x_seq
        
        Q = self.w_q(x_seq).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x_seq).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x_seq).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        
        output = self.w_o(attention_output)
        output = self.layer_norm(output + residual)
        
        # Reshape back to image format: (B, C, H, W)
        output = output.transpose(1, 2).view(B, C, H, W)
        
        return output


class SelfAttention(nn.Module):
    """Simplified Self-Attention for computer vision"""
    
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, H * W)
        proj_value = self.value_conv(x).view(B, -1, H * W)
        
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        out = self.gamma * out + x
        return out


class PositionalEncoding2D(nn.Module):
    """2D 위치 인코딩"""
    
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, seq_len, d_model = x.shape
        
        pe = torch.zeros(seq_len, d_model, device=x.device)
        
        # Position indices
        position_h = torch.arange(H, device=x.device).float().unsqueeze(1).repeat(1, W).flatten()
        position_w = torch.arange(W, device=x.device).float().repeat(H)
        
        # Dimension indices
        div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device).float() *
                           -(math.log(self.max_len) / d_model))
        
        # Apply sinusoidal encoding
        pe[:, 0::4] = torch.sin(position_h.unsqueeze(1) * div_term[:d_model//4])
        pe[:, 1::4] = torch.cos(position_h.unsqueeze(1) * div_term[:d_model//4])
        pe[:, 2::4] = torch.sin(position_w.unsqueeze(1) * div_term[:d_model//4])
        pe[:, 3::4] = torch.cos(position_w.unsqueeze(1) * div_term[:d_model//4])
        
        return x + pe.unsqueeze(0)


class ChannelAttention(nn.Module):
    """채널 어텐션 (Squeeze-and-Excitation)"""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Global pooling
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))
        
        # Channel attention
        attention = self.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        
        return x * attention


class SpatialAttention(nn.Module):
    """공간 어텐션"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Spatial attention
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention_input))
        
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x