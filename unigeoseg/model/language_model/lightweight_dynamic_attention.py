"""
轻量级动态跨模态注意力集成方案
"""

import torch
import torch.nn as nn
from torch import einsum
from .einops_exts import rearrange_many

class DynamicPerceiverAttention(nn.Module):
    """
    在原有PerceiverAttention基础上添加动态权重机制
    保持原有接口，只增加动态调整功能
    """
    def __init__(self, dim, dim_head=64, heads=8, enable_dynamic=True):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.enable_dynamic = enable_dynamic
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        
        # 轻量级动态权重生成器
        if self.enable_dynamic:
            self.dynamic_weight_generator = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.ReLU(),
                nn.Linear(dim // 4, heads),  # 为每个注意力头生成权重
                nn.Softmax(dim=-1)
            )

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latents (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents[:, :, 0:1]), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        # 计算注意力
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        
        # 动态权重调整（轻量级）
        if self.enable_dynamic:
            # 基于latent特征生成动态权重
            dynamic_weights = self.dynamic_weight_generator(latents.mean(dim=2))  # [b, T, heads]
            dynamic_weights = dynamic_weights.unsqueeze(-1).unsqueeze(-1)  # [b, T, heads, 1, 1]
            
            # 应用动态权重到注意力
            attn = attn * dynamic_weights

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)


class LightweightDynamicD_Projector(nn.Module):
    """
    轻量级动态D_Projector，保持原有接口
    """
    def __init__(self, dim=512, depth=3, dim_head=64, heads=8, ff_mult=4, enable_dynamic=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        DynamicPerceiverAttention(dim=dim, dim_head=dim_head, heads=heads, enable_dynamic=enable_dynamic),
                        nn.Sequential(
                            nn.LayerNorm(dim),
                            nn.Linear(dim, int(dim * ff_mult), bias=False),
                            nn.GELU(),
                            nn.Linear(int(dim * ff_mult), dim, bias=False),
                        )
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, latents, x):
        """
        保持原有接口不变
        """
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        
        return self.norm(latents)[:, 0, 0:1]







