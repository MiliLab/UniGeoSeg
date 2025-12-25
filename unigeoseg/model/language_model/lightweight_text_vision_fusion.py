"""
轻量级文本-视觉动态融合方案
在get_SEG_embedding中集成，不影响现有流程
"""

import torch
import torch.nn as nn

class LightweightTextVisionFusion(nn.Module):
    """
    轻量级文本-视觉动态融合模块
    只在需要时激活，对现有流程影响最小
    """
    def __init__(self, text_dim=2048, vision_dim=1024, fusion_dim=256, enable_dynamic=True):
        super().__init__()
        self.enable_dynamic = enable_dynamic
        self.fusion_dim = fusion_dim
        
        if self.enable_dynamic:
            # 轻量级文本特征压缩器
            self.text_compressor = nn.Sequential(
                nn.Linear(text_dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, fusion_dim)
            )
            
            # 轻量级视觉特征投影器
            self.vision_projector = nn.Sequential(
                nn.Linear(vision_dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, fusion_dim)
            )
            
            # 动态权重生成器
            self.dynamic_weight_generator = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 4),
                nn.ReLU(),
                nn.Linear(fusion_dim // 4, 1),
                nn.Sigmoid()  # 输出0-1之间的权重
            )
            
            # 融合投影器
            self.fusion_projector = nn.Linear(fusion_dim, text_dim)

    def forward(self, text_features, vision_features=None):
        """
        Args:
            text_features: [hidden_size] 文本特征
            vision_features: [vision_dim] 视觉特征（可选）
        Returns:
            enhanced_text_features: [hidden_size] 增强的文本特征
        """
        if not self.enable_dynamic or vision_features is None:
            return text_features
        
        # 压缩文本特征
        compressed_text = self.text_compressor(text_features)  # [fusion_dim]
        
        # 投影视觉特征
        projected_vision = self.vision_projector(vision_features)  # [fusion_dim]
        
        # 计算动态权重
        dynamic_weight = self.dynamic_weight_generator(compressed_text)  # [1]
        
        # 动态融合
        fused_features = compressed_text + dynamic_weight * projected_vision  # [fusion_dim]
        
        # 投影回原始维度
        enhanced_text = self.fusion_projector(fused_features)  # [text_dim]
        
        # 残差连接
        return text_features + 0.1 * enhanced_text  # 轻量级融合


class AdaptiveLatentFusion(nn.Module):
    """
    自适应latent融合，在现有latent基础上增加动态性
    """
    def __init__(self, hidden_size=2048, num_latents=8, enable_adaptive=True):
        super().__init__()
        self.enable_adaptive = enable_adaptive
        self.num_latents = num_latents
        
        if self.enable_adaptive:
            # 自适应权重生成器
            self.adaptive_weight_generator = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 8),
                nn.ReLU(),
                nn.Linear(hidden_size // 8, num_latents),
                nn.Softmax(dim=-1)
            )

    def forward(self, text_features, global_latents):
        """
        Args:
            text_features: [hidden_size] 文本特征
            global_latents: [num_latents, hidden_size] 全局latent
        Returns:
            enhanced_features: [hidden_size] 增强特征
        """
        if not self.enable_adaptive:
            # 使用原有的固定权重方法
            similarity = torch.matmul(global_latents, text_features.unsqueeze(-1))
            weights = torch.softmax(similarity, dim=0)
            latent_contribution = torch.sum(global_latents * weights, dim=0)
            return 0.95 * text_features + 0.05 * latent_contribution
        
        # 自适应权重方法
        adaptive_weights = self.adaptive_weight_generator(text_features)  # [num_latents]
        latent_contribution = torch.sum(global_latents * adaptive_weights.unsqueeze(-1), dim=0)
        
        # 动态调整融合比例
        fusion_strength = adaptive_weights.max()  # 基于最大权重调整融合强度
        return (1 - 0.05 * fusion_strength) * text_features + 0.05 * fusion_strength * latent_contribution







