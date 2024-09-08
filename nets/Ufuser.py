import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
import numbers


class Restormer_CNN_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Restormer_CNN_block, self).__init__()
        self.embed = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")
        self.GlobalFeature = GlobalFeatureExtraction(dim=out_dim, num_heads=8)
        self.LocalFeature = LocalFeatureExtraction(dim=out_dim)
        self.FFN = nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, stride=1, padding=1, bias=False,
                             padding_mode="reflect")

    def forward(self, x):
        x = self.embed(x)
        x1 = self.GlobalFeature(x)
        x2 = self.LocalFeature(x)
        out = self.FFN(torch.cat((x1, x2), 1))
        return out


# class Restormer_decoder_block(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(Restormer_decoder_block, self).__init__()
#         self.embed = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")
#         self.GlobalFeature = CrossGlobalFeatureExtraction(dim=out_dim, num_heads=8)
#         self.LocalFeature = CrossLocalFeatureExtraction(dim=out_dim)
#         self.FFN = nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, stride=1, padding=1, bias=False,
#                              padding_mode="reflect")
#
#     def forward(self, x1, x2):
#         x1 = self.embed(x1)
#         x2 = self.embed(x2)
#
#         global_x = self.GlobalFeature(x1, x2)
#         local_x = self.LocalFeature(x1, x2)
#         out = self.FFN(torch.cat((global_x, local_x), 1))
#         return out


# class CrossGlobalFeatureExtraction(nn.Module):
#     def __init__(self,
#                  dim,
#                  num_heads,
#                  ffn_expansion_factor=1.,
#                  qkv_bias=False, ):
#         super(CrossGlobalFeatureExtraction, self).__init__()
#         self.norm1 = LayerNorm(dim, 'WithBias')
#         self.attn = CrossAttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias, )
#         self.conv = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1)
#         self.norm2 = LayerNorm(dim, 'WithBias')
#         self.mlp = Mlp(in_features=dim, out_fratures=dim,
#                        ffn_expansion_factor=ffn_expansion_factor, )
#
#     def forward(self, x1, x2):
#         x1, x2 = self.attn(self.norm1(x1), self.norm1(x2))
#         x = self.conv(torch.cat((x1, x2), 1))
#         x = x + self.mlp(self.norm2(x))
#         return x


class GlobalFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,
                 qkv_bias=False, ):
        super(GlobalFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias, )
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim, out_fratures=dim,
                       ffn_expansion_factor=ffn_expansion_factor, )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LocalFeatureExtraction(nn.Module):
    def __init__(self,
                 dim=64,
                 num_blocks=2,
                 ):
        super(LocalFeatureExtraction, self).__init__()
        self.Extraction = nn.Sequential(*[ResBlock(dim, dim) for i in range(num_blocks)])

    def forward(self, x):
        return self.Extraction(x)


class CrossLocalFeatureExtraction(nn.Module):
    def __init__(self, dim=64, num_blocks=2):
        super(CrossLocalFeatureExtraction, self).__init__()
        self.extraction = nn.Sequential(*[ResBlock(dim, dim) for _ in range(num_blocks)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        # Extract features
        x1 = self.extraction(x1)
        x2 = self.extraction(x2)
        Fx = self.avgpool(x1)
        Fy = self.avgpool(x2)

        # Step 2: Transformed features average pooling
        F_x = self.avgpool(Fx)
        F_y = self.avgpool(Fy)

        # Step 3: Calculate attention weights
        A_x = torch.sigmoid(self.conv1(F_y * Fx))
        A_y = torch.sigmoid(self.conv1(F_x * Fy))

        x1 = x1 + A_x * x1
        x2 = x2 + A_y * x2
        x = self.conv2(torch.cat((x1, x2), 1))

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,
                      padding_mode="reflect"),
        )

    def forward(self, x):
        out = self.conv(x)
        return out + x


class AttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False, ):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out


# class CrossAttentionBase(nn.Module):
#     def __init__(self,
#                  dim,
#                  num_heads=8,
#                  qkv_bias=False):
#         super(CrossAttentionBase, self).__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
#         self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
#         self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
#
#     def forward(self, x1, x2):
#         b1, c1, h1, w1 = x1.shape
#         b2, c2, h2, w2 = x2.shape
#
#         # Make sure the inputs have the same shape
#         assert b1 == b2 and c1 == c2 and h1 == h2 and w1 == w2, "Input tensors must have the same shape"
#
#         # Compute qkv for both x1 and x2
#         qkv1 = self.qkv(x1)
#         q1, k1, v1 = qkv1.chunk(3, dim=1)
#         qkv2 = self.qkv(x2)
#         q2, k2, v2 = qkv2.chunk(3, dim=1)
#
#         # Reshape and rearrange for multi-head attention
#         q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v1 = rearrange(v1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v2 = rearrange(v2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#
#         # Normalize q and k
#         q1 = torch.nn.functional.normalize(q1, dim=-1)
#         k1 = torch.nn.functional.normalize(k1, dim=-1)
#         q2 = torch.nn.functional.normalize(q2, dim=-1)
#         k2 = torch.nn.functional.normalize(k2, dim=-1)
#
#         # Cross-attention: x1 as q, x2 as k
#         attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
#         attn1 = attn1.softmax(dim=-1)
#         out1 = (attn1 @ v2)
#
#         # Cross-attention: x2 as q, x1 as k
#         attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
#         attn2 = attn2.softmax(dim=-1)
#         out2 = (attn2 @ v1)
#
#         # Rearrange back to original shape
#         out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h1, w=w1)
#         out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h2, w=w2)
#
#         # Apply final projection
#         out1 = self.proj(out1)
#         out2 = self.proj(out2)
#
#         return out1, out2


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self,
                 in_features,
                 out_fratures,
                 ffn_expansion_factor=2,
                 bias=False):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias, padding_mode="reflect")

        self.project_out = nn.Conv2d(
            hidden_features, out_fratures, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class Fusion_Restormer_CNN_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Fusion_Restormer_CNN_block, self).__init__()
        self.embed = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")
        self.GlobalFeature = GlobalFeatureExtraction(dim=out_dim, num_heads=8)
        self.LocalFeature = LocalFeatureExtraction(dim=out_dim)
        self.fusion = FusionBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.embed(x)
        x1 = self.GlobalFeature(x)
        x2 = self.LocalFeature(x)

        out = self.fusion(x1, x2)
        return out

# 思路来源：PointMBF: A Multi-scale Bidirectional Fusion Network for Unsupervised RGB-D Point Cloud Registration
class FusionBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FusionBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp_i = Mlp(in_dim, in_dim)
        self.mlp_v = Mlp(in_dim, in_dim)
        self.cbam_i = cbam_block(in_dim)
        self.cbam_v = cbam_block(in_dim)
        self.CNN_fuse = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        )


    def forward(self, x_i, x_v):
        x_i_mlp = self.mlp_i(x_i)
        x_v_mlp = self.mlp_v(x_v)
        agg_i = x_i_mlp + self.cbam_v(x_v_mlp)
        agg_v = x_v_mlp + self.cbam_i(x_i_mlp)
        x = agg_i + agg_v
        x = self.CNN_fuse(x)
        return x


# add Query block
class QueryPrompter(nn.Module):
    def __init__(self, dim):
        super(QueryPrompter, self).__init__()
        self.dilated_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, dilation=2, padding=2)

    def forward(self, x):
        queries = self.dilated_conv(x)
        return queries


class QueryBasedRestormer_CNN_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(QueryBasedRestormer_CNN_block, self).__init__()
        self.embed = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")
        self.query_prompter = QueryPrompter(out_dim)
        self.attention = CrossAttention(out_dim, num_heads=8)
        self.FFN = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")

    def forward(self, x):
        x = self.embed(x)
        queries = self.query_prompter(x)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2)  # Flatten spatial dimensions
        out = self.attention(x, queries)
        out = out.transpose(1, 2).view(B, C, H, W)  # Reshape back to spatial dimensions
        out = self.FFN(out)
        return out, queries


class QueryBasedRestormer_decoder_block(nn.Module):
    def __init__(self, in_dim, out_dim, num_queries):
        super(QueryBasedRestormer_decoder_block, self).__init__()
        self.embed = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")
        self.attention = CrossAttention(out_dim, num_heads=8)
        self.FFN = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")

    def forward(self, x1, x2):
        x1 = self.embed(x1)
        x2 = self.embed(x2)
        queries1 = self.query_prompter(x1)
        queries2 = self.query_prompter(x2)
        B, C, H, W = x1.shape
        x1 = x1.view(B, C, -1).transpose(1, 2)  # Flatten spatial dimensions
        x2 = x2.view(B, C, -1).transpose(1, 2)  # Flatten spatial dimensions
        out1 = self.attention(x1, queries2)
        out2 = self.attention(x2, queries1)
        out = out1 + out2
        out = out.transpose(1, 2).view(B, C, H, W)  # Reshape back to spatial dimensions
        out = self.FFN(out)
        return out, queries1, queries2


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, queries):
        B, N, C = queries.shape
        q = self.q_proj(queries).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, C)
        out = self.out_proj(out)
        return out


class QueryCrossAttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False):
        super(QueryCrossAttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q_proj1 = nn.Linear(dim, dim)
        self.k_proj1 = nn.Linear(dim, dim)
        self.v_proj1 = nn.Linear(dim, dim)
        self.q_proj2 = nn.Linear(dim, dim)
        self.k_proj2 = nn.Linear(dim, dim)
        self.v_proj2 = nn.Linear(dim, dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x1, x2, q1, q2):
        b1, c1, h1, w1 = x1.shape
        if x2 is not None:
            b2, c2, h2, w2 = x2.shape
            # Make sure the inputs have the same shape
            assert b1 == b2 and c1 == c2 and h1 == h2 and w1 == w2, "Input tensors must have the same shape"
        x1_flat = rearrange(x1, 'b c h w -> b (h w) c')
        q1 = rearrange(q1, 'b c h w -> b (h w) c')
        q2 = rearrange(q2, 'b c h w -> b (h w) c')
        if x2 is not None:
            x2_flat = rearrange(x2, 'b c h w -> b (h w) c')
        q1 = self.q_proj1(q1)
        k1 = self.k_proj1(x1_flat)
        v1 = self.v_proj1(x1_flat)
        q2 = self.q_proj1(q2)
        if x2 is not None:
            k2 = self.k_proj1(x2_flat)
            v2 = self.v_proj1(x2_flat)

        # Reshape and rearrange for multi-head attention
        q1 = rearrange(q1, 'b n (head c) -> b head c n', head=self.num_heads)
        k1 = rearrange(k1, 'b n (head c) -> b head c n', head=self.num_heads)
        v1 = rearrange(v1, 'b n (head c) -> b head c n', head=self.num_heads)
        q2 = rearrange(q2, 'b n (head c) -> b head c n', head=self.num_heads)
        if x2 is not None:
            k2 = rearrange(k2, 'b n (head c) -> b head c n', head=self.num_heads)
            v2 = rearrange(v2, 'b n (head c) -> b head c n', head=self.num_heads)

        # Normalize q and k
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        if x2 is not None:
            k2 = torch.nn.functional.normalize(k2, dim=-1)

        # Cross-attention: x1 as q, x2 as k
        if x2 is not None:
            attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
        else:
            attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        if x2 is not None:
            out1 = (attn1 @ v2)
        else:
            out1 = (attn1 @ v1)

        # Cross-attention: x2 as q, x1 as k
        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        out2 = (attn2 @ v1)

        # Rearrange back to original shape
        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h1, w=w1)
        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h1, w=w1)

        # Apply final projection
        out1 = self.proj(out1)
        out2 = self.proj(out2)

        return out1, out2


# class QueryCrossGlobalFeatureExtraction(nn.Module):
#     def __init__(self,
#                  dim,
#                  num_heads,
#                  ffn_expansion_factor=1.,
#                  qkv_bias=False, ):
#         super(QueryCrossGlobalFeatureExtraction, self).__init__()
#         self.norm1 = LayerNorm(dim, 'WithBias')
#         self.attn = CrossAttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias, )
#         self.conv = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1)
#         self.norm2 = LayerNorm(dim, 'WithBias')
#         self.mlp = Mlp(in_features=dim, out_fratures=dim,
#                        ffn_expansion_factor=ffn_expansion_factor, )
#
#     def forward(self, x1, x2):
#         x1, x2 = self.attn(self.norm1(x1), self.norm1(x2))
#         x = self.conv(torch.cat((x1, x2), 1))
#         x = x + self.mlp(self.norm2(x))
#         return x


# class QueryCrossLocalFeatureExtraction(nn.Module):
#     def __init__(self, dim=64, num_blocks=2):
#         super(QueryCrossLocalFeatureExtraction, self).__init__()
#         self.extraction = nn.Sequential(*[ResBlock(dim, dim) for _ in range(num_blocks)])
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
#         self.conv2 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1)
#
#     def forward(self, x1, x2):
#         # Extract features
#         x1 = self.extraction(x1)
#         x2 = self.extraction(x2)
#         Fx = self.avgpool(x1)
#         Fy = self.avgpool(x2)
#
#         # Step 2: Transformed features average pooling
#         F_x = self.avgpool(Fx)
#         F_y = self.avgpool(Fy)
#
#         # Step 3: Calculate attention weights
#         A_x = torch.sigmoid(self.conv1(F_y * Fx))
#         A_y = torch.sigmoid(self.conv1(F_x * Fy))
#
#         x1 = x1 + A_x * x1
#         x2 = x2 + A_y * x2
#         x = self.conv2(torch.cat((x1, x2), 1))
#
#         return x


class QueryRestormer_CNN_block(nn.Module):
    def __init__(self, in_dim, out_dim, use_query=True):
        super(QueryRestormer_CNN_block, self).__init__()
        self.embed = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")
        self.GlobalFeature = GlobalFeatureExtraction(dim=out_dim, num_heads=8)
        self.LocalFeature = LocalFeatureExtraction(dim=out_dim)
        self.FFN = nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, stride=1, padding=1, bias=False,
                             padding_mode="reflect")
        self.query_prompter = QueryPrompter(out_dim)
        self.use_query = use_query

    def forward(self, x):
        x = self.embed(x)
        x1 = self.GlobalFeature(x)
        x2 = self.LocalFeature(x)
        out = self.FFN(torch.cat((x1, x2), 1))
        if self.use_query:
            query = self.query_prompter(out)
            return out, query
        else:
            return out


class QueryRestormer_decoder_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(QueryRestormer_decoder_block, self).__init__()
        self.embed = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")
        self.att = QueryCrossAttentionBase(out_dim, num_heads=8)
        self.FFN = nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, stride=1, padding=1, bias=False,
                             padding_mode="reflect")

    def forward(self, x1, x2, q1, q2):
        x1 = self.embed(x1)

        if x2 is not None:
            x2 = self.embed(x2)
            out1, out2 = self.att(x1, x2, q1, q2)
        else:
            out1, out2 = self.att(x1, None, q1, q2)
        out = self.FFN(torch.cat((out1, out2), 1))
        return out

# 更新 Ufuser 类
class Ufuser(nn.Module):
    def __init__(self):
        super(Ufuser, self).__init__()
        num_queries = [16384, 4096, 1024, 256]
        channel = [8, 16, 32, 32]
        self.V_en_1 = QueryRestormer_CNN_block(1, channel[0])
        self.V_en_2 = QueryRestormer_CNN_block(channel[0], channel[1])
        self.V_en_3 = QueryRestormer_CNN_block(channel[1], channel[2])
        self.V_en_4 = QueryRestormer_CNN_block(channel[2], channel[3])

        self.I_en_1 = QueryRestormer_CNN_block(1, channel[0])
        self.I_en_2 = QueryRestormer_CNN_block(channel[0], channel[1])
        self.I_en_3 = QueryRestormer_CNN_block(channel[1], channel[2])
        self.I_en_4 = QueryRestormer_CNN_block(channel[2], channel[3])

        # 源代码的fusionblock
        self.f_1 = QueryRestormer_CNN_block(channel[0] * 2, channel[0], use_query=False)
        self.f_2 = QueryRestormer_CNN_block(channel[1] * 2, channel[1], use_query=False)
        self.f_3 = QueryRestormer_CNN_block(channel[2] * 2, channel[2], use_query=False)
        self.f_4 = QueryRestormer_CNN_block(channel[3] * 2, channel[3], use_query=False)
        # self.f_1 = Fusion_Restormer_CNN_block(channel[0] * 2, channel[0])
        # self.f_2 = Fusion_Restormer_CNN_block(channel[1] * 2, channel[1])
        # self.f_3 = Fusion_Restormer_CNN_block(channel[2] * 2, channel[2])
        # self.f_4 = Fusion_Restormer_CNN_block(channel[3] * 2, channel[3])

        self.V_down1 = nn.Conv2d(channel[0], channel[0], kernel_size=3, stride=2, padding=1, bias=False,
                                 padding_mode="reflect")
        self.V_down2 = nn.Conv2d(channel[1], channel[1], kernel_size=3, stride=2, padding=1, bias=False,
                                 padding_mode="reflect")
        self.V_down3 = nn.Conv2d(channel[2], channel[2], kernel_size=3, stride=2, padding=1, bias=False,
                                 padding_mode="reflect")

        self.I_down1 = nn.Conv2d(channel[0], channel[0], kernel_size=3, stride=2, padding=1, bias=False,
                                 padding_mode="reflect")
        self.I_down2 = nn.Conv2d(channel[1], channel[1], kernel_size=3, stride=2, padding=1, bias=False,
                                 padding_mode="reflect")
        self.I_down3 = nn.Conv2d(channel[2], channel[2], kernel_size=3, stride=2, padding=1, bias=False,
                                 padding_mode="reflect")

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(channel[3], channel[2], 4, 2, 1, bias=False),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(channel[2], channel[1], 4, 2, 1, bias=False),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(channel[1], channel[0], 4, 2, 1, bias=False),
            nn.ReLU()
        )

        self.de_1 = QueryRestormer_decoder_block(channel[0], channel[0])
        self.de_2 = QueryRestormer_decoder_block(channel[1], channel[1])
        self.de_3 = QueryRestormer_decoder_block(channel[2], channel[2])
        self.de_4 = QueryRestormer_decoder_block(channel[3], channel[3])

        self.last = nn.Sequential(
            nn.Conv2d(channel[0], 1, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.Sigmoid()
        )

    def forward(self, i, v):
        i_1, i_queries1 = self.I_en_1(i)
        i_2, i_queries2 = self.I_en_2(self.I_down1(i_1))
        i_3, i_queries3 = self.I_en_3(self.I_down2(i_2))
        i_4, i_queries4 = self.I_en_4(self.I_down3(i_3))

        v_1, v_queries1 = self.V_en_1(v)
        v_2, v_queries2 = self.V_en_2(self.V_down1(v_1))
        v_3, v_queries3 = self.V_en_3(self.V_down2(v_2))
        v_4, v_queries4 = self.V_en_4(self.V_down3(v_3))

        f1 = self.f_1(torch.cat((i_1, v_1), dim=1))
        f2 = self.f_2(torch.cat((i_2, v_2), dim=1))
        f3 = self.f_3(torch.cat((i_3, v_3), dim=1))
        f4 = self.f_4(torch.cat((i_4, v_4), dim=1))
        # f1 = self.f_1(i_1, v_1)
        # f2 = self.f_2(i_2, v_2)
        # f3 = self.f_3(i_3, v_3)
        # f4 = self.f_4(i_4, v_4)

        out = self.up4(self.de_4(f4, None, i_queries4, v_queries4))
        out = self.up3(self.de_3(out, f3, i_queries3, v_queries3))
        out = self.up2(self.de_2(out, f2, i_queries2, v_queries2))
        out = self.de_1(out, f1, i_queries1, v_queries1)

        output = self.last(out)
        return output
