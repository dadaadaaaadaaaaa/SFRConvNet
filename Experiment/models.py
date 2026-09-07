"""MACNet 的 PyTorch 网络结构库。

本模块定义旧版黑洞质量条件残差网络、Transformer 与当前 SFRConvNet。
模型共同处理多通道二维气体场和网格坐标，并可按 cache 契约接收全局条件标量；
输出既可表示直接 SFR，也可表示 signed closure，但物理反变换与 ``B+C_hat``
组合由训练和分析模块负责。
"""
import math
import torch
import torch.nn as nn
from typing import List, Tuple, Optional


# =============================================
# Module Definition
# =============================================
'''
Legacy accretion models:
``MACNetRes_mbh`` concatenates explicit BH mass after spatial pooling, while
``MACNetFiLM`` uses explicit BH mass to modulate its residual blocks. These
semantics are separate from ``SFRConvNet``, whose optional FiLM scalar is the
transformed Kennicutt-Schmidt SFR baseline.
'''
class Residual(nn.Module): # 残差块

    def __init__(self, in_channels, out_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides) # 卷积层1 
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) # 卷积层2
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides) # 1*1卷积层，目的是使输入X与Y的维度强行对齐。只会在阶段中第一个块使用。
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels) # 批归一化层1
        self.bn2 = nn.BatchNorm2d(out_channels) # 批归一化层2

    def forward(self, X):
        Y = nn.functional.relu(self.bn1(self.conv1(X))) # 卷积层1的输出 
        Y = self.bn2(self.conv2(Y)) # 卷积层2的输出
        if self.conv3:
            X = self.conv3(X) # 如果使用1*1卷积，则对X应用卷积层3
        return nn.functional.relu(Y + X) # 将残差块的输出与输入相加，并使用ReLU激活函数


def resnet_block(num_residuals, in_channels, out_channels, half=False): # 将多个残差快串联成一个残差块，命名为layer
    layers = []
    for i in range(num_residuals):
        if i == 0: # layer中第一个块，会使用卷积层3（1*1卷积层）
            if half:
                layers.append(Residual(in_channels, out_channels, use_1x1conv=True, strides=2)) # 如果half为True，则会用步幅为2的卷积层来对输入降维
            else:
                layers.append(Residual(in_channels, out_channels, use_1x1conv=True))
        else:  
            layers.append(Residual(out_channels, out_channels))
    return nn.Sequential(*layers) # 将layers中的所有模块组合成一个序列


class MACNetRes_mbh(nn.Module):
    """Legacy accretion model conditioned explicitly on black-hole mass."""

    def __init__(self, in_channels=15): # 输入通道数=15，即输入15种2D物理场   # 输入二维物理网格，输出256通道的高维特征图，构建一个res_block
        super().__init__()
        in_channels = in_channels
        # Define the Residual blocks
        blocks = []
        res_arch = ((2, 32, True), (2, 64, True), (2, 128, False), (2, 256, False)) # 建立4个layer，每个阶段都是2个块，输出通道依次翻倍。前两次空间分辨率减半，后两次不变。
                                                                                    # 输入的二维物理网格被浓缩成256通道的高维特征图
        for num_residuals, out_channels, half in res_arch:
            # print(in_channels, out_channels, num_residuals, first_block)
            blocks.append(
                resnet_block(
                    num_residuals=num_residuals,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    half=half,
                )
            )
            in_channels = out_channels # 后一个输入通道数=前一个输入通道数
        self.res_blocks = nn.Sequential(*blocks) # 将blocks中的所有模块组合成一个序列

        # Define the Linear blocks
        self.flatten_block = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()) # 全局池化，对每个通道在空间维度上求平均值，浓缩成1*1的特征图。然后全局压平，变成1*256的特征图。
        self.liner_block = nn.Sequential(
            nn.Linear(in_channels + 1, in_channels // 2), # 线性层，输入维度=256+1=257，输出维度=256/2=128
            nn.LeakyReLU(),
            nn.Linear(in_channels // 2, 1), # 线性层，输入维度=128，输出维度=1
        )

    def forward(self, x, mbh):
        mbh = torch.tensor(mbh, dtype=torch.float32, device=x.device) if not isinstance(mbh, torch.Tensor) else mbh
        mbh = mbh.float().to(x.device).view(-1, 1)

        x = self.res_blocks(x)
        x = self.flatten_block(x)
        x = torch.cat((x, mbh), dim=1) # Legacy model: concatenate the explicit BH-mass scalar after pooling.
        x = self.liner_block(x)
        return x.view(-1) # 输出最终预测值



class FiLMResidual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, strides=1): # 和Residual一样
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X, gamma, beta): # 引入gamma和beta
        Y = nn.functional.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y)) # 依旧经历两个卷积层

        # FiLM modulation
        if gamma is not None and beta is not None:
            # gamma, beta: [B, C] -> [B, C, 1, 1] # reshape gamma & beta，将单一数值广播到整个二维平面上
            Y = (1 + gamma.view(-1, self.out_channels, 1, 1)) * Y + beta.view(-1, self.out_channels, 1, 1) # 使用gamma和beta对Y进行调制，Y  = (1 + gamma) * Y + beta

        if self.conv3:
            X = self.conv3(X)
        return nn.functional.relu(Y + X) # 将调制后的Y与X相加，并使用ReLU激活函数


class MACNetFiLM(nn.Module):
    """Legacy accretion model whose FiLM conditioner is black-hole mass."""
    def __init__(self, in_channels=15):
        super().__init__()
        in_c = in_channels
        self.blocks = nn.ModuleList()
        self.channel_counts = [] # 记录每个阶段的输出通道数，以便后续生成对应维度的gamma和beta

        # Consistent with MACNetRes_mbh 与前文一致
        res_arch = ((2, 32, True), (2, 64, True), (2, 128, False), (2, 256, False))

        for num_residuals, out_channels, half in res_arch:
            for i in range(num_residuals):
                stride = 1
                use_1x1 = False
                curr_in = out_channels

                if i == 0:
                    stride = 2 if half else 1
                    use_1x1 = True
                    curr_in = in_c

                self.blocks.append(FiLMResidual(curr_in, out_channels, use_1x1conv=use_1x1, strides=stride))
                self.channel_counts.append(out_channels)

            in_c = out_channels

        self.total_features = sum(self.channel_counts) # 统计多少个通道需要被控制
        # FiLM Generator
        self.film_generator = nn.Sequential(nn.Linear(1, 64), nn.LeakyReLU(), nn.Linear(64, self.total_features * 2)) # Input scalar is explicit BH mass; output is per-block gamma/beta.

        self.flatten_block = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()) # 池化操作，与前文一致

        # Final regression from last block features
        last_out_channels = res_arch[-1][1]
        self.liner_block = nn.Sequential(
            nn.Linear(last_out_channels, last_out_channels // 2),
            nn.LeakyReLU(),
            nn.Linear(last_out_channels // 2, 1), # 输入维度=256（而非257），输出一个预测值
        )

    def forward(self, x, mbh):
        mbh = torch.tensor(mbh, dtype=torch.float32, device=x.device) if not isinstance(mbh, torch.Tensor) else mbh
        mbh = mbh.float().to(x.device).view(-1, 1)

        film_params = self.film_generator(mbh)
        gammas_betas = torch.split(film_params, [c * 2 for c in self.channel_counts], dim=1) # 按照每个阶段的通道数对gamma和beta进行切分，切分成8组

        for block, params in zip(self.blocks, gammas_betas):
            gamma, beta = torch.chunk(params, 2, dim=1) # 分离gamma和beta
            x = block(x, gamma, beta) # 将gamma和beta传入每个块中进行调制

        x = self.flatten_block(x)
        x = self.liner_block(x)
        return x.view(-1)


# Continuous Fourier feature for (r, theta) coordinates
'''
卷积神经网络无法理解当前输入特征图中每个像素点的空间位置关系（具有平移不变性），所以我们需要为每个像素点提供一个位置编码，告诉模型这个像素点在空间中的位置。
这里我们使用傅里叶特征来编码每个像素点的极坐标位置（r, theta）。傅里叶特征通过对输入坐标进行不同频率的正弦和余弦变换，生成一个高维的特征向量，使得模型能够捕捉到不同尺度的空间关系。
'''
"""
    Encode continuous (r, theta) with fixed (non-trainable) sinusoidal features.
    Works across arbitrary grids/resolutions.
"""

class FourierPositionalEncoding2D(nn.Module):
    def __init__(
        self,
        num_bands: int = 32,
        max_frequency: float = 64.0,
        include_input: bool = True,
        theta_pi_periodic: bool = True,
        r_log_scale: bool = True,
        eps: float = 1e-6,
    ):
        """
        Args:
            num_bands: number of frequency bands per coordinate
            max_frequency: highest frequency (log-spaced up to this)
            include_input: whether to include raw [r, theta] (normalized) in output
        """
        super().__init__()
        self.num_bands = num_bands
        self.include_input = include_input
        # log-spaced frequencies: [1, ..., max_frequency]
        self.theta_pi_periodic = theta_pi_periodic
        self.r_log_scale = r_log_scale
        self.eps = eps
        # 非持久 buffer 不进入 checkpoint；每个 trial 独立建立坐标缓存。
        self.register_buffer("_fixed_r", None, persistent=False)
        self.register_buffer("_fixed_theta", None, persistent=False)
        self.register_buffer("_fixed_features", None, persistent=False)
        self.register_buffer(
            "freq_bands",
            torch.logspace(0.0, math.log10(max_frequency), steps=num_bands), # 生成num_bands个频率，呈对数空间分布，从1到max_frequency
                                                                         )   # 低轨道频率捕捉大尺度空间，高频率捕捉小尺度空间

    @staticmethod
    def _wrap_pi(theta: torch.Tensor) -> torch.Tensor: # 将角度映射到[0, π)范围内，适用于θ具有π周期性的情况
        # map to [0, π)
        return torch.remainder(theta, math.pi)

    """
    计算输出特征的维度：如果include_input为True，则包含原始的r和theta（2维），每个坐标经过num_bands个频率的sin和cos变换后会产生2*num_bands维特征，
    所以总维度是2（原始输入） + 4*num_bands（sin和cos特征）。
    """
    @property
    def out_dim(self) -> int:
        base = 2 if self.include_input else 0  # r, theta
        # for each of r,theta -> sin&cos per band -> 2 * num_bands
        return base + 2 * 2 * self.num_bands

    def forward(self, r: torch.Tensor, theta: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        r, theta: (B, 1, H, W)
        Returns: (B, Dpos, H, W)
        """
        B, _, H, W = r.shape
        if self._fixed_features is not None and normalize and not r.requires_grad and not theta.requires_grad:
            if (r.shape[1:] == self._fixed_r.shape[1:]
                    and r.device == self._fixed_r.device
                    and r.dtype == self._fixed_r.dtype
                    and torch.equal(r, self._fixed_r.expand_as(r))
                    and torch.equal(theta, self._fixed_theta.expand_as(theta))):
                return self._fixed_features.expand(B, -1, -1, -1)

        # r normalization
        if self.r_log_scale: # 对r取对数并归一化，以更好地处理跨越多个数量级的半径值。每个r都属于[0,1]
            r_safe = torch.clamp(r, min=self.eps)
            r_n = (torch.log(r_safe) - torch.log(r_safe).amin((2, 3), True)) / (
                torch.log(r_safe).amax((2, 3), True) - torch.log(r_safe).amin((2, 3), True) + self.eps
            )
        else:
            if normalize:
                r_n = (r - r.amin((2, 3), True)) / (r.amax((2, 3), True) - r.amin((2, 3), True) + self.eps)
            else:
                r_n = (r - r.min()) / (r.max() - r.min() + self.eps)

        # theta handling
        if self.theta_pi_periodic: # 如果θ具有π周期性，我们将θ映射到[0, π)范围内，并使用2倍频率的傅里叶特征来捕捉这种周期性。这样可以更有效地表示θ的空间关系。
            theta = self._wrap_pi(theta)  # [0, π)
            # theta_unit = theta / math.pi  # [0,1) for optional raw export
            # even harmonics: 2k * theta
            # we reuse log-spaced freq_bands on the base, then multiply by 2
            fb_theta = (2.0 * self.freq_bands).view(1, -1, 1).to(theta.device)
        else:
            # fallback to 2π periodic if ever needed # 如果θ没有π周期性，我们将θ映射到[0, 2π)范围内，并使用原始频率的傅里叶特征来表示θ。这样可以捕捉到θ的完整周期性。
            theta = torch.remainder(theta, 2 * math.pi)
            # theta_unit = theta / (2 * math.pi)
            fb_theta = self.freq_bands.view(1, -1, 1).to(theta.device)

        # if normalize:
        #     r = _minmax_norm(r)  # [0,1] per-sample
        #     # normalize theta to [0,1] by mapping [-pi, pi] or [0, 2pi] if needed
        #     # 如果你的 theta 已经是 [0, 2π) 或 [-π, π]，这里做一个鲁棒归一化：
        #     th_min = theta.amin(dim=(2, 3), keepdim=True)
        #     th_max = theta.amax(dim=(2, 3), keepdim=True)
        #     theta = (theta - th_min) / (th_max - th_min + 1e-6)

        r_flat = r_n.view(B, 1, H * W) # 去掉H和W的二维结构，展平为H*W的一维结构
        t_flat = theta.view(B, 1, H * W)

        # (1, num_bands, 1)
        fb_r = self.freq_bands.view(1, -1, 1).to(r.device)

        # project
        r_proj = r_flat * fb_r  # (B, num_bands, HW)
        t_proj = t_flat * fb_theta # (B, num_bands, HW)

        # sin/cos # 傅里叶激活
        r_sin = torch.sin(2 * math.pi * r_proj)
        r_cos = torch.cos(2 * math.pi * r_proj)
        t_sin = torch.sin(2 * t_proj)
        t_cos = torch.cos(2 * t_proj)

        pos_list = []
        if self.include_input:
            pos_list += [r_flat, t_flat]  # (B,1,HW) each
        pos_list += [r_sin, r_cos, t_sin, t_cos]  # (B,num_bands,HW) each 将1个r，1个θ，num_bands个r_sin,r_cos,theta_sin,theta_cos拼接起来，形成一个高维的特征向量，告诉模型每个像素点在空间中的位置关系

        pos = torch.cat(pos_list, dim=1).view(B, -1, H, W)  # (B, Dpos, HW) 重新将展平的H*W恢复为二维的H和W，形状为(B, Dim_pos, H, W)
        return pos


# Generic scalar FiLM conditioner. In the current SFR pipeline the scalar is
# SFRConvNet 的标量条件现在是 log10(M_filter)；旧吸积模型仍使用独立黑洞质量代码。
class FiLMConditioner(nn.Module):  # Transformer架构，独立于前面的cnn架构
    """
    Map scalar baseline (e.g. KS SFR) to (gamma, beta) for feature modulation.
    Can produce one pair per Transformer layer.
    """

    def __init__(self, n_layers: int, d_model: int, hidden: int = 128):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * n_layers * d_model),
        )

    def forward(self, y_baseline: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        y_baseline: (B,) or (B,1)
        Returns lists gamma_list, beta_list each len=n_layers,
        with tensors of shape (B, 1, d_model) ready to broadcast over tokens.
        """
        if y_baseline.dim() == 1:
            y_baseline = y_baseline.unsqueeze(-1)
        B = y_baseline.shape[0]
        out = self.net(y_baseline)  # (B, 2*n_layers*d_model)
        out = out.view(B, self.n_layers, 2, self.d_model)  # (B,L,2,D)
        gammas = out[:, :, 0, :].unsqueeze(1)  # (B,1,L,D)
        betas = out[:, :, 1, :].unsqueeze(1)  # (B,1,L,D)
        # split by layer
        gamma_list = [gammas[:, :, i, :] for i in range(self.n_layers)]  # each (B,1,D)
        beta_list = [betas[:, :, i, :] for i in range(self.n_layers)]
        return gamma_list, beta_list


# Transformer blocks (Pre-Norm)
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, p_drop: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, p_drop: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=p_drop, batch_first=True)
        self.drop1 = nn.Dropout(p_drop)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, p_drop)

    def forward(
        self,
        x: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
    ):
        # x: (B, N, D)
        # FiLM pre-attn
        if gamma is not None and beta is not None:
            x = (1 + gamma) * x + beta  # broadcast over tokens

        # Self-attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(h)

        # FiLM pre-MLP
        if gamma is not None and beta is not None:
            x = (1 + gamma) * x + beta

        # MLP
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x


# Main model
class AccretionTransformer(nn.Module):
    """
    Input:
        x: (B, C_in, H, W)   -- multi-channel galaxy "image"
        r: (B, 1, H, W)      -- radial coordinate
        theta: (B, 1, H, W)  -- azimuthal coordinate
        y_baseline: (B,)     -- scalar baseline
    Output:
        y: (B,)              -- scalar (e.g., log SFR)
    """

    def __init__(
        self,
        c_in: int,
        d_model: int = 256, 
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = 1024,
        pos_num_bands: int = 32, 
        pos_max_freq: float = 64.0,
        p_drop: float = 0.1,
    ):
        super().__init__()

        # Positional encoding from (r, theta)
        self.posenc = FourierPositionalEncoding2D(
            num_bands=pos_num_bands,
            max_frequency=pos_max_freq,
            include_input=True,
            theta_pi_periodic=False,
        )
        d_pos = self.posenc.out_dim

        # Projections to d_model
        # Use sum fusion: content_proj(x) + pos_proj(pos)
        self.content_proj = nn.Conv2d(c_in, d_model, kernel_size=1)
        self.pos_proj = nn.Conv2d(d_pos, d_model, kernel_size=1)
        self.drop = nn.Dropout(p_drop)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, p_drop=p_drop) for _ in range(n_layers)]
        )

        # FiLM conditioner (per-layer γ/β)
        self.film = FiLMConditioner(n_layers=n_layers, d_model=d_model, hidden=128)

        # Head for regression
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, C_in, H, W)
        r: torch.Tensor,  # (B, 1,    H, W)
        theta: torch.Tensor,  # (B, 1,    H, W)
        log_mass_filter: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """
        x: (B, C_in, H, W), r, theta: (B, 1, H, W), log_mass_filter: (B,)
        """
        B, C, H, W = x.shape

        # Positional features from (r, theta)
        pos = self.posenc(r, theta, normalize=True)  # (B, Dpos, H, W)

        # Project and fuse (sum)
        # content: (B, d_model, H, W)
        # pos:     (B, d_model, H, W)
        feat = self.content_proj(x) + self.pos_proj(pos)
        feat = self.drop(feat)

        # Flatten to tokens
        tokens = feat.flatten(2).transpose(1, 2)  # (B, N=H*W, D)

        # CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, N+1, D)

        # 由 log10(M_filter) 生成逐层 FiLM 参数。
        gammas, betas = self.film(log_mass_filter)  # lists of (B,1,D)

        # Pass through layers
        x_tok = tokens
        for i, blk in enumerate(self.blocks):
            gamma = gammas[i].expand(B, x_tok.size(1), -1)  # (B,N+1,D)
            beta = betas[i].expand(B, x_tok.size(1), -1)
            x_tok = blk(x_tok, gamma=gamma, beta=beta)

        # Readout: CLS
        cls_out = x_tok[:, 0, :]  # (B, D)
        y = self.head(cls_out).squeeze(-1)  # (B,)
        return y

# =============================================
# CNN-based model with Fourier PE + FiLM (append to models.py)
# =============================================

'''
通道注意力机制
在普通CNN中，所有通道的地位都是平等的，但实际上，某个通道可能及其重要（比如代表“激波”的通道，而某个噪音通道则不重要
因此需要为网络赋予通道注意力
'''
class SEBlock(nn.Module):
    """Squeeze-and-Excitation: lightweight channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8) # 为每一个通道生成一个权重。所有权重对应的矩阵为w，通过一个MLP生成
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        w = self.pool(x).view(B, C) # 强行压缩(B,C,H,W) -> (B, C, 1, 1) -> (B,C)
        w = self.fc(w).view(B, C, 1, 1) 
        return x * w # 每个通道乘以对应的权重，实现通道注意力机制


class ConvFiLMBlock(nn.Module):
    """
    Pre-activation ResNet block with:
      - BN → ReLU → Conv3×3 → BN → ReLU → Conv3×3
      - FiLM modulation (γ, β) applied after the first BN
      - SE channel attention at the end
      - Optional 1×1 skip projection + stride-2 downsampling
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        p_drop: float = 0.1,
        se_reduction: int = 4,
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch) # bn: 归一化
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.se = SEBlock(out_ch, reduction=se_reduction)
        self.drop = nn.Dropout2d(p_drop)

        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        identity = self.skip(x)

        out = self.bn1(x)
        out = torch.relu(out)
        out = self.conv1(out) # 归一化 -> 激活 -> 卷积，比老版的卷积->归一化->激活更早引入非线性，能让梯度回传更加顺畅

        out = self.bn2(out)
        # FiLM modulation: scale + shift after normalization
        if gamma is not None and beta is not None:
            out = (1 + gamma) * out + beta # 第二层卷积前引入gamma和beta，也就是FiLM调制
        out = torch.relu(out)
        out = self.drop(out) # 卷积层之间引入dropout，让模型随机丢弃一些特征，防止过拟合
        out = self.conv2(out)

        out = self.se(out) # 卷积层2后引入SEBlock，进行通道注意力机制，让模型学会关注重要的特征通道
        return out + identity


class ConvFiLMConditioner(nn.Module): # FiLM调制
    """
    Map scalar baseline → per-stage (γ, β) for 2D feature maps.
    Each γ/β has shape (B, C_stage, 1, 1) to broadcast over spatial dims.
    """

    def __init__(self, stage_channels: List[int], hidden: int = 128):
        super().__init__()
        self.stage_channels = stage_channels
        total_out = 2 * sum(stage_channels)  # γ + β for each stage
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, total_out),
        )

    def forward(
        self, y_baseline: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if y_baseline.dim() == 1:
            y_baseline = y_baseline.unsqueeze(-1)
        B = y_baseline.shape[0]
        out = self.net(y_baseline)  # (B, total_out)

        gamma_list, beta_list = [], []
        offset = 0
        for ch in self.stage_channels:
            gamma_list.append(out[:, offset : offset + ch].view(B, ch, 1, 1))
            offset += ch
            beta_list.append(out[:, offset : offset + ch].view(B, ch, 1, 1))
            offset += ch
        return gamma_list, beta_list


class SFRConvNet(nn.Module):
    """
    CNN backbone with Fourier positional encoding + FiLM conditioning.

    Architecture:
      1. Fourier PE of (r, θ) → fused with content channels via concatenation
      2. Stem conv to lift to base_ch
      3. Optional FiLM conditioning from log10(M_filter)
         Each stage: ConvFiLMBlock × depth, stride-2 downsampling at stage boundaries
      4. Global average pooling → regression head → scalar output

    Input signature matches AccretionTransformer exactly:
        x:       (B, C_in, H, W)
        r:       (B, 1,    H, W)
        theta:   (B, 1,    H, W)

    Output:
        y: (B,)。``direct_sfr`` 时表示变换后的 SFR；``signed_closure`` 时表示
        变换后的闭合项。物理量反变换和 ``B+C_hat`` 在训练/推理契约中完成。
    """

    def __init__(
        self,
        c_in: int, # 输入通道数，即输入的二维物理场的数量
        base_ch: int = 64, # 第一阶段的输出通道数，后续阶段的输出通道数依次翻倍。比如base_ch=64，则四个阶段的输出通道数分别为64, 128, 256, 512
        stage_depths: Tuple[int, ...] = (2, 2, 2, 2), # 每个阶段的块数，默认为4个阶段，每个阶段2个块
        pos_num_bands: int = 32, # 傅里叶变换通道数
        pos_max_freq: float = 64.0, # 傅里叶变换最大频率
        p_drop: float = 0.1, # dropout机制的概率
        se_reduction: int = 4, # SEBlock的通道数缩减率，控制SEBlock中隐藏层的大小，较小的reduction会增加模型容量，但也可能导致过拟合
        use_film: bool = False, # 若为真，使用 cache 声明的活动条件调制特征；当前为 R_m 的 log10(M_filter)。
        prediction_mode: str = "direct_sfr",
    ):
        """
        Args:
            c_in:          Number of input physical-quantity channels.
            base_ch:       Base channel count; doubles each stage.
            stage_depths:  Number of residual blocks per stage.
            pos_num_bands: Fourier encoding frequency bands.
            pos_max_freq:  Max frequency for Fourier encoding.
            p_drop:        Dropout probability.
            se_reduction:  SE block channel reduction ratio.
        """
        super().__init__()
        if prediction_mode not in {"direct_sfr", "signed_closure"}:
            raise ValueError(
                "prediction_mode must be 'direct_sfr' or 'signed_closure', "
                f"got {prediction_mode!r}."
            )
        # 网络头始终输出无界标量；该标量的语义由训练契约明确指定。
        self.prediction_mode = prediction_mode

        # ---- Positional encoding (reuse existing module) ----
        self.posenc = FourierPositionalEncoding2D( # 傅里叶变换，提取位置信息 (15.32.32) -> (145,32,32)
            num_bands=pos_num_bands,
            max_frequency=pos_max_freq,
            include_input=True,
            theta_pi_periodic=False,
        )
        d_pos = self.posenc.out_dim

        # ---- Stem: fuse content + positional features ---- # stem层，压缩通道数。 (145,32,32) -> (64,32,32)
        stem_in = c_in + d_pos # 
        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )

        # ---- Residual stages ---- # CNN主体部分
        stage_channels = []
        stages = []
        in_ch = base_ch
        for i, depth in enumerate(stage_depths):
            out_ch = base_ch * (2 ** i)
            stage_channels.append(out_ch)
            blocks = []
            for j in range(depth):
                stride = 2 if (j == 0 and i > 0) else 1
                blocks.append(
                    ConvFiLMBlock(
                        in_ch if j == 0 else out_ch,
                        out_ch,
                        stride=stride,
                        p_drop=p_drop,
                        se_reduction=se_reduction,
                    )
                )
            stages.append(nn.ModuleList(blocks))
            in_ch = out_ch

        self.stages = nn.ModuleList(stages)
        self.stage_channels = stage_channels
        last_ch = stage_channels[-1]
        self.use_film = use_film

        # ---- FiLM conditioner ----
        if self.use_film:
            self.film = ConvFiLMConditioner(stage_channels=stage_channels, hidden=128)

        # ---- Regression head ---- # 池化，压平，归一化，线性层，激活，dropout，线性层，输出
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(last_ch),
            nn.Linear(last_ch, last_ch // 2),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(last_ch // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,        # (B, C_in, H, W)
        r: torch.Tensor,        # (B, 1,    H, W)
        theta: torch.Tensor,    # (B, 1,    H, W)
        conditioning_scalar: Optional[torch.Tensor] = None, # (B,)；物理语义由 cache contract 决定。
    ) -> torch.Tensor:
        # Positional features
        pos = self.posenc(r, theta, normalize=True)  # (B, D_pos, H, W)

        # Fuse content + positional
        feat = torch.cat([x, pos], dim=1)  # (B, C_in + D_pos, H, W)
        feat = self.stem(feat)

        # 生成 FiLM 参数
        if self.use_film and conditioning_scalar is not None:
            gammas, betas = self.film(conditioning_scalar)
        else:
            gammas, betas = [None] * len(self.stages), [None] * len(self.stages)

        # Residual stages
        for i, stage in enumerate(self.stages):
            gamma = gammas[i]
            beta = betas[i]
            for block in stage:
                feat = block(feat, gamma=gamma, beta=beta)

        # Regression
        y = self.head(feat).squeeze(-1)  # (B,) 预测值
        return y
