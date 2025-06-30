
def pad_to_stride(x, patch, stride):
    B, F, T, C = x.shape
    need = (math.ceil((F - patch) / stride) * stride + patch) - F
    if need > 0:
        x = nn.functional.pad(x, (0,0,0,0,0,need))   # F-dim pad
    return x, need

def freq_patchify_overlap(x, patch, stride, mode='mean'):
    B, F_pad, T, C = x.shape
    starts = torch.arange(0, F_pad - patch + 1, stride, device=x.device)
    P = starts.numel()
    idx_map = starts.unsqueeze(1) + torch.arange(patch, device=x.device)
    # gather → (B,P,T,patch,C)
    patches = x[:, idx_map, :, :].permute(0,1,3,2,4)
    rep = patches.mean(3) if mode == 'mean' else patches.max(3).values
    return rep, idx_map               # rep:(B,P,T,C)

def unpatchify_overlap(patch_out, idx_map, F_pad):
    B, P, T, C = patch_out.shape
    out = torch.zeros(B, F_pad, T, C, device=patch_out.device, 
                      dtype=patch_out.dtype)
    cnt = torch.zeros(F_pad, device=patch_out.device, 
                      dtype=patch_out.dtype)
    for p in range(P):                      # P≲32 → loop OK
        idx = idx_map[p]                   # (patch,)
        out[:, idx] += patch_out[:, p].unsqueeze(1)
        cnt[idx] += 1
    out /= cnt.view(1, F_pad, 1, 1)
    return out

# ─────────────────────────────────────────
# 2. F-patch attention block 
# ─────────────────────────────────────────
class FreqPatchAttnBlock(nn.Module):
    def __init__(self, d_model, nhead, patch, stride, ffn_exp=4, drop=0.1):
        super().__init__()
        self.patch, self.stride = patch, stride
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*ffn_exp),
            nn.ReLU(),
            nn.Linear(d_model*ffn_exp, d_model),
            nn.Dropout(drop))
        self.ln2, self.dp = nn.LayerNorm(d_model), nn.Dropout(drop)

    def forward(self, x):                   # x:(B,F,T,C)
        B, F0, T, C = x.shape
        x_pad, pad = pad_to_stride(x, self.patch, self.stride)
        Q, idx = freq_patchify_overlap(x_pad, self.patch, self.stride)  # (B,P,T,C)
        B, P = Q.shape[:2]
        # print(P)
        K = x_pad.permute(0,2,1,3).reshape(B*T, F0+pad, C)   # (B*T,F,C)
        Qv = Q.permute(0,2,1,3).reshape(B*T, P, C)           # (B*T,P,C)
        # print(Qv.dtype, K.dtype, K.shape, Qv.shape)
        attn, _ = self.attn(Qv, K, K)
        y = self.ln1(Qv + self.dp(attn))
        # print(y.dtype)
        z = self.ln2(y + self.dp(self.ffn(y)))
        # print(z.dtype)
        z = z.reshape(B, T, P, C).permute(0,2,1,3)           # (B,P,T,C)
        x_out = unpatchify_overlap(z, idx, F0+pad)[:, :F0]    # (B,F,T,C)
        return x_out

# ─────────────────────────────────────────
# 3. 전체 모델
# ─────────────────────────────────────────
class CausalDilatedConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        # Causal padding: 과거만 참고
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             dilation=dilation, padding=0)
        
    def forward(self, x):
        # x: [batch, channels, seq_len]
        x = F.pad(x, (self.padding, 0)) 
        return self.conv(x)

class DilatedTCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        
        # Dilated causal convolutions
        self.conv1 = CausalDilatedConv1D(channels, channels, kernel_size, dilation)
        self.conv2 = CausalDilatedConv1D(channels, channels, kernel_size, dilation)
        
        # Normalization and activation
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(channels, channels, 1) if channels != channels else nn.Identity()
        
    def forward(self, x):
        # x: [batch, seq_len, channels]
        identity = x
        
        x = x.transpose(1, 2)  # [batch, channels, seq_len]
        x = self.conv1(x)
        x = x.transpose(1, 2)  # [batch, seq_len, channels]
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = x.transpose(1, 2)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.norm2(x)
        
        # Residual connection
        return identity + x

class DilatedTCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_blocks=4, kernel_size=3):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        
        # Dilated blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList([
            DilatedTCNBlock(hidden_dim, kernel_size, dilation=2**i)
            for i in range(num_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Conv1d(hidden_dim, input_dim, 1)
        
        print(f"TCN Receptive Field: {self.calculate_receptive_field(kernel_size, num_blocks)}")
        
    def calculate_receptive_field(self, kernel_size, num_blocks):
        receptive_field = 1
        for i in range(num_blocks):
            dilation = 2 ** i
            receptive_field += (kernel_size - 1) * dilation
        return receptive_field
        
    def forward(self, x):
        
        x = x.transpose(1, 2)  # [batch, input_dim, seq_len]
        x = self.input_proj(x)
        x = x.transpose(1, 2)  # [batch, seq_len, hidden_dim]
        
        for block in self.blocks:
            x = block(x)
            
        x = x.transpose(1, 2)
        x = self.output_proj(x)
        return x.transpose(1, 2)
class FrequencyWiseTCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, num_blocks=3, kernel_size=3): 
        super().__init__()
        self.input_dim = input_dim
        
        self.tcn = LightDilatedTCN(
            input_dim=input_dim,  
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            kernel_size=kernel_size
        )
        
    def forward(self, x):
        # 입력: (B, F, T, tcn_d_model) 
        B, F, T, d_model = x.shape
        
        # 주파수별 독립 처리를 위한 reshape
        x = x.reshape(B * F, T, d_model)  # (B*F, T, tcn_d_model)
        
        # 주파수별 독립 TCN 처리
        x = self.tcn(x)  # (B*F, T, tcn_d_model)
        
        # 원래 형태로 복원
        x = x.reshape(B, F, T, d_model)  # (B, F, T, tcn_d_model)
        
        return x  # 출력: (B, F, T, tcn_d_model)
class LightDilatedTCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_blocks=3, kernel_size=3):
        super().__init__()
        
        # Depthwise separable convolutions for efficiency
        self.blocks = nn.ModuleList([
            nn.Sequential(
                # Depthwise
                nn.Conv1d(input_dim if i == 0 else hidden_dim, 
                         input_dim if i == 0 else hidden_dim,
                         kernel_size, dilation=2**i, 
                         padding=(kernel_size-1)*2**i//2, 
                         groups=input_dim if i == 0 else hidden_dim),
                # Pointwise
                nn.Conv1d(input_dim if i == 0 else hidden_dim, hidden_dim, 1),
                nn.ReLU()
            ) for i in range(num_blocks)
        ])
        
        self.output_conv = nn.Conv1d(hidden_dim, input_dim, 1)
        
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = x.transpose(1, 2)  # [batch, input_dim, seq_len]
        
        for block in self.blocks:
            x = block(x)
            
        x = self.output_conv(x)
        return x.transpose(1, 2)
def build_time_pe(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # (max_len, d_model)

def build_freq_pe(max_freq_bins: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_freq_bins, d_model)
    # 주파수는 로그 스케일이 더 적합 (인간 청각과 유사)
    freq_pos = torch.log(torch.arange(1, max_freq_bins + 1, dtype=torch.float64))
    freq_pos = freq_pos / freq_pos.max()  # 0~1로 정규화
    
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float64) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(freq_pos.unsqueeze(1) * div)
    pe[:, 1::2] = torch.cos(freq_pos.unsqueeze(1) * div)
    return pe  # (max_freq_bins, d_model)
# PatchFreq + Dilated TCN

class SimpleTimeGroupAttnBlock(nn.Module):
    def __init__(self, d_model, nhead, freq_patch=8, freq_stride=4, 
                 time_group=4, ffn_exp=4, drop=0.1):
        super().__init__()
        self.freq_patch, self.freq_stride = freq_patch, freq_stride
        self.time_group = time_group
        
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*ffn_exp),
            nn.ReLU(),
            nn.Linear(d_model*ffn_exp, d_model),
            nn.Dropout(drop))
        self.ln2, self.dp = nn.LayerNorm(d_model), nn.Dropout(drop)
        # self.max_patches = (513 - freq_patch) // freq_stride + 1
        # self.max_patches = (513 - freq_patch) // freq_stride + 1
        # self.register_buffer("patch_pe", build_freq_pe(self.max_patches, d_model), persistent=False)
        self.patch_pe = nn.Embedding(200, d_model) 
    


    def forward(self, x):  # x:(B,F,T,C)
        B, F0, T, C = x.shape
        
        if T % self.time_group != 0:
            pad_need = self.time_group - (T % self.time_group)
            x = F.pad(x, (0, 0, 0, pad_need))
            T_padded = T + pad_need
        else:
            T_padded = T
        
        num_groups = T_padded // self.time_group
        x_pad, pad = pad_to_stride(x, self.freq_patch, self.freq_stride)
        Q, idx = freq_patchify_overlap(x_pad, self.freq_patch, self.freq_stride)
        _, P = Q.shape[:2] 
        
        patch_ids = torch.arange(P, device=Q.device)  # [0, 1, 2, ..., P-1]
        pe = self.patch_pe(patch_ids)  # (P, d_model)
        Q = Q + pe.unsqueeze(0).unsqueeze(2)  # (B,P,T,C) + (1,P,1,C)
  
        x_reshaped = x_pad[:, :F0, :T_padded].reshape(B * num_groups, F0, self.time_group, C)
         
        Q_reshaped = Q[:, :, :T_padded].reshape(B * num_groups, P, self.time_group, C)
        
        K = x_reshaped.permute(0, 2, 1, 3).reshape(B * num_groups * self.time_group, F0, C)
      
        Qv = Q_reshaped.permute(0, 2, 1, 3).reshape(B * num_groups * self.time_group, P, C)

        attn, _ = self.attn(Qv, K, K)
        y = self.ln1(Qv + self.dp(attn))
        z = self.ln2(y + self.dp(self.ffn(y)))
        

        z = z.reshape(B * num_groups, self.time_group, P, C).permute(0, 2, 1, 3)
        
        # (B*num_groups, P, time_group, C) → (B, P, T_padded, C)
        z_final = z.reshape(B, P, T_padded, C)
        
        # 복원
        x_out = unpatchify_overlap(z_final, idx, F0+pad)[:, :F0, :T]
        return x_out
class PatchFreqNet(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256, tf_d_model=32, tcn_d_model=8, depth=2,
                 freq_patch=8, time_group=4,      
                 freq_stride=4,    
                 nhead=4, cin=2, cout=2, ffn_exp=4,
                 tcn_hidden=16, tcn_blocks=3, tcn_kernel=3, 
                 use_light_tcn=""):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_size = n_fft // 2 + 1  # 513
        
        self.in_proj = nn.Linear(cin, tf_d_model)
        self.blocks = nn.ModuleList([
            SimpleTimeGroupAttnBlock(tf_d_model, nhead, 
                                   freq_patch, freq_stride,
                                   time_group, ffn_exp) 
            for _ in range(depth)
        ])
        
        self.ln_tf = nn.LayerNorm(tf_d_model)
        self.to_tcn = nn.Linear(tf_d_model, tcn_d_model)
        
        if use_light_tcn == "light":
            self.temporal_tcn = LightDilatedTCN(
                input_dim=tcn_d_model * self.f_size,
                hidden_dim=tcn_hidden,
                num_blocks=tcn_blocks,
                kernel_size=tcn_kernel
            )
            self._temporal_forward = self._light_temporal_forward
        else:
            self.temporal_tcn = FrequencyWiseTCN(
                input_dim=tcn_d_model,
                hidden_dim=tcn_hidden,
                num_blocks=tcn_blocks,
                kernel_size=tcn_kernel
            )
            self._temporal_forward = self._wise_temporal_forward
        
        self.out_proj = nn.Linear(tcn_d_model, cout)
    def _light_temporal_forward(self, x):
        B, F, T, _ = x.shape
        x_flat = x.permute(0,2,1,3).reshape(B, T, -1)  # (B,T,tcn_d_model*F)
        x_temporal = self.temporal_tcn(x_flat)  # (B,T,tcn_d_model*F)
        x_temporal = x_temporal.reshape(B, T, F, -1)  # (B,T,F,tcn_d_model)
        return x_temporal.permute(0,2,1,3)  # (B,F,T,tcn_d_model)
    
    def _wise_temporal_forward(self, x):
        return self.temporal_tcn(x)  # (B,F,T,tcn_d_model)

    def forward(self, x, is_istft=False):  # x:(B,1,F,T,2)
        B, _, F, T, _ = x.shape
        x = x.squeeze(1)  # (B,F,T,2)
        
        x = self.in_proj(x)  # (B,F,T,tf_d_model)

        for blk in self.blocks:
            x = blk(x)
        x = self.ln_tf(x)  # (B,F,T,tf_d_model)
        
        x = self.to_tcn(x)  # (B,F,T,tcn_d_model)

        x_temporal = self._temporal_forward(x)  # (B,F,T,tcn_d_model)
       
        out = self.out_proj(x_temporal)  # (B,T,F,cout)
        if is_istft:
            est_c = torch.complex(out[:,0], out[:,1])  # (B,F,T)
            full  = reconstruct_full_spectrum(est_c)
            wav = torch.istft(full, n_fft=self.n_fft,
                              hop_length=self.hop_length,
                              window=torch.hann_window(self.n_fft,
                                     device=out.device),
                              normalized=True)
            return wav, _
        return out.unsqueeze(1),_  # (B,1,F,T,cout)
