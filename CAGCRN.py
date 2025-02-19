"""
CAGCRN: Real-Time Speech Enhancement with a Lightweight Model for Joint Acoustic Echo Cancellation and Noise Suppression.
"""
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class ERB(nn.Module):
    """Equivalent Rectangular Bandwidth (ERB) filterbank module.
    
    This module implements ERB filterbank analysis and synthesis. It converts linear frequency
    scale features to ERB scale and back.
    
    Args:
        low_freq_idx (int): Starting frequency bin index for ERB processing
        num_erb_bands (int): Number of ERB bands to generate
        nfft (int, optional): FFT size. Defaults to 512.
        high_freq (int, optional): Highest frequency in Hz. Defaults to 8000.
        sample_rate (int, optional): Sampling rate in Hz. Defaults to 16000.
    """
    def __init__(self, low_freq_idx, num_erb_bands, nfft=512, high_freq=8000, sample_rate=16000):
        super().__init__()
        
        # Store parameters
        self.low_freq_idx = low_freq_idx
        self.num_erb_bands = num_erb_bands
        
        # Generate ERB filterbank matrices
        erb_filters = self._create_erb_filterbank(
            low_freq_idx, num_erb_bands, nfft, high_freq, sample_rate)
        
        # Create linear transformations for analysis and synthesis
        num_freqs = nfft//2 + 1
        self.analysis = nn.Linear(num_freqs-low_freq_idx, num_erb_bands, bias=False)
        self.synthesis = nn.Linear(num_erb_bands, num_freqs-low_freq_idx, bias=False)
        
        # Set filter weights as non-trainable parameters
        self.analysis.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.synthesis.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def _hz_to_erb(self, freq_hz):
        """Convert frequency in Hz to ERB scale."""
        return 21.4 * torch.log10(0.00437 * freq_hz + 1)

    def _erb_to_hz(self, erb_scale):
        """Convert ERB scale back to frequency in Hz."""
        return (10**(erb_scale/21.4) - 1) / 0.00437

    def _create_erb_filterbank(self, low_freq_idx, num_erb_bands, nfft, high_freq, sample_rate):
        """Create ERB filterbank matrices for analysis and synthesis.
        
        Returns:
            torch.Tensor: ERB filterbank matrix of shape (num_erb_bands, num_freqs-low_freq_idx)
        """
        # Calculate frequency range in Hz
        low_freq = low_freq_idx / nfft * sample_rate
        
        # Convert to ERB scale
        erb_low = self._hz_to_erb(torch.tensor(low_freq))
        erb_high = self._hz_to_erb(torch.tensor(high_freq))
        
        # Generate ERB points
        erb_points = torch.linspace(erb_low, erb_high, num_erb_bands)
        
        # Convert back to frequency bins
        freq_bins = torch.round(self._erb_to_hz(erb_points)/sample_rate * nfft).int()
        
        # Initialize filterbank matrix
        erb_filters = torch.zeros((num_erb_bands, nfft//2 + 1), dtype=torch.float32)
        
        # Generate triangular filters
        for i in range(num_erb_bands):
            if i == 0:
                # First filter
                start, end = freq_bins[i], freq_bins[i+1]
                erb_filters[i, start:end] = (end - torch.arange(start, end)) / (end - start + 1e-8)
            elif i == num_erb_bands - 1:
                # Last filter
                start, end = freq_bins[i-1], freq_bins[i]
                erb_filters[i, start:end+1] = 1 - erb_filters[i-1, start:end+1]
            else:
                # Middle filters
                start, mid, end = freq_bins[i-1], freq_bins[i], freq_bins[i+1]
                erb_filters[i, start:mid] = (torch.arange(start, mid) - start) / (mid - start + 1e-8)
                erb_filters[i, mid:end] = (end - torch.arange(mid, end)) / (end - mid + 1e-8)
        
        # Trim and return
        return erb_filters[:, low_freq_idx:]

    def bm(self, x):
        """ERB analysis (Bark mapping).
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, F)
            
        Returns:
            torch.Tensor: ERB-scale features of shape (B, C, T, F_erb)
        """
        x_low = x[..., :self.low_freq_idx]
        x_high = self.analysis(x[..., self.low_freq_idx:])
        return torch.cat([x_low, x_high], dim=-1)
    
    def bs(self, x_erb):
        """ERB synthesis (Bark synthesis).
        
        Args:
            x_erb (torch.Tensor): ERB-scale features of shape (B, C, T, F_erb)
            
        Returns:
            torch.Tensor: Linear-scale features of shape (B, C, T, F)
        """
        x_low = x_erb[..., :self.low_freq_idx]
        x_high = self.synthesis(x_erb[..., self.low_freq_idx:])
        return torch.cat([x_low, x_high], dim=-1)

class ConvBlock(nn.Module):
    """Basic convolutional block with batch normalization and activation.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels 
        kernel_size (tuple): Size of the convolving kernel
        stride (tuple): Stride of the convolution
        padding (tuple): Padding added to both sides of input
        groups (int, optional): Number of blocked connections. Defaults to 1
        use_deconv (bool, optional): Whether to use transposed convolution. Defaults to False
        is_last (bool, optional): Whether this is the last block. Defaults to False
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_deconv=False, is_last=False):
        super().__init__()
        # Choose between regular conv and transposed conv
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        # Use Tanh for last block, PReLU otherwise
        self.act = nn.Tanh() if is_last else nn.PReLU()
        
    def forward(self, x):
        """Forward pass through conv block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output after convolution, batch norm and activation
        """
        return self.act(self.bn(self.conv(x)))


class DGConvBlock(nn.Module):
    """ dilated group convolution block .
    
    Uses a depthwise convolution followed by pointwise convolutions for efficient processing.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (tuple): Size of the convolving kernel
        stride (tuple): Stride of the convolution
        padding (tuple): Padding added to both sides of input
        dilation (tuple): Dilation factor for convolution
        use_deconv (bool, optional): Whether to use transposed convolution. Defaults to False
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, use_deconv=False):
        super().__init__()
        self.use_deconv = use_deconv
        # Calculate padding size for dilated convolution
        self.pad_size = (kernel_size[0]-1) * dilation[0]
        
        # Choose between regular conv and transposed conv
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        
        # First pointwise convolution (1x1)
        self.point_conv1 = conv_module(in_channels, out_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(out_channels)
        self.point_act = nn.PReLU()

        # Depthwise convolution
        self.depth_conv = conv_module(out_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding,
                                    dilation=dilation, groups=out_channels)
        self.depth_bn = nn.BatchNorm2d(out_channels)
        self.depth_act = nn.PReLU()

        # Second pointwise convolution (1x1)
        self.point_conv2 = conv_module(out_channels, out_channels, 1)
        self.point_bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """Forward pass through the depthwise-separable conv block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, F)
            
        Returns:
            torch.Tensor: Output tensor after depthwise-separable convolution
        """
        x1 = x  # Save residual
        # First pointwise conv
        x = self.point_act(self.point_bn1(self.point_conv1(x)))
        # Pad for dilated conv
        x = nn.functional.pad(x, [0, 0, self.pad_size, 0])
        # Depthwise conv
        x = self.depth_act(self.depth_bn(self.depth_conv(x)))
        # Second pointwise conv
        x = self.point_bn2(self.point_conv2(x))
        # Add residual connection
        x = x + x1
        return x

class CATA(nn.Module):
    """Cross-Attention Temporal Alignment module.
    
    This module performs self-attention on microphone signal and cross-attention with 
    reference signal for temporal alignment.
    
    Args:
        in_channels (int): Number of input channels
        hidden_channels (int): Number of hidden channels for attention computation
    """
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        # Microphone signal self-attention Q,K,V transformations
        self.pconv_mic = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, groups=in_channels),
            nn.Conv2d(in_channels, hidden_channels, 1)
        )
        self.pconv_mic2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1, groups=hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, 1)
        )
        
        # Reference signal processing
        self.pconv_ref = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, groups=in_channels),
            nn.Conv2d(in_channels, hidden_channels, 1)
        )

        # Learnable delay factor for temporal alignment
        self.delay_factor = nn.Parameter(torch.randn(1)) 
      
        # Final fusion layer to combine mic and ref features
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_channels*2, in_channels*2, 1),
            nn.BatchNorm2d(in_channels*2),
            nn.PReLU()
        )
        
    def sliding_window(self, x, window_size, step=1):
        """Apply sliding window to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, F)
            window_size (int): Size of the sliding window
            step (int, optional): Step size for window sliding. Defaults to 1
            
        Returns:
            torch.Tensor: Windowed tensor of shape (B, C, T', window_size, F)
        """
        B, C, T, F = x.shape
        return x.unfold(2, window_size, step).permute(0, 1, 2, 4, 3)
        
    def forward(self, x_mic, x_ref):
        """Forward pass through CATA module.
        
        Args:
            x_mic (torch.Tensor): Microphone signal features of shape (B, C, T, F)
            x_ref (torch.Tensor): Reference signal features of shape (B, C, T, F)
            
        Returns:
            tuple: (
                combined_features: Combined and aligned features (B, C*2, T, F),
                mic_attention_output: Processed microphone features (B, C, T, F)
            )
        """
        # 1. Microphone self-attention processing
        Q_mic = self.pconv_mic(x_mic)  # (B,H,T,F)
        K_mic = self.pconv_mic2(Q_mic)  # (B,H,T,F)
        V_mic = x_mic  # (B,C,T,F)
        
        # Compute attention scores for mic signal
        attn_scores_mic = torch.matmul(Q_mic.permute(0,2,3,1), K_mic.permute(0,2,1,3))  # (B,T,F,F)
        attn_scores_mic = F.softmax(attn_scores_mic / torch.sqrt(torch.tensor(Q_mic.shape[1], dtype=torch.float32)), dim=-1)
        
        # Apply attention to mic features
        mic_out = torch.matmul(attn_scores_mic, V_mic.permute(0,2,3,1)).permute(0,3,1,2)  # (B,C,T,F)
        
        # 2. Reference signal temporal alignment
        continuous_delay = torch.tensor([11.0], device=x_mic.device) - torch.tensor([10.0], device=x_mic.device) * torch.tanh(self.delay_factor)
        effective_delay = torch.floor(continuous_delay).int()
        
        if self.training:
            effective_delay = continuous_delay.detach().int()
            
        # Process reference signal
        K_ref = self.pconv_ref(x_ref)
        K_ref = F.pad(K_ref, (0, 0, effective_delay.item() - 1, 0))       
        Ku = self.sliding_window(K_ref, effective_delay.item())
        
        # Compute alignment scores
        ref_score = Q_mic.unsqueeze(-2) * Ku  # (B,C,T,D,F)
        ref_score = torch.softmax(ref_score, dim=-1) # (B,C,T,D,F)

        # Apply alignment to reference features
        V_ref = F.pad(x_ref, (0, 0, effective_delay.item() - 1, 0)) 
        V_ref = self.sliding_window(V_ref, effective_delay.item()) # (B,C,T,D,F)
        ref_out = torch.sum(V_ref * ref_score, dim=-2) # (B,C,T,F)
        
        # 3. Combine mic and ref features
        combined = torch.cat([mic_out, ref_out], dim=1) # (B,C*2,T,F)
        out = self.fusion(combined)
        
        return out, mic_out


class Encoder(nn.Module):
    """Encoder module for processing microphone and reference signals.
    
    This module processes the near-end microphone signal and far-end reference signal
    through parallel convolutional paths, then aligns and combines them using
    cross-attention.
    """
    def __init__(self):
        super().__init__()
        # Initial convolutions for microphone signal
        self.mic_convs_pre = nn.ModuleList([
            ConvBlock(9, 12, (1,5), stride=(1,2), padding=(0,2), groups=1, use_deconv=False, is_last=False),
            DGConvBlock(12, 12, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False), 
        ])
        
        # Initial convolutions for reference signal
        self.ref_convs_pre = nn.ModuleList([
            ConvBlock(9, 12, (1,5), stride=(1,2), padding=(0,2), groups=1, use_deconv=False, is_last=False),
            DGConvBlock(12, 12, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False), 
        ])
        
        # Processing path for combined features
        self.mix_convs = nn.ModuleList([
            ConvBlock(24, 24, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False),
            DGConvBlock(24, 24, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False), 
            ConvBlock(24, 24, (1,5), stride=(1,1), padding=(0,2), groups=2, use_deconv=False, is_last=False),          
            DGConvBlock(24, 24, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False),
            ConvBlock(24, 24, (1,5), stride=(1,1), padding=(0,2), groups=2, use_deconv=False, is_last=False), 
            DGConvBlock(24, 24, (3,3), stride=(1,1), padding=(0,1), dilation=(4,1), use_deconv=False),
        ])

        # Additional processing path for microphone features
        self.mic_convs = nn.ModuleList([
            ConvBlock(12, 12, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False),
            DGConvBlock(12, 12, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False), 
            ConvBlock(12, 12, (1,5), stride=(1,1), padding=(0,2), groups=2, use_deconv=False, is_last=False),          
            DGConvBlock(12, 12, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False),
            ConvBlock(12, 24, (1,5), stride=(1,1), padding=(0,2), groups=2, use_deconv=False, is_last=False), 
            DGConvBlock(24, 24, (3,3), stride=(1,1), padding=(0,1), dilation=(4,1), use_deconv=False),
        ])
        
        # Cross-attention and temporal alignment module
        self.align = CATA(12, 12)
        
        # Subband feature extraction using unfold operations
        self.unfold_mic = nn.Unfold(
            kernel_size=(1,3), 
            stride=(1,1), 
            padding=(0,1)
        )
        self.unfold_ref = nn.Unfold(
            kernel_size=(1,3), 
            stride=(1,1), 
            padding=(0,1)
        )

    def _extract_subband_features(self, x, unfold):
        """Extracts subband features using unfold operation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B,C,T,F)
            unfold (nn.Unfold): Unfold layer for feature extraction
            
        Returns:
            torch.Tensor: Subband features of shape (B,C*3,T,F)
        """
        B, C, T, F = x.shape
        x_unfolded = unfold(x)
        return x_unfolded.reshape(B, C*3, T, F)

    def forward(self, mic, ref):
        """Forward pass through the encoder.
        
        Args:
            mic (torch.Tensor): Near-end microphone signal of shape (B,C,T,F)
            ref (torch.Tensor): Far-end reference signal of shape (B,C,T,F)
            
        Returns:
            tuple: (
                mixed_features: Combined features after processing,
                encoder_outputs: List of intermediate features for skip connections,
                mic_features: Processed microphone features
            )
        """
        en_outs = []
        
        # Extract subband features
        mic = self._extract_subband_features(mic, self.unfold_mic)
        ref = self._extract_subband_features(ref, self.unfold_ref)
        
        # Initial parallel processing of mic and ref signals
        for i in range(len(self.mic_convs_pre)):
            mic = self.mic_convs_pre[i](mic)
            ref = self.ref_convs_pre[i](ref)
            en_outs.append(mic)
        
        # Align and combine features using cross-attention
        mix, _ = self.align(mic, ref)
        
        # Process combined features
        for conv in self.mix_convs:
            mix = conv(mix)
            en_outs.append(mix)

        # Additional microphone feature processing
        for conv in self.mic_convs:
            mic = conv(mic)
            
        return mix, en_outs, mic
    
class TFGRU(nn.Module):
    """Time-Frequency Grouped Recurrent Neural Network.
    
    This module combines intra- and inter-chunk processing using grouped GRU cells.
    Each GRU is split into two parallel groups to reduce complexity while maintaining performance.
    
    Args:
        input_size (int): Input feature dimension
        width (int): Frequency dimension size
        hidden_size (int): Hidden state dimension
    """
    def __init__(self, input_size, width, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size
        
        # Intra-chunk (time-direction) processing
        self.intra_gru = self._create_grouped_gru(
            input_size=input_size,
            hidden_size=hidden_size//2,
            bidirectional=True
        )
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_norm = nn.LayerNorm((width, hidden_size), eps=1e-8)
        
        # Inter-chunk (frequency-direction) processing
        self.inter_gru = self._create_grouped_gru(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=False
        )
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_norm = nn.LayerNorm((width, hidden_size), eps=1e-8)

    def _create_grouped_gru(self, input_size, hidden_size, bidirectional):
        """Creates a grouped GRU with two parallel paths.
        
        Args:
            input_size (int): Input dimension
            hidden_size (int): Hidden state dimension
            bidirectional (bool): Whether to use bidirectional GRU
            
        Returns:
            nn.ModuleDict: Dictionary containing two parallel GRU paths
        """
        return nn.ModuleDict({
            'path1': nn.GRU(
                input_size=input_size//2,
                hidden_size=hidden_size//2,
                batch_first=True,
                bidirectional=bidirectional
            ),
            'path2': nn.GRU(
                input_size=input_size//2,
                hidden_size=hidden_size//2,
                batch_first=True,
                bidirectional=bidirectional
            )
        })

    def _grouped_gru_forward(self, x, gru_dict, h=None):
        """Forward pass through grouped GRU.
        
        Args:
            x (torch.Tensor): Input of shape (B, seq_len, input_size)
            gru_dict (nn.ModuleDict): Dictionary of parallel GRUs
            h (torch.Tensor, optional): Initial hidden state
            
        Returns:
            tuple: (output, hidden_state)
        """
        # Split input along feature dimension
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        
        # Split hidden state if provided
        if h is not None:
            h1, h2 = torch.chunk(h, chunks=2, dim=-1)
            h1, h2 = h1.contiguous(), h2.contiguous()
        else:
            h1 = h2 = None
            
        # Forward through parallel GRUs
        y1, h1 = gru_dict['path1'](x1, h1)
        y2, h2 = gru_dict['path2'](x2, h2)
        
        # Combine outputs
        y = torch.cat([y1, y2], dim=-1)
        h = torch.cat([h1, h2], dim=-1) if h1 is not None else None
        
        return y, h

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, F)
            
        Returns:
            torch.Tensor: Processed tensor of shape (B, C, T, F)
        """
        # Reshape for RNN processing
        x = x.permute(0, 2, 3, 1)  # (B, T, F, C)
        batch_size = x.shape[0]
        
        # Intra-chunk processing (along time)
        intra_x = x.reshape(-1, x.shape[2], x.shape[3])  # (B*T, F, C)
        intra_x, _ = self._grouped_gru_forward(intra_x, self.intra_gru)
        intra_x = self.intra_fc(intra_x)
        intra_x = intra_x.reshape(batch_size, -1, self.width, self.hidden_size)
        intra_x = self.intra_norm(intra_x)
        intra_out = x + intra_x
        
        # Inter-chunk processing (along frequency)
        x = intra_out.permute(0, 2, 1, 3)  # (B, F, T, C)
        inter_x = x.reshape(-1, x.shape[2], x.shape[3])
        inter_x, _ = self._grouped_gru_forward(inter_x, self.inter_gru)
        inter_x = self.inter_fc(inter_x)
        inter_x = inter_x.reshape(batch_size, self.width, -1, self.hidden_size)
        inter_x = inter_x.permute(0, 2, 1, 3)
        inter_x = self.inter_norm(inter_x)
        inter_out = intra_out + inter_x
        
        # Return to original format
        return inter_out.permute(0, 3, 1, 2)  # (B, C, T, F)
        
class TFAG(nn.Module):
    """Time-Frequency Adaptive Gating module.
    
    This module performs multi-scale feature extraction in both time and frequency domains,
    followed by adaptive gating to selectively combine features.
    
    Args:
        channels (int): Number of input channels
    """
    def __init__(self, channels):
        super().__init__()
        
        reduced_channels = channels // 2  # Reduce channels for efficiency
        
        # Multi-scale temporal feature extraction
        self.temporal_scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, reduced_channels, (5,1), padding=(2,0), dilation=(1,1), groups=reduced_channels),
                nn.BatchNorm2d(reduced_channels),
                nn.PReLU()
            ),
            nn.Sequential(
                nn.Conv2d(channels, reduced_channels, (5,1), padding=(4,0), dilation=(2,1), groups=reduced_channels), 
                nn.BatchNorm2d(reduced_channels),
                nn.PReLU()
            )
        ])
        
        # Multi-scale frequency feature extraction
        self.freq_scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, reduced_channels, (1,3), padding=(0,1), dilation=(1,1), groups=reduced_channels),
                nn.BatchNorm2d(reduced_channels),
                nn.PReLU()
            ),
            nn.Sequential(
                nn.Conv2d(channels, reduced_channels, (1,3), padding=(0,2), dilation=(1,2), groups=reduced_channels),
                nn.BatchNorm2d(reduced_channels),
                nn.PReLU()
            )
        ])
        
        # Adaptive gate generation network
        self.gate_generator = nn.Sequential(
            nn.Conv2d(reduced_channels*4, reduced_channels, 1, groups=reduced_channels),
            nn.BatchNorm2d(reduced_channels),
            nn.PReLU(),
            nn.Conv2d(reduced_channels, channels, 1),
            nn.Sigmoid()
        )
        
        # Efficient feature fusion
        self.fusion = nn.Conv2d(channels, channels, 1, groups=channels)

    def forward(self, x1, x2):
        """Forward pass through TFAG module.
        
        Args:
            x1 (torch.Tensor): First input features of shape (B, C, T, F)
            x2 (torch.Tensor): Second input features of shape (B, C, T, F)
            
        Returns:
            torch.Tensor: Gated and fused features of shape (B, C, T, F)
        """
        mix = x1 + x2
        
        # Multi-scale feature extraction
        temp_feats = [conv(mix) for conv in self.temporal_scales]
        freq_feats = [conv(mix) for conv in self.freq_scales]
        
        # Feature aggregation
        multi_scale_feats = torch.cat(temp_feats + freq_feats, dim=1)
        
        # Generate adaptive gating weights
        gate = self.gate_generator(multi_scale_feats)
        
        # Feature selection and fusion
        out = gate * x1 + (1 - gate) * x2
        out = self.fusion(out)
        
        return out

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList([

            DGConvBlock(24, 24, (3,3), stride=(1,1), padding=(2*4,1), dilation=(4,1), use_deconv=True),
            ConvBlock(24, 24, (1,5), stride=(1,1), padding=(0,2), groups=2, use_deconv=True, is_last=False),
            DGConvBlock(24, 24, (3,3), stride=(1,1), padding=(2*2,1), dilation=(2,1), use_deconv=True),
            ConvBlock(24, 24, (1,5), stride=(1,1), padding=(0,2), groups=2, use_deconv=True, is_last=False),
            DGConvBlock(24, 24, (3,3), stride=(1,1), padding=(2*1,1), dilation=(1,1), use_deconv=True),
            ConvBlock(24, 12, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True, is_last=False),
            DGConvBlock(12, 12, (3,3), stride=(1,1), padding=(2*1,1), dilation=(1,1), use_deconv=True),
            ConvBlock(12, 2, (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=True)
        ])

    def forward(self, x, en_outs):
        N_layers = len(self.de_convs)
        for i in range(N_layers):
            x = self.de_convs[i](x + en_outs[N_layers-1-i])
            # print (x.shape)
        return x
    
class Mask(nn.Module):
    """Complex Ratio Mask"""
    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        s_real = spec[:,0] * mask[:,0] - spec[:,1] * mask[:,1]
        s_imag = spec[:,1] * mask[:,0] + spec[:,0] * mask[:,1]
        s = torch.stack([s_real, s_imag], dim=1)  # (B,2,T,F)
        return s


class CAGCRN(nn.Module):
    """Cross-Attention Gated Convolutional Recurrent Network for speech enhancement.
    
    This model combines ERB filterbank processing, cross-attention temporal alignment,
    time-frequency GRU processing, and adaptive gating for joint acoustic echo 
    cancellation and noise suppression.
    """
    def __init__(self):
        super().__init__()
        # ERB filterbank for frequency-domain processing
        self.erb = ERB(65, 64)
        
        # Encoder for feature extraction and alignment
        self.encoder = Encoder()
        
        # Time-Frequency GRU modules for sequential processing
        self.tfgru1 = TFGRU(24, 33, 24)
        self.tfgru2 = TFGRU(24, 33, 24)
        self.tfgru3 = TFGRU(24, 33, 24)
        
        # Decoder for feature reconstruction
        self.decoder = Decoder()
        
        # Complex ratio mask generation
        self.mask = Mask() 
        
        # Time-Frequency Adaptive Gating module
        self.multi_scale_gate = TFAG(24)
        
    def forward(self, mic_spec, ref_spec):
        """Forward pass through CAGCRN.
        
        Args:
            mic_spec (torch.Tensor): Near-end microphone spectrogram of shape (B, F, T, 2)
            ref_spec (torch.Tensor): Far-end reference spectrogram of shape (B, F, T, 2)
            
        Returns:
            torch.Tensor: Enhanced spectrogram of shape (B, F, T, 2)
        """
        # Save original mic spectrogram for mask application
        spec_ori = mic_spec

        # Process microphone signal
        mic_real = mic_spec[..., 0].permute(0,2,1)  
        mic_imag = mic_spec[..., 1].permute(0,2,1)
        mic_mag = torch.sqrt(mic_real**2 + mic_imag**2 + 1e-12)
        mic_feat = torch.stack([mic_mag, mic_real, mic_imag], dim=1) 

        # Process reference signal
        ref_real = ref_spec[..., 0].permute(0,2,1)
        ref_imag = ref_spec[..., 1].permute(0,2,1) 
        ref_mag = torch.sqrt(ref_real**2 + ref_imag**2 + 1e-12)
        ref_feat = torch.stack([ref_mag, ref_real, ref_imag], dim=1) 

        # ERB filterbank analysis
        mic_feat = self.erb.bm(mic_feat)     
        ref_feat = self.erb.bm(ref_feat)  
        
        # Encoder processing with feature alignment
        feat_mix, en_outs, feat_mic = self.encoder(mic_feat, ref_feat)
        
        # Time-Frequency GRU processing
        feat_mix = self.tfgru2(self.tfgru1(feat_mix))
        feat_mic = self.tfgru3(feat_mic)
        
        # Adaptive gating between mixed and mic features
        feat_mix = self.multi_scale_gate(feat_mix, feat_mic)
        
        # Decoder processing with skip connections
        m_feat = self.decoder(feat_mix, en_outs)

        # ERB filterbank synthesis
        m = self.erb.bs(m_feat)

        # Apply complex ratio mask
        spec_enh = self.mask(m, spec_ori.permute(0,3,2,1)) 
        spec_enh = spec_enh.permute(0,3,2,1)  

        return spec_enh

if __name__ == "__main__":
    # Create model instance in evaluation mode
    model = CAGCRN().eval()
    
    from thop import profile
    
    # Create random input tensors for testing
    x1 = torch.randn(1, 257, 63, 2)  # Microphone signal
    x2 = torch.randn(1, 257, 63, 2)  # Reference signal
    
    # Calculate FLOPs (Floating Point Operations)
    flops, params = profile(model, inputs=(x1, x2))
    
    # Calculate MACs (Multiply-Accumulate Operations)
    macs = flops / 2  # Convert FLOPs to MACs
    
    # Print raw numbers
    print(f"FLOPs: {flops:,}")
    print(f"MACs: {macs:,}")
    print(f"Parameters: {params:,}")
    
    # Print in human-readable format (G = 10^9, M = 10^6, K = 10^3)
    print(f"FLOPs: {flops/1e9:.2f}G")
    print(f"MACs: {macs/1e6:.2f}M")
    print(f"Parameters: {params/1e3:.2f}K")
