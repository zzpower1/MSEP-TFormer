import torch
from torch import nn
from torch.nn import functional as F
import math
from models.attention import *


class ResConvBlock(nn.Module):
    """Residual convolution block
    Input shape: (N,C,L)
    """

    def __init__(self, io_channels, kernel_size, drop_rate):
        super().__init__()

        self.conv_padding_same = (
            (kernel_size - 1) // 2,
            kernel_size - 1 - (kernel_size - 1) // 2,
        )

        self.bn0 = nn.BatchNorm1d(num_features=io_channels)
        self.relu0 = nn.GELU()
        self.dropout0 = nn.Dropout1d(p=drop_rate)
        self.conv0 = nn.Conv1d(
            in_channels=io_channels, out_channels=32, kernel_size=kernel_size
        )

        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.GELU()
        self.dropout1 = nn.Dropout1d(p=drop_rate)
        self.conv1 = nn.Conv1d(
            in_channels=32, out_channels=io_channels, kernel_size=kernel_size
        )

    def forward(self, x):
        x1 = self.bn0(x)
        x1 = self.relu0(x1)
        x1 = self.dropout0(x1)
        x1 = F.pad(x1, self.conv_padding_same, "constant", 0)
        x1 = self.conv0(x1)

        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.dropout1(x1)
        x1 = F.pad(x1, self.conv_padding_same, "constant", 0)
        x1 = self.conv1(x1)
        # out = x + x1
        return x1
    

class AttentionLayer(nn.Module): # Substitution of standard attention layer
    def __init__(self, in_channels, hidden_size, seq_len, drop_rate, attn_width=None, att_mode='CA', trans_mode='None'): # hidden_dim must be (input dimension/2 +1)
        super().__init__()
        if att_mode=='PA':
            self.att = PixelAttention(seq_len=seq_len, trans_mode=trans_mode)
        elif att_mode=='CA':
            self.att = ChannelAttention(in_channels=in_channels, trans_mode=trans_mode)
        elif att_mode=='SA':
            self.att = SpatialAttention(trans_mode=trans_mode)
        elif att_mode=='SEA':
            self.att = SelfAttention(in_channels=in_channels, hidden_size=hidden_size, attn_width=attn_width, trans_mode=trans_mode)
        elif att_mode=='GF':
            self.att = GlobalFilter(in_channels=in_channels, seq_len=seq_len, attn_width=attn_width)
        elif att_mode=='DF':
            self.att = DWTFilter(in_channels=in_channels, seq_len=seq_len)

    def forward(self, x):
        x = self.att(x)

        return x
    

class FeedForward(nn.Module):
    def __init__(self, io_channels, feedforward_dim, drop_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(io_channels, feedforward_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(feedforward_dim, io_channels),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):

        return self.net(x)


class TransformerLayer(nn.Module):
    def __init__(self, io_channels, seq_len, feedforward_dim, drop_rate, att_mode, attn_width=None, trans_mode='None'):
        super().__init__()

        self.attn = AttentionLayer(in_channels=io_channels, hidden_size=feedforward_dim, seq_len=seq_len, drop_rate=drop_rate, attn_width=attn_width, att_mode=att_mode, trans_mode=trans_mode)
        self.ln0 = nn.LayerNorm(normalized_shape=io_channels)

        self.ff = FeedForward(
            io_channels=io_channels,
            feedforward_dim=feedforward_dim,
            drop_rate=drop_rate,
        )
        self.ln1 = nn.LayerNorm(normalized_shape=io_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x1 = self.ln0(x)
        x1 = self.attn(x1)
        x1 += x

        x2 = self.ln1(x1)
        x2 = self.ff(x2)
        x2 += x1

        return x2.permute(0, 2, 1)


class TransformerLayer2(nn.Module):
    def __init__(self, io_channels, seq_len, feedforward_dim, drop_rate, att_mode='CA', attn_width=None):
        super().__init__()

        self.attn = AttentionLayer(in_channels=io_channels, hidden_size=feedforward_dim, seq_len=seq_len, drop_rate=drop_rate, attn_width=attn_width, att_mode=att_mode)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x1 = self.attn(x)
        x1 = x + x1

        return x1.permute(0, 2, 1)


class PatchEmbed(nn.Module):
    def __init__(self, projection_dim, seq_len):
        super(PatchEmbed, self).__init__()
        self.positional_embedding = nn.Parameter(torch.zeros(1, projection_dim, seq_len))
        nn.init.trunc_normal_(self.positional_embedding, std=.02)

    def forward(self, x):
        embedded = x + self.positional_embedding  

        return embedded
    

class Transformer(nn.Module):
    def __init__(self, io_channels, seq_len, feedforward_dim, drop_rate, num_transformer_layers, att_mode, attn_width=None, trans_mode='None'):
        super().__init__()
        self.patch_embed = PatchEmbed(projection_dim=io_channels, seq_len=seq_len)

        self.transformers = nn.ModuleList(
            [
                TransformerLayer(
                    io_channels=io_channels,
                    seq_len=seq_len,
                    feedforward_dim=feedforward_dim,
                    drop_rate=drop_rate,
                    attn_width=attn_width,
                    att_mode=att_mode,
                    trans_mode=trans_mode
                )
                for _ in range(num_transformer_layers)
            ]
        )

        self.ln = nn.LayerNorm(normalized_shape=io_channels)

    def forward(self, x):
        x = self.patch_embed(x)
        for transformer in self.transformers:
            x = transformer(x)

        x = x.permute(0, 2, 1)
        x = self.ln(x)
        x = x.permute(0, 2, 1)

        return x
    

class BiLSTMBlock(nn.Module):
    """Bi-LSTM block
    Input shape: (N,C,L)
    """

    def __init__(self, in_channels, out_channels, drop_rate):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=out_channels,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(p=drop_rate)
        self.conv = nn.Conv1d(
            in_channels=2*out_channels, out_channels=out_channels, kernel_size=1
        )
        self.bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        return x


class Encoder(nn.Module):
    def __init__(self, 
            seq_len=24,
            conv_dim=[3, 16, 32, 64],
            transformer_io_channels=96,
            num_lstm_blocks=3,
            attn_width=None,
            feedforward_dim=64,
            num_transformer_layers=2,
            drop_rate=0.1,
            att_mode='CA', 
            trans_mode='None'
        ):
        super(Encoder, self).__init__()

        conv_layers = [
            [nn.Conv1d(conv_dim[0], conv_dim[1], 7, 2, padding=3), nn.Dropout(drop_rate), nn.MaxPool1d(4)],  
            [nn.Conv1d(conv_dim[1], conv_dim[2], 5, 2, padding=2), nn.Dropout(drop_rate), nn.MaxPool1d(2)],
            [nn.Conv1d(conv_dim[2], conv_dim[3], 3, 2, padding=1), nn.Dropout(drop_rate), nn.MaxPool1d(2)]
        ]
        conv_modules = []
        for conv_layer in conv_layers:
            conv_modules.extend(conv_layer)
        self.ConvBlocks = nn.Sequential(*conv_modules)

        # Bi-LSTM
        # self.bilstms = BiLSTMBlock(in_channels=conv_dim[3], out_channels=transformer_io_channels, drop_rate=drop_rate)

        # self.patch_embed = PatchEmbed(projection_dim=transformer_io_channels, seq_len=seq_len)
            
        self.transformers = Transformer(
            io_channels=transformer_io_channels,
            seq_len=seq_len,
            feedforward_dim=feedforward_dim,
            drop_rate=drop_rate,
            num_transformer_layers=num_transformer_layers,
            attn_width=attn_width,
            att_mode=att_mode,
            trans_mode=trans_mode
        )

    def forward(self, x):
        x = self.ConvBlocks(x)

        # x = self.bilstms(x)

        # x = self.patch_embed(x)
        x = self.transformers(x)

        return x


class UpSamplingBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        out_samples,
        kernel_size,
        up_size,
        kernel_l1_alpha,
        bias_l1_alpha,
    ):
        super().__init__()

        assert kernel_l1_alpha >= 0.0
        assert bias_l1_alpha >= 0.0

        self.out_samples = out_samples

        self.conv_padding_same = (
            (kernel_size - 1) // 2,
            kernel_size - 1 - (kernel_size - 1) // 2,
        )

        self.upsampling = nn.Upsample(scale_factor=up_size)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.relu = nn.ReLU()

        if kernel_l1_alpha > 0.0:
            self.conv.weight.register_hook(
                lambda grad: grad.data
                + kernel_l1_alpha * torch.sign(self.conv.weight.data)
            )
        if bias_l1_alpha > 0.0:
            self.conv.bias.register_hook(
                lambda grad: grad.data + bias_l1_alpha * torch.sign(self.conv.bias.data)
            )

    def forward(self, x):
        x = self.upsampling(x)
        
        x = x[:, :, : self.out_samples]
        x = F.pad(x, self.conv_padding_same, "constant", 0)
        x = self.conv(x)
        x = self.relu(x)

        return x
      

class IdentityNTuple(nn.Identity):
    def __init__(self, *args, ntuple: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        assert ntuple >= 1
        self.ntuple = ntuple

    def forward(self, input: torch.Tensor):
        if self.ntuple > 1:
            return (super().forward(input),) * self.ntuple
        else:
            return super().forward(input)


class Blocklstm(nn.Module):
    def __init__(self, seq_len, drop_rate):
        super().__init__()
        self.lstm = nn.LSTM(input_size=seq_len, hidden_size=seq_len, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)

        return x


class PixelAttention(nn.Module): # Substitution of standard attention layer
    def __init__(self, in_channels, ratio=2): # hidden_dim must be (input dimension/2 +1)
        super().__init__()
        self.fc = nn.Sequential(
                nn.Conv1d(in_channels, in_channels//ratio, 1, bias=False),
                nn.GELU(),
                nn.Conv1d(in_channels//ratio, in_channels, 1, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x):
        x_weight = self.fc(x)
        x = x * x_weight

        return x
    

class Decoder(nn.Module):

    def __init__(
        self,
        conv_channels=[64, 64, 32, 32, 16, 16, 8],
        conv_kernels=[3, 5, 5, 7, 7, 9, 11],
        up_kernels=[2, 2, 2, 2, 2, 2, 2],
        resconv_kernels1=[3, 3, 3],
        resconv_kernels2=[2, 2, 2],
        transformer_io_channels=96,
        drop_rate=0.1,
        out_samples=6000,
        conv_kernel_l1_regularization=0.0,
        conv_bias_l1_regularization=0.0
    ):
        super().__init__()

        self.res_convs1 = nn.Sequential(
            *[
                ResConvBlock(io_channels=transformer_io_channels, kernel_size=kers, drop_rate=drop_rate)
                for kers in resconv_kernels1
            ]
        )

        self.res_convs2 = nn.Sequential(
            *[
                ResConvBlock(io_channels=transformer_io_channels, kernel_size=kers, drop_rate=drop_rate)
                for kers in resconv_kernels2
            ]
        )

        # self.att = PixelAttention(transformer_io_channels)

        crop_sizes = [out_samples]
        for i in range(len(conv_kernels)-1):
            crop_sizes.insert(0, math.ceil(crop_sizes[0] / up_kernels[i]))

        self.upsamplings = nn.Sequential(
            *[
                UpSamplingBlock(
                    in_channels=inc,
                    out_channels=outc,
                    out_samples=crop,
                    kernel_size=kers,
                    up_size=ups,
                    kernel_l1_alpha=conv_kernel_l1_regularization,
                    bias_l1_alpha=conv_bias_l1_regularization,
                )
                for inc, outc, crop, kers, ups in zip(
                    [transformer_io_channels] + conv_channels[:-1],
                    conv_channels,
                    crop_sizes,
                    conv_kernels,
                    up_kernels
                )
            ]
        )

        self.conv_out = nn.Conv1d(in_channels=conv_channels[-1], out_channels=1, kernel_size=11, padding=5)

    def forward(self, x):

        x1 = self.res_convs1(x)
        x1 = x1 + x

        x2 = self.res_convs2(x1)
        x2 = x2 + x1

        x = self.upsamplings(x2)

        x = self.conv_out(x)
        x = x.sigmoid()

        return x
    