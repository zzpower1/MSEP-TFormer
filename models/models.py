import torch
import torch.nn as nn
from models.base import *


class Predict(nn.Module):
    def __init__(self, min_size, projection_dim, final_dim, num_predictions, drop_rate):
        super(Predict, self).__init__()

        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(projection_dim, num_predictions)

        conv_layers = []
        mlp_layers = []
        for _ in range(num_predictions):
            conv_layer = nn.Sequential(
                nn.Conv1d(projection_dim, final_dim, 1, 1),
                nn.BatchNorm1d(final_dim),
                nn.GELU()
            )
            conv_layers.append(conv_layer)

            mlp_layer = nn.Sequential(
                nn.Dropout(p=drop_rate),
                nn.Linear(min_size*final_dim, 1)
            )
            mlp_layers.append(mlp_layer)

        self.post_convs = nn.ModuleList(conv_layers)
        self.mlps = nn.ModuleList(mlp_layers)

    def forward(self, x):
        outputs=[]
        for conv, mlp in zip(self.post_convs, self.mlps):
            x_conv = conv(x)
            x_mlp = mlp(x_conv.flatten(1))
            outputs.append(x_mlp)
            
        return outputs


# class Fusion(nn.Module):
#     def __init__(self, channel):
#         super(Fusion, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(channel, channel, 3, 1, 1),
#             nn.BatchNorm1d(channel),
#             nn.GELU(),
#             nn.Dropout(p=0.1)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv1d(channel, channel, 3, 1, 1),
#             nn.BatchNorm1d(channel),
#             nn.GELU(),
#             nn.Dropout(p=0.1)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv1d(2*channel, channel, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x1, x2):
#         x1 = self.conv1(x1)
#         x2 = self.conv2(x2)
#         x3 = torch.cat([x1, x2], dim=1)
#         x3 = self.conv3(x3)
#         x_out = x3*x2+(1-x3)*x1

#         return x_out


class ABlock(nn.Module):
    def __init__(self, g, drop_rate):
        super(ABlock, self).__init__()
        self.act = nn.Sequential(
            nn.BatchNorm1d(g),
            nn.GELU(),
            nn.Dropout(p=drop_rate)
        )

    def forward(self, x):
        x = self.act(x)

        return x


class Fusion(nn.Module):
    def __init__(self, channel, drop_rate):
        super(Fusion, self).__init__()
        self.depthwise = nn.Conv1d(channel, channel, kernel_size=5, stride=1, padding=2, groups=channel)
        self.pointwise = nn.Conv1d(channel, channel, 1)
        self.act_d = ABlock(channel, drop_rate)
        self.act_p = ABlock(channel, drop_rate)

        self.act = nn.Sigmoid()

    def forward(self, x1, x2):
        res = x1
        x_weight = self.act(x2)

        x1 = self.depthwise(x1)
        x1 = self.act_d(x1)
        x1 = x1 * x_weight

        x1 = self.pointwise(x1)
        x1 = self.act_p(x1)

        return x1 + res


class Enhance(nn.Module):
    def __init__(self, g, drop_rate):
        super(Enhance, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(g, g, kernel_size=1),
            nn.BatchNorm1d(g),
            nn.GELU(),
            nn.Dropout(p=drop_rate),
            nn.Conv1d(g, g, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(g),
            nn.GELU(),
            nn.Dropout(p=drop_rate),
            nn.Conv1d(g, g, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm1d(g),
            nn.GELU(),
            nn.Dropout(p=drop_rate)
        )

    def forward(self, x):

        return self.conv(x)


class FF(nn.Module):
    def __init__(self, transformer_io_channels, drop_rate):
        super(FF, self).__init__()

        # self.transformer = TransformerLayer(io_channels=transformer_io_channels, seq_len=min_size, feedforward_dim=feedforward_dim, drop_rate=drop_rate, att_mode=att_mode, trans_mode=trans_mode)
        self.fuse = Fusion(transformer_io_channels, drop_rate)
        self.eh = Enhance(transformer_io_channels, drop_rate)

    def forward(self, x1, x2):
        # x1 = self.transformer(x1)
        x = self.fuse(x1, x2)
        x = self.eh(x)

        return x
    

class UNet(nn.Module):
    """
    att_mode: 'PA', 'CA', 'SA', 'SEA', ''GF'
    trans_mode: 'DCT', 'None'
    """
    def __init__(self, conv_dim=[3, 32, 64], resconv_kernels=[5, 5, 5], min_size=47, transformer_io_channels=96, final_dim=32, num_transformer_layers=3, drop_rate=0.1, att_mode='GF', trans_mode='None'):
        super(UNet, self).__init__()
        conv_dim = conv_dim + [transformer_io_channels]

        self.encoder = Encoder(
            conv_dim=conv_dim, 
            seq_len=min_size, 
            transformer_io_channels=transformer_io_channels, 
            feedforward_dim=256, 
            drop_rate=drop_rate, 
            num_transformer_layers=num_transformer_layers, 
            att_mode=att_mode, 
            trans_mode=trans_mode
        )
        self.res_convs = nn.Sequential(
            *[
                ResConvBlock(
                    io_channels=transformer_io_channels, kernel_size=kers, drop_rate=drop_rate
                )
                for kers in resconv_kernels
            ]
        )
        self.transformers1 = Transformer(
            io_channels=transformer_io_channels,
            seq_len=min_size,
            feedforward_dim=256,
            drop_rate=drop_rate,
            num_transformer_layers=2,
            att_mode=att_mode,
            trans_mode=trans_mode
        )
        self.transformers2 = Transformer(
            io_channels=transformer_io_channels,
            seq_len=min_size,
            feedforward_dim=256,
            drop_rate=drop_rate,
            num_transformer_layers=2,
            att_mode=att_mode,
            trans_mode=trans_mode
        )
        self.decoder_p = Decoder(transformer_io_channels=transformer_io_channels, drop_rate=drop_rate)
        self.decoder_s = Decoder(transformer_io_channels=transformer_io_channels, drop_rate=drop_rate)

        self.ff1 = FF(transformer_io_channels, drop_rate)
        self.ff2 = FF(transformer_io_channels, drop_rate)
        self.ff3 = FF(transformer_io_channels, drop_rate)

        self.predict_t = Predict(min_size, transformer_io_channels, final_dim=final_dim, num_predictions=1, drop_rate=drop_rate)
        self.predict_l = Predict(min_size, transformer_io_channels, final_dim=final_dim, num_predictions=2, drop_rate=drop_rate)
        self.predict_m = Predict(min_size, transformer_io_channels, final_dim=final_dim, num_predictions=1, drop_rate=drop_rate)

    def forward(self, x_in):
        x_in = self.encoder(x_in)

        x_ps = self.transformers1(x_in)
        
        x_s = self.decoder_s(x_ps)
        x_p = self.decoder_p(x_ps)

        x_tra = self.transformers2(x_in)

        x_ft = self.ff1(x_tra, x_tra)
        x_t = self.predict_t(x_ft)[0]

        x_fl = self.ff2(x_tra, x_ft)
        x_e, x_d = self.predict_l(x_fl)

        x_fm = self.ff3(x_tra, x_fl)
        x_m = self.predict_m(x_fm)[0]

        return x_m, x_e, x_d, x_t, x_p, x_s
