import torch
from torch import nn
# import torch_dct as dct
import math
import torch.nn.functional as F


class DotProductAttention(nn.Module):
    _epsilon = 1e-6
    def __init__(self, attn_width=None):
        super(DotProductAttention, self).__init__()
        self.attn_width = attn_width
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V):
        d_model = Q.size()[-1]
        attention_scores = torch.bmm(Q, K.transpose(1, 2))/math.sqrt(d_model)

        attention_scores = torch.exp(attention_scores - torch.max(attention_scores, dim=-1, keepdim=True).values)

        if self.attn_width is not None:
            mask = (
                torch.ones(attention_scores.shape[-2:], dtype=torch.bool, device=attention_scores.device)
                .tril(self.attn_width // 2 - 1)
                .triu(-self.attn_width // 2)
            )
            attention_scores = attention_scores.where(mask, 0)

        attention_weights = self.softmax(attention_scores)

        return torch.bmm(attention_weights, V)


class SelfAttention(nn.Module):
    def __init__(self, in_channels, hidden_size, num_heads=4, attn_width=None, trans_mode='None'):
        super(SelfAttention, self).__init__()
        self.trans_mode=trans_mode
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.Wq = nn.Linear(in_channels, hidden_size, bias=False)
        self.Wk = nn.Linear(in_channels, hidden_size, bias=False)
        self.Wv = nn.Linear(in_channels, hidden_size, bias=False)
        self.Wo = nn.Linear(hidden_size, in_channels, bias=False)
        self.attention = DotProductAttention(attn_width=attn_width)

    def _transpose_qkv(self, X):
        self._batch, self._seq_len = X.size()[0], X.size()[1]
        X = X.view([self._batch, self._seq_len, self.num_heads, self.hidden_size//self.num_heads])
        X = X.permute([0, 2, 1, 3])
        return X.contiguous().view([self._batch*self.num_heads, self._seq_len, self.hidden_size//self.num_heads])

    def _transpose_output(self, X):
        X = X.view([self._batch, self.num_heads, -1, self.hidden_size//self.num_heads])
        X = X.permute([0, 2, 1, 3])
        return X.contiguous().view([self._batch, -1, self.hidden_size])

    def forward(self, x):
        x1 = x

        if self.trans_mode == 'DCT':
            x = dct.dct(x)

        Q = self._transpose_qkv(self.Wq(x))
        K = self._transpose_qkv(self.Wk(x))
        V = self._transpose_qkv(self.Wv(x1))
        
        output = self.attention(Q, K, V)
        output_concat = self._transpose_output(output)
        out = self.Wo(output_concat)

        return out


# class SelfAttention(nn.Module):
#     """Single head attention
#     Input shape: (N,C,L)
#     """

#     _epsilon = 1e-6

#     def __init__(self, in_channels, hidden_size, attn_width=None, trans_mode='None'):
#         super().__init__()
#         self.attn_width = attn_width
#         self.trans_mode=trans_mode
#         self.Wx = nn.Parameter(torch.empty((in_channels, hidden_size)))
#         self.Wt = nn.Parameter(torch.empty((in_channels, hidden_size)))
#         self.bh = nn.Parameter(torch.empty(hidden_size))
#         self.Wa = nn.Parameter(torch.empty((hidden_size, 1)))
#         self.ba = nn.Parameter(torch.empty(1))

#         self._reset_parameters()

#     def _reset_parameters(self) -> None:
#         nn.init.xavier_uniform_(self.Wx)
#         nn.init.xavier_uniform_(self.Wt)
#         nn.init.xavier_uniform_(self.Wa)
#         nn.init.zeros_(self.bh)
#         nn.init.zeros_(self.ba)

#     def forward(self, x):

#         # (N,L,C),(C,d) -> (N,L,1,d)
#         q = torch.matmul(x, self.Wt).unsqueeze(2)

#         # (N,L,C),(C,d) -> (N,1,L,d)
#         k = torch.matmul(x, self.Wx).unsqueeze(1)

#         if self.trans_mode == 'DCT':
#             q = dct.dct(q)
#             k = dct.dct(k)

#         # (N,L,1,d),(N,1,L,d),(d,) -> (N,L,L,d)
#         h = torch.tanh(q + k + self.bh)

#         # (N,L,d),(d,1) -> (N,L,L,1) -> (N,L,L)
#         e = (torch.matmul(h, self.Wa) + self.ba).squeeze(-1)

#         # (N,L,L)
#         e = torch.exp(e - torch.max(e, dim=-1, keepdim=True).values)

#         # Masked attention
#         if self.attn_width is not None:
#             mask = (
#                 torch.ones(e.shape[-2:], dtype=torch.bool, device=e.device)
#                 .tril(self.attn_width // 2 - 1)
#                 .triu(-self.attn_width // 2)
#             )
#             e = e.where(mask, 0)

#         # (N,L,L)
#         s = torch.sum(e, dim=-1, keepdim=True)
#         a = e / (s + self._epsilon)

#         # (N,L,L),(N,L,C) -> (N,L,C)
#         v = torch.matmul(a, x)

#         return v


class GlobalFilter(nn.Module): # Substitution of standard attention layer
    def __init__(self, in_channels, seq_len, attn_width=None): # hidden_dim must be (input dimension/2 +1)
        super().__init__()
        self.attn_width = attn_width
        self.complex_weight = nn.Parameter(torch.randn(seq_len, in_channels//2 +1, 2, dtype=torch.float32) * 0.02)

    def forward(self, x): #  B * sequencelength * channel
        x = torch.fft.rfft2(x, dim=(-2,-1), norm='ortho') # B * sequencelength * (channel/2 +1)
        weight = torch.view_as_complex(self.complex_weight)
        if self.attn_width is not None:
            mask = (
                torch.ones(weight.shape[-2:], dtype=torch.bool, device=weight.device)
                .tril(self.attn_width // 2 - 1)
                .triu(-self.attn_width // 2)
            )
            weight = weight.where(mask, 0)

        x = x * weight  # B * sequencelength * (channel/2 +1)
        x = torch.fft.irfft2(x, dim=(-2, -1), norm='ortho') # B * sequencelength * channel

        return x


def dwt(x):
    x1 = x[:, :, 0::2] / 2
    x2 = x[:, :, 1::2] / 2
    x_l = x1 + x2
    x_h = x1 - x2

    return x_l, x_h

def iwt(x1, x2):
    b, c, l1 = x1.size()
    l2 = x2.size()[-1]
    h = torch.zeros([b, c, l1+l2]).to(x1.device)

    h[:, :, 0::2] = x1 + x2
    h[:, :, 1::2] = x1 - x2

    return h


class DWTFilter(nn.Module): # Substitution of standard attention layer
    def __init__(self, in_channels, seq_len): # hidden_dim must be (input dimension/2 +1)
        super().__init__()
        self.weight1 = nn.Parameter(torch.randn(seq_len, in_channels//2, dtype=torch.float32))
        self.weight2 = nn.Parameter(torch.randn(seq_len, in_channels//2, dtype=torch.float32))

    def forward(self, x): #  B * sequencelength * channel
        x_l, x_h = dwt(x)
        x_l = x_l * self.weight1
        x_h = x_h * self.weight2
        x = iwt(x_l, x_h)

        return x
    

class PixelAttention(nn.Module): # Substitution of standard attention layer
    def __init__(self, seq_len, ratio=0.5, trans_mode='None'): # hidden_dim must be (input dimension/2 +1)
        super().__init__()
        self.trans_mode=trans_mode
        self.fc = nn.Sequential(
            nn.Linear(seq_len, int(seq_len//ratio), bias=False),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(int(seq_len//ratio), seq_len, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        if self.trans_mode == 'DCT':
            x1 = dct.dct(x)
            x_weight = self.fc(x1)
            x1 = x1 * x_weight
            x = dct.idct(x1)

        elif self.trans_mode == 'None':
            x_weight = self.fc(x)
            x = x * x_weight

        return x.permute(0, 2, 1)
    

class ChannelAttention(nn.Module): # Substitution of standard attention layer
    def __init__(self, in_channels, ratio=4, trans_mode='None'): # hidden_dim must be (input dimension/2 +1)
        super().__init__()
        self.trans_mode=trans_mode
        self.avg_pool=nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                nn.Conv1d(in_channels, int(in_channels//ratio), 1, bias=False),
                nn.Dropout(p=0.1),
                nn.GELU(),
                nn.Conv1d(int(in_channels//ratio), in_channels, 1, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        if self.trans_mode == 'DCT':
            x1 = dct.dct(x)
            x_weight = self.fc(self.avg_pool(x1))
            x1 = x1 * x_weight
            x = dct.idct(x1)

        elif self.trans_mode == 'None':
            x_weight = self.fc(self.avg_pool(x))
            x = x * x_weight

        return x.permute(0, 2, 1)


class SpatialAttention(nn.Module): # Substitution of standard attention layer
    def __init__(self, trans_mode='None'): # hidden_dim must be (input dimension/2 +1)
        super().__init__()
        self.trans_mode=trans_mode
        self.conv = nn.Sequential(
            nn.Conv1d(2, 1, 7, padding=3, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        if self.trans_mode == 'DCT':
            x1 = dct.dct(x)
            avg_out = torch.mean(x1, dim=1, keepdim=True)
            max_out, _ = torch.max(x1, dim=1, keepdim=True)
            x_cat = torch.cat([avg_out, max_out], dim=1)
            x_weight = self.conv(x_cat)
            x1 = x1 * x_weight
            x = dct.idct(x1)
        elif self.trans_mode == 'None':
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            x_cat = torch.cat([avg_out, max_out], dim=1)
            x_weight = self.conv(x_cat)
            x = x * x_weight

        return x.permute(0, 2, 1)
    