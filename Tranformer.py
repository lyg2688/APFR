
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads

    def forward(self, querys, keys, values):

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (split_size ** 0.5)
        scores = F.softmax(scores, dim=3)
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out


class Block(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(Block, self).__init__()

        self.MHA = MultiHeadAttention(dim, num_heads)
        self.Linear = nn.Linear(dim, dim)

    def forward(self, q, k, v):

        MHA_out = self.MHA(q, k, v)
        out = self.Linear(MHA_out)

        return out


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.block1 = Block(dim)

    def forward(self, q, k, v):
        x = self.block1(q, k, v)

        return x


class Transformer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Transformer, self).__init__()

        self.k = nn.Linear(in_dim, out_dim)
        self.q = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)

        self.Atte = Attention(out_dim)
        self.out = nn.Linear(out_dim, 1)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn = self.Atte(q, k, v)
        out = self.out(attn)

        return out
