import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List
from torch import Tensor
from timm.models.layers import DropPath
from torch.autograd import Variable
from utils import my_softmax, get_offdiag_indices, gumbel_softmax
from torch.nn import MultiheadAttention

_EPS = 1e-10

class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)

# class Mlp(nn.Module):
#     """Multilayer perceptron."""
#
#     def __init__(
#             self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
#     ):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # qkv:[3,1,8,220,32]   qkv[0]、qkv[1]、qkv[2]:[1,8,220,32]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, drop_path_ratio=0.1):
        super(EncoderBlock, self).__init__()
        self.nomal_layer1 = nn.LayerNorm(embed_dim)
        self.atten = Attention(dim=embed_dim)
        self.mlp = MLP(n_in=embed_dim, n_hid=embed_dim, n_out=embed_dim)
        self.nomal_layer2 = nn.LayerNorm(embed_dim)
        self.drop_path1 = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.atten(self.nomal_layer1(x)))
        x = x + self.drop_path2(self.mlp(self.nomal_layer2(x)))
        return x

class MLPAttenEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
        super(MLPAttenEncoder, self).__init__()

        self.factor = factor

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        # self.atten1 = EncoderBlock(embed_dim=n_hid)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        # self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.atten3 = EncoderBlock(embed_dim=n_hid)
        self.atten4 = EncoderBlock(embed_dim=n_hid)
        self.atten5 = EncoderBlock(embed_dim=n_hid)
        self.atten6 = EncoderBlock(embed_dim=n_hid)
        if self.factor:
            self.mlp7 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp7 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_normal_(m.weight.data)
    #             m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        # 通过边到节点的信息聚合操作，得到每个节点的邻居信息的平均值
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        # 通过节点到边的信息转换操作，得到表示每条边的信息
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        #print("Inputs size:", inputs.size())
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        #print("New Inputs size:", x.size())
        x = self.mlp1(x)  # 2-layer ELU net per node
        # print("atten1 x size:", x.size())
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x
        #print("mlp2 x size:", x.size())
        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.atten3(x)
            x = self.atten4(x)
            x = self.atten5(x)
            x = self.atten6(x)
            #print("atten3 x size:", x.size())
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp7(x)
            #print("mlp4 x size:", x.size())
        else:
            x = self.atten3(x)
            x = self.atten4(x)
            x = self.atten5(x)
            x = self.atten6(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp7(x)
        #print("x size:", x.size())
        return self.fc_out(x)

# class MLP(nn.Module):
#     """Two-layer fully-connected ELU net with batch norm."""
#
#     def __init__(self, n_in, n_hid, n_out, do_prob=0.):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(n_in, n_hid)
#         self.fc2 = nn.Linear(n_hid, n_out)
#         self.bn = nn.BatchNorm1d(n_out)
#         self.dropout_prob = do_prob
#
#         self.init_weights()
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight.data)
#                 m.bias.data.fill_(0.1)
#             elif isinstance(m, nn.BatchNorm1d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def batch_norm(self, inputs):
#         x = inputs.view(inputs.size(0) * inputs.size(1), -1)
#         x = self.bn(x)
#         return x.view(inputs.size(0), inputs.size(1), -1)
#
#     def forward(self, inputs):
#         # Input shape: [num_sims, num_things, num_features]
#         x = F.elu(self.fc1(inputs))
#         x = F.dropout(x, self.dropout_prob, training=self.training)
#         x = F.elu(self.fc2(x))
#         return self.batch_norm(x)

class CNN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(CNN, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=None, padding=0,
                                 dilation=1, return_indices=False,
                                 ceil_mode=False)

        self.conv1 = nn.Conv1d(n_in, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(n_hid)
        self.conv2 = nn.Conv1d(
            n_hid, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(n_hid)
        self.conv_predict = nn.Conv1d(n_hid, n_out, kernel_size=1)
        self.conv_attention = nn.Conv1d(n_hid, 1, kernel_size=1)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        # Input shape: [num_sims * num_edges, num_dims, num_timesteps]

        x = F.relu(self.conv1(inputs))
        x = self.bn1(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        pred = self.conv_predict(x)
        attention = my_softmax(self.conv_attention(x), axis=2)

        edge_prob = (pred * attention).mean(dim=2)
        return edge_prob

class MLPEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
        super(MLPEncoder, self).__init__()

        self.factor = factor

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        # 通过边到节点的信息聚合操作，得到每个节点的邻居信息的平均值
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        # 通过节点到边的信息转换操作，得到表示每条边的信息
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        #print("Inputs size:", inputs.size())
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        #print("New Inputs size:", x.size())
        x = self.mlp1(x)  # 2-layer ELU net per node
        #print("mlp1 x size:", x.size())
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x
        #print("mlp2 x size:", x.size())
        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            #print("mlp3 x size:", x.size())
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
            #print("mlp4 x size:", x.size())
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        #print("x size:", x.size())
        return self.fc_out(x)

class CNNEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
        super(CNNEncoder, self).__init__()
        self.dropout_prob = do_prob

        self.factor = factor

        self.cnn = CNN(n_in * 2, n_hid, n_hid, do_prob)
        self.mlp1 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
        self.fc_out = nn.Linear(n_hid, n_out)

        if self.factor:
            print("Using factor graph CNN encoder.")
        else:
            print("Using CNN encoder.")

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge_temporal(self, inputs, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.

        x = inputs.view(inputs.size(0), inputs.size(1), -1)

        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.view(inputs.size(0) * receivers.size(1),
                                   inputs.size(2), inputs.size(3))
        receivers = receivers.transpose(2, 1)

        senders = torch.matmul(rel_send, x)
        senders = senders.view(inputs.size(0) * senders.size(1),
                               inputs.size(2),
                               inputs.size(3))
        senders = senders.transpose(2, 1)

        # receivers and senders have shape:
        # [num_sims * num_edges, num_dims, num_timesteps]
        edges = torch.cat([receivers, senders], dim=1)
        return edges

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):

        # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send)
        x = self.cnn(edges)
        x = x.view(inputs.size(0), (inputs.size(1) - 1) * inputs.size(1), -1)
        x = self.mlp1(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp2(x)

            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp3(x)

        return self.fc_out(x)

# class TransformerEncoderLayer(nn.Module):
#
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
#                  activation="relu", normalize_before=False):
#         super().__init__()
#         self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#
#         self.activation = _get_activation_fn(activation)
#         self.normalize_before = normalize_before
#
#         self.debug_mode = False
#         self.debug_name = None
#
#     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
#         return tensor if pos is None else tensor + pos
#
#     def forward_post(self,
#                      src,
#                      src_mask: Optional[Tensor] = None,
#                      src_key_padding_mask: Optional[Tensor] = None,
#                      pos: Optional[Tensor] = None):
#         # print(src.shape,src_key_padding_mask.shape)
#         q = k = self.with_pos_embed(src, pos)
#         # print(q.shape,k.shape)
#         src2, corr = self.self_attn(q, k, value=src, attn_mask=src_mask,
#                                     key_padding_mask=src_key_padding_mask)
#
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src
#
#     def forward_pre(self, src,
#                     src_mask: Optional[Tensor] = None,
#                     src_key_padding_mask: Optional[Tensor] = None,
#                     pos: Optional[Tensor] = None):
#         src2 = self.norm1(src)
#         q = k = self.with_pos_embed(src2, pos)
#         src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
#                               key_padding_mask=src_key_padding_mask)[0]
#
#         src = src + self.dropout1(src2)
#         src2 = self.norm2(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
#         src = src + self.dropout2(src2)
#         return src
#
#     def forward(self, src,
#                 src_mask: Optional[Tensor] = None,
#                 src_key_padding_mask: Optional[Tensor] = None,
#                 pos: Optional[Tensor] = None):
#         if self.normalize_before:
#             return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
#         return self.forward_post(src, src_mask, src_key_padding_mask, pos)
#
# class MLPEncoderWithTransformer(nn.Module):
#     def __init__(self, n_in, n_hid, n_out, num_layers=6, nhead=8, dim_feedforward=2048, dropout=0.1):
#         super(MLPEncoderWithTransformer, self).__init__()
#
#         self.mlp_encoder = MLP(n_in, n_hid, n_hid)  # GNN编码器中的MLP
#
#         # 初始化Transformer编码器
#         encoder_layers = nn.TransformerEncoderLayer(
#             d_model=n_hid,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout
#         )
#         self.transformer_encoder = nn.TransformerEncoder(
#             encoder_layers,
#             num_layers=num_layers
#         )
#
#         self.fc_out = nn.Linear(n_hid, n_out)
#
#     def forward(self, inputs):
#         # 使用MLP对输入进行编码
#         x = inputs.view(inputs.size(0), inputs.size(1), -1)
#         x = self.mlp_encoder(x)
#         print(x.size)
#         x = x.transpose(0, 1)
#         # Transformer编码器处理MLP的输出
#         x = self.transformer_encoder(x)
#
#         # 将处理后的结果传递给输出层
#         return self.fc_out(x)


class SimulationDecoder(nn.Module):
    """Simulation-based decoder."""

    def __init__(self, loc_max, loc_min, vel_max, vel_min, suffix):
        super(SimulationDecoder, self).__init__()

        self.loc_max = loc_max
        self.loc_min = loc_min
        self.vel_max = vel_max
        self.vel_min = vel_min

        self.interaction_type = suffix

        if '_springs' in self.interaction_type:
            print('Using spring simulation decoder.')
            self.interaction_strength = .1
            self.sample_freq = 1
            self._delta_T = 0.1
            self.box_size = 5.
        elif '_charged' in self.interaction_type:
            print('Using charged particle simulation decoder.')
            self.interaction_strength = 1.
            self.sample_freq = 100
            self._delta_T = 0.001
            self.box_size = 5.
        elif '_charged_short' in self.interaction_type:
            print('Using charged particle simulation decoder.')
            self.interaction_strength = .1
            self.sample_freq = 10
            self._delta_T = 0.001
            self.box_size = 1.
        else:
            print("Simulation type could not be inferred from suffix.")

        self.out = None

        # NOTE: For exact reproduction, choose sample_freq=100, delta_T=0.001

        self._max_F = 0.1 / self._delta_T

    def unnormalize(self, loc, vel):
        loc = 0.5 * (loc + 1) * (self.loc_max - self.loc_min) + self.loc_min
        vel = 0.5 * (vel + 1) * (self.vel_max - self.vel_min) + self.vel_min
        return loc, vel

    def renormalize(self, loc, vel):
        loc = 2 * (loc - self.loc_min) / (self.loc_max - self.loc_min) - 1
        vel = 2 * (vel - self.vel_min) / (self.vel_max - self.vel_min) - 1
        return loc, vel

    def clamp(self, loc, vel):
        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        vel[over] = -torch.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        vel[under] = torch.abs(vel[under])

        return loc, vel

    def set_diag_to_zero(self, x):
        """Hack to set diagonal of a tensor to zero."""
        mask = torch.diag(torch.ones(x.size(1))).unsqueeze(0).expand_as(x)
        inverse_mask = torch.ones(x.size(1), x.size(1)) - mask
        if x.is_cuda:
            inverse_mask = inverse_mask.cuda()
        inverse_mask = Variable(inverse_mask)
        return inverse_mask * x

    def set_diag_to_one(self, x):
        """Hack to set diagonal of a tensor to one."""
        mask = torch.diag(torch.ones(x.size(1))).unsqueeze(0).expand_as(x)
        inverse_mask = torch.ones(x.size(1), x.size(1)) - mask
        if x.is_cuda:
            mask, inverse_mask = mask.cuda(), inverse_mask.cuda()
        mask, inverse_mask = Variable(mask), Variable(inverse_mask)
        return mask + inverse_mask * x

    def pairwise_sq_dist(self, x):
        xx = torch.bmm(x, x.transpose(1, 2))
        rx = (x ** 2).sum(2).unsqueeze(-1).expand_as(xx)
        return torch.abs(rx.transpose(1, 2) + rx - 2 * xx)

    def forward(self, inputs, relations, rel_rec, rel_send, pred_steps=1):
        # Input has shape: [num_sims, num_things, num_timesteps, num_dims]
        # Relation mx shape: [num_sims, num_things*num_things]

        # Only keep single dimension of softmax output
        relations = relations[:, :, 1]

        loc = inputs[:, :, :-1, :2].contiguous()
        vel = inputs[:, :, :-1, 2:].contiguous()

        # Broadcasting/shape tricks for parallel processing of time steps
        loc = loc.permute(0, 2, 1, 3).contiguous()
        vel = vel.permute(0, 2, 1, 3).contiguous()
        loc = loc.view(inputs.size(0) * (inputs.size(2) - 1),
                       inputs.size(1), 2)
        vel = vel.view(inputs.size(0) * (inputs.size(2) - 1),
                       inputs.size(1), 2)

        loc, vel = self.unnormalize(loc, vel)

        offdiag_indices = get_offdiag_indices(inputs.size(1))
        edges = Variable(torch.zeros(relations.size(0), inputs.size(1) *
                                     inputs.size(1)))
        if inputs.is_cuda:
            edges = edges.cuda()
            offdiag_indices = offdiag_indices.cuda()

        edges[:, offdiag_indices] = relations.float()

        edges = edges.view(relations.size(0), inputs.size(1),
                           inputs.size(1))

        self.out = []

        for _ in range(0, self.sample_freq):
            x = loc[:, :, 0].unsqueeze(-1)
            y = loc[:, :, 1].unsqueeze(-1)

            xx = x.expand(x.size(0), x.size(1), x.size(1))
            yy = y.expand(y.size(0), y.size(1), y.size(1))
            dist_x = xx - xx.transpose(1, 2)
            dist_y = yy - yy.transpose(1, 2)

            if '_springs' in self.interaction_type:
                forces_size = -self.interaction_strength * edges
                pair_dist = torch.cat(
                    (dist_x.unsqueeze(-1), dist_y.unsqueeze(-1)),
                    -1)

                # Tricks for parallel processing of time steps
                pair_dist = pair_dist.view(inputs.size(0), (inputs.size(2) - 1),
                                           inputs.size(1), inputs.size(1), 2)
                forces = (
                    forces_size.unsqueeze(-1).unsqueeze(1) * pair_dist).sum(
                    3)
            else:  # charged particle sim
                e = (-1) * (edges * 2 - 1)
                forces_size = -self.interaction_strength * e

                l2_dist_power3 = torch.pow(self.pairwise_sq_dist(loc), 3. / 2.)
                l2_dist_power3 = self.set_diag_to_one(l2_dist_power3)

                l2_dist_power3 = l2_dist_power3.view(inputs.size(0),
                                                     (inputs.size(2) - 1),
                                                     inputs.size(1),
                                                     inputs.size(1))
                forces_size = forces_size.unsqueeze(
                    1) / (l2_dist_power3 + _EPS)

                pair_dist = torch.cat(
                    (dist_x.unsqueeze(-1), dist_y.unsqueeze(-1)),
                    -1)
                pair_dist = pair_dist.view(inputs.size(0), (inputs.size(2) - 1),
                                           inputs.size(1), inputs.size(1), 2)
                forces = (forces_size.unsqueeze(-1) * pair_dist).sum(3)

            forces = forces.view(inputs.size(0) * (inputs.size(2) - 1),
                                 inputs.size(1), 2)

            if '_charged' in self.interaction_type:  # charged particle sim
                # Clip forces
                forces[forces > self._max_F] = self._max_F
                forces[forces < -self._max_F] = -self._max_F

            # Leapfrog integration step
            vel = vel + self._delta_T * forces
            loc = loc + self._delta_T * vel

            # Handle box boundaries
            loc, vel = self.clamp(loc, vel)

        loc, vel = self.renormalize(loc, vel)

        loc = loc.view(inputs.size(0), (inputs.size(2) - 1), inputs.size(1), 2)
        vel = vel.view(inputs.size(0), (inputs.size(2) - 1), inputs.size(1), 2)

        loc = loc.permute(0, 2, 1, 3)
        vel = vel.permute(0, 2, 1, 3)

        out = torch.cat((loc, vel), dim=-1)
        # Output has shape: [num_sims, num_things, num_timesteps-1, num_dims]

        return out


class MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid,
                 do_prob=0., skip_first=False):
        super(MLPDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([receivers, senders], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        pre_msg.size(2), self.msg_out_shape))
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.

        inputs = inputs.transpose(1, 2).contiguous()

        sizes = [rel_type.size(0), inputs.size(1), rel_type.size(1),
                 rel_type.size(2)]
        rel_type = rel_type.unsqueeze(1).expand(sizes)

        time_steps = inputs.size(1)
        assert (pred_steps <= time_steps)
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, 0::pred_steps, :, :]
        curr_rel_type = rel_type[:, 0::pred_steps, :, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        # Run n prediction steps
        for step in range(0, pred_steps):
            last_pred = self.single_step_forward(last_pred, rel_rec, rel_send,
                                                 curr_rel_type)
            preds.append(last_pred)

        sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                 preds[0].size(2), preds[0].size(3)]

        output = Variable(torch.zeros(sizes))
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i::pred_steps, :, :] = preds[i]

        pred_all = output[:, :(inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()


class RNNDecoder(nn.Module):
    """Recurrent decoder module."""

    def __init__(self, n_in_node, edge_types, n_hid,
                 do_prob=0., skip_first=False):
        super(RNNDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_hid, n_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)])
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_i = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_n = nn.Linear(n_in_node, n_hid, bias=True)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned recurrent interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, inputs, rel_rec, rel_send,
                            rel_type, hidden):

        # node2edge
        receivers = torch.matmul(rel_rec, hidden)
        senders = torch.matmul(rel_send, hidden)
        pre_msg = torch.cat([receivers, senders], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        self.msg_out_shape))
        if inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
            norm = float(len(self.msg_fc2)) - 1.
        else:
            start_idx = 0
            norm = float(len(self.msg_fc2))

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = torch.tanh(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = torch.tanh(self.msg_fc2[i](msg))
            msg = msg * rel_type[:, :, i:i + 1]
            all_msgs += msg / norm

        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2,
                                                                        -1)
        agg_msgs = agg_msgs.contiguous() / inputs.size(2)  # Average

        # GRU-style gated aggregation
        r = torch.sigmoid(self.input_r(inputs) + self.hidden_r(agg_msgs))
        i = torch.sigmoid(self.input_i(inputs) + self.hidden_i(agg_msgs))
        n = torch.tanh(self.input_n(inputs) + r * self.hidden_h(agg_msgs))
        hidden = (1 - i) * n + i * hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        pred = inputs + pred

        return pred, hidden

    def forward(self, data, rel_type, rel_rec, rel_send, pred_steps=1,
                burn_in=False, burn_in_steps=1, dynamic_graph=False,
                encoder=None, temp=None):

        inputs = data.transpose(1, 2).contiguous()

        time_steps = inputs.size(1)

        # inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # rel_type has shape:
        # [batch_size, num_atoms*(num_atoms-1), num_edge_types]

        hidden = Variable(
            torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape))
        if inputs.is_cuda:
            hidden = hidden.cuda()

        pred_all = []

        for step in range(0, inputs.size(1) - 1):

            if burn_in:
                if step <= burn_in_steps:
                    ins = inputs[:, step, :, :]
                else:
                    ins = pred_all[step - 1]
            else:
                assert (pred_steps <= time_steps)
                # Use ground truth trajectory input vs. last prediction
                if not step % pred_steps:
                    ins = inputs[:, step, :, :]
                else:
                    ins = pred_all[step - 1]

            if dynamic_graph and step >= burn_in_steps:
                # NOTE: Assumes burn_in_steps = args.timesteps
                logits = encoder(
                    data[:, :, step - burn_in_steps:step, :].contiguous(),
                    rel_rec, rel_send)
                rel_type = gumbel_softmax(logits, tau=temp, hard=True)

            pred, hidden = self.single_step_forward(ins, rel_rec, rel_send,
                                                    rel_type, hidden)
            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)

        return preds.transpose(1, 2).contiguous()

# def _get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")