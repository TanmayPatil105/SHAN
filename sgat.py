import torch as th
from torch import nn

import numpy as np
import torch.nn.functional as F
from dgl import function as fn
from dgl._ffi.base import DGLError
from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch.utils import Identity


sig = nn.Sigmoid()
hardtanh = nn.Hardtanh(0,1)
gamma = -0.1
zeta = 1.1
beta = 0.66
eps = 1e-20
const1 = beta*np.log(-gamma/zeta + eps)

def l0_train(logAlpha, min, max):
    U = th.rand(logAlpha.size()).type_as(logAlpha) + eps
    s = sig((th.log(U / (1 - U)) + logAlpha) / beta)
    s_bar = s * (zeta - gamma) + gamma

    # @values : [ 0 - 1]
    mask = F.hardtanh(s_bar, min, max)
    return mask

def l0_test(logAlpha, min, max):
    s = sig(logAlpha/beta)
    s_bar = s * (zeta - gamma) + gamma
    mask = F.hardtanh(s_bar, min, max)
    return mask

def get_loss2(logAlpha):
    return sig(logAlpha - const1)


class SGATConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
    ):
        super(SGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False
            )
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
        self.attn_l = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.attn_r = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.has_linear_res = False
        self.has_explicit_bias = False
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=bias
                )
                self.has_linear_res = True
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)

        if bias and not self.has_linear_res:
            self.bias = nn.Parameter(
                th.FloatTensor(size=(num_heads * out_feats,))
            )
            self.has_explicit_bias = True
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.has_explicit_bias:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.constant_(self.res_fc.bias, 0)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_weight=None, get_attention=False):
        self.graph = graph

        with self.graph.local_scope():
            if not self._allow_zero_in_degree:
                if (self.graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            # Transformation Matrix Tφi
            src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(
                *src_prefix_shape, self._num_heads, self._out_feats
            )
            if self.graph.is_block:
                feat_dst = feat_src[: self.graph.number_of_dst_nodes()]
                h_dst = h_dst[: self.graph.number_of_dst_nodes()]
                dst_prefix_shape = (
                    self.graph.number_of_dst_nodes(),
                ) + dst_prefix_shape[1:]

            """
            attn_l   ->   Wh_i
            attn_r   ->   Wh_j
            feat_src ->   h_i`
            feat_dst ->   h_j`
            """
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            self.graph.srcdata.update({'ft': feat_src, 'el': el})
            self.graph.dstdata.update({'er': er})
            
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            self.graph.apply_edges(self.edge_attention)

            # compute attention
            self.edge_softmax()

            # message passing
            self.graph.edata['a_drop'] = self.attn_drop(self.graph.edata['a'])
            self.graph.update_all(fn.u_mul_e('ft', 'a_drop', 'ft'), fn.sum('ft', 'ft'))
            rst = self.graph.ndata['ft']
            
            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(
                    *dst_prefix_shape, -1, self._out_feats
                )
                rst = rst + resval
            
            # bias
            if self.has_explicit_bias:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)),
                    self._num_heads,
                    self._out_feats
                )
            
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, self.graph.edata['a']
            else:
                return rst

    def norm(self, edges):
        # normalize attention
        a = edges.data['a'] / edges.dst['z']
        return {'a' : a}

    def edge_attention(self, edges):
        tmp = edges.src['el'] + edges.dst['er']
        logits = tmp

        if self.training:
            m = l0_train(tmp, 0, 1)
        else:
            m = l0_test(tmp, 0, 1)
            self.loss = get_loss2(logits[:,0,:]).sum()

        return {'e': m}

    def normalize(self, logits):
        self._logits_name = "_logits"
        self._normalizer_name = "_norm"
        self.graph.edata[self._logits_name] = logits

        self.graph.update_all(fn.copy_e(self._logits_name, self._logits_name),
                          fn.sum(self._logits_name, self._normalizer_name))

        return self.graph.edata.pop(self._logits_name), self.graph.ndata.pop(self._normalizer_name)

    def edge_softmax(self):
        scores, normalizer = self.normalize(self.graph.edata.pop('e'))
        self.graph.ndata['z'] = normalizer[:,0,:].unsqueeze(1)

        self.graph.edata['a'] = scores[:,0,:].unsqueeze(1)
