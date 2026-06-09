import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ms_gcn import MultiScale_GraphConv as MS_GCN
from .ms_tcn import MultiScale_TemporalConv as MS_TCN
from .ms_gtcn import SpatialTemporal_MS_GCN, UnfoldTemporalWindows
from .mlp import MLP
from .activation import activation_factory

from .graph import Graph

class MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales=2,
                 window_size=3,
                 window_stride=1,
                 window_dilation=1,
                 embed_factor=1,
                 activation='relu'):
        super().__init__()
        self.window_size = window_size
        self.out_channels = out_channels
        self.embed_channels_in = self.embed_channels_out = out_channels // embed_factor
        if embed_factor == 1:
            self.in1x1 = nn.Identity()
            self.embed_channels_in = self.embed_channels_out = in_channels
            # The first STGC block changes channels right away; others change at collapse
            if in_channels == 3:
                self.embed_channels_out = out_channels
        else:
            self.in1x1 = MLP(in_channels, [self.embed_channels_in])

        self.gcn3d = nn.Sequential(
            UnfoldTemporalWindows(window_size, window_stride, window_dilation),
            SpatialTemporal_MS_GCN(
                in_channels=self.embed_channels_in,
                out_channels=self.embed_channels_out,
                A_binary=A_binary,
                num_scales=num_scales,
                window_size=window_size,
                use_Ares=True
            )
        )

        self.out_conv = nn.Conv3d(self.embed_channels_out, out_channels, kernel_size=(1, self.window_size, 1))
        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, _, T, V = x.shape
        x = self.in1x1(x)
        # Construct temporal windows and apply MS-GCN
        x = self.gcn3d(x)

        # Collapse the window dimension
        x = x.view(N, self.embed_channels_out, -1, self.window_size, V)
        x = self.out_conv(x).squeeze(dim=3)
        x = self.out_bn(x)

        # no activation
        return x


class MultiWindow_MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_sizes=[3,5],    # [3,5]
                 window_stride=1,
                 window_dilations=[1,1]):   # [1,1]

        super().__init__()
        self.gcn3d = nn.ModuleList([
            MS_G3D(
                in_channels,
                out_channels,
                A_binary,
                num_scales,
                window_size,
                window_stride,
                window_dilation
            )
            for window_size, window_dilation in zip(window_sizes, window_dilations)
        ])

        # self.global_gcn3d = SpatialTemporal_MS_GCN(
        #         in_channels=in_channels,
        #         out_channels=out_channels,
        #         A_binary=A_binary,
        #         num_scales=num_scales,
        #         window_size=1,
        #         use_Ares=True,
        #         activation='linear'
        #     )

    def forward(self, x):
        # Input shape: (N, C, T, V)
        res = x
        out_sum = 0
        for gcn3d in self.gcn3d:
            out_sum += gcn3d(x)

        # out_sum = self.global_gcn3d(out_sum)
        # no activation
        out_features = F.relu(out_sum, inplace=True) + res
        return out_features
        # return out_sum






def use(self):
    self.linked_edges_p2p = [(0, 1),(1, 2), (1, 3), (1, 4), (1, 5)]
    self.node_nums_part = self.args.branch_g
    self.part_num_scales = 3
    self.graph_p = Graph(self.node_nums_part, self.linked_edges_p2p)
    self.stgc_p2p = MultiWindow_MS_G3D(self.hidden * 4, self.hidden * 4, self.graph_p.A_binary, self.part_num_scales)