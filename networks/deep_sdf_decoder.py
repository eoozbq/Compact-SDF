#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F
from networks.embedder import *
import torch.nn.init as init
import numpy as np
from torch.nn.utils import weight_norm
import tinycudann as tcnn


class SdfDecoder(nn.Module):
    def __init__(self, latent_size=256, hidden_dim=512,
                 skip_connection=True, tanh_act=False, ues_pe=False,
                 geo_init=True, input_size=None
                 ):
        super().__init__()
        self.latent_size = latent_size
        input_size = latent_size + 3 if input_size is None else input_size
        self.skip_connection = skip_connection
        self.tanh_act = tanh_act
        self.use_pe = ues_pe
        if self.use_pe:
            embed_fn, input_ch = get_embedder(8, input_dims=3)
            self.embed_fn = embed_fn
            input_size = input_size + input_ch
        self.input_size = input_size
        skip_dim = hidden_dim + self.input_size if skip_connection else hidden_dim

        self.block1 = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Linear(skip_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.block3 = nn.Linear(hidden_dim, 1)

        if geo_init:
            for m in self.block3.modules():
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, mean=2 * np.sqrt(np.pi) / np.sqrt(hidden_dim), std=0.000001)
                    init.constant_(m.bias, -0.5)

            for m in self.block2.modules():
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, mean=0.0, std=np.sqrt(2) / np.sqrt(hidden_dim))
                    init.constant_(m.bias, 0.0)

            for m in self.block1.modules():
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, mean=0.0, std=np.sqrt(2) / np.sqrt(hidden_dim))
                    init.constant_(m.bias, 0.0)

    def forward(self, x):
        '''
        x: concatenated xyz and shape features, shape: B, N, D+3
        '''
        if self.use_pe:
            xyz = x[:, -3:]
            input_pe = self.embed_fn(xyz)
            x = torch.cat([x, input_pe], 1)

        block1_out = self.block1(x)

        # skip connection, concat
        if self.skip_connection:
            block2_in = torch.cat([x, block1_out], dim=-1)
        else:
            block2_in = block1_out

        block2_out = self.block2(block2_in)

        out = self.block3(block2_out)

        if self.tanh_act:
            out = nn.Tanh()(out)

        return out


class grid_feature(nn.Module):
    """
    Simple MLP for predicting SDF value given a point tensor
    """

    def __init__(self, channels, world_size, xyz_min, xyz_max, **kwargs):
        super(grid_feature, self).__init__()
        self.channels = channels
        self.world_size = world_size
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.grid = nn.Parameter(torch.zeros(1, channels, world_size[0], world_size[1], world_size[2]))

    def forward(self, input):  # input xyz, size is [1096,8,3] <--sdf
        '''
        xyz: global coordinates to query
        '''
        xyz = input.reshape(-1, 3)
        shape = xyz.shape[:-1]
        ind_norm = xyz.reshape(1, 1, 1, -1, 3).flip((-1,))
        out = F.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=True)
        feature = out.reshape(self.channels, -1).T.reshape(*shape, self.channels)
        return feature

    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            scale_grid = nn.Parameter(torch.rand(1, self.channels, new_world_size))
        else:
            scale_grid = nn.Parameter(
                F.interpolate(self.grid.data, size=tuple(new_world_size), mode='trilinear', align_corners=True))
        return scale_grid

    def get_dense_grid(self):
        return self.grid

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size}'


class combined_model(nn.Module):
    def __init__(
            self,
            latent_size,
            world_size,
    ):
        super(combined_model, self).__init__()

        def make_sequence():
            return []

        self.world_size = [world_size + 1, world_size + 1, world_size + 1]
        self.xyz_min = [-1, -1, -1]
        self.xyz_max = [1, 1, 1]
        self.hidden_dim = 512
        self.decoder = grid_feature(channels=int(latent_size // 2), world_size=self.world_size,
                                    xyz_min=self.xyz_min,
                                    xyz_max=self.xyz_max)

        self.local_mlp = SdfDecoder(latent_size=latent_size + latent_size // 2, hidden_dim=self.hidden_dim,
                                    skip_connection=True, tanh_act=False, ues_pe=False)

        self.global_mlp = SdfDecoder(latent_size=latent_size, hidden_dim=self.hidden_dim,
                                     skip_connection=True, tanh_act=False, ues_pe=True)

    def forward(self, latent_vecs, xyz_g, xyz_l):
        features = self.decoder.forward(xyz_l)

        x_l = torch.cat([latent_vecs, features, xyz_l], 1)
        x_l = torch.cat([latent_vecs, features, xyz_l], 1)
        x_l_sdf = self.local_mlp(x_l)

        x_g = torch.cat([latent_vecs, xyz_g], 1)
        x_g_sdf = self.global_mlp(x_g)

        return x_g_sdf, x_l_sdf, features
