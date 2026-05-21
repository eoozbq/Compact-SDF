#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import json
import random
import torch
import torch.utils.data
import deep_sdf.workspace as ws


class shapenet_data(torch.utils.data.Dataset):
    def __init__(
            self,
            data_source,
            split,
            subsample,
    ):
        self.subsample = subsample
        self.npypath = data_source
        self.kv = {'benches': '02828884', 'chairs': '03001627', 'planes': '02691156', 'tables': '04379243',
                   'lamps': '03636649',
                   'sofas': '04256520'}
        class_id = self.kv[split.split('_')[1]]
        with open(split, "r") as f:
            train_split = json.load(f)
            train_list = train_split['ShapeNetV2'][class_id]
        self.npyfiles = train_list

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

    def __len__(self):
        return len(self.npyfiles)

    def get_sample(self, pos_tensor, neg_tensor, surface_points):

        half = int(self.subsample / 2)
        quater = int(self.subsample / 4)
        surface = (torch.rand(half) * surface_points.shape[0]).long().cuda()
        surface_random = torch.index_select(surface_points, 0, surface)
        pos = quater if pos_tensor.shape[0] > quater else pos_tensor.shape[0]
        neg = quater if neg_tensor.shape[0] > quater else neg_tensor.shape[0]
        random_pos = (torch.rand(pos) * pos_tensor.shape[0]).long().cuda()
        random_neg = (torch.rand(neg) * neg_tensor.shape[0]).long().cuda()
        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
        if sample_pos.shape[0] + sample_neg.shape[0] + surface_random.shape[0] == self.subsample:
            samples = torch.cat([sample_pos, sample_neg, surface_random], 0).clone().detach().to(torch.float32)
        else:
            num_points = self.subsample - (sample_pos.shape[0] + sample_neg.shape[0] + surface_random.shape[0])
            sel = (torch.rand(num_points) * surface_points.shape[0]).long().cuda()
            sel_random = torch.index_select(surface_points, 0, sel)
            samples = torch.cat([sample_pos, sample_neg, surface_random, sel_random], 0).clone().detach().to(
                torch.float32)
        return samples.to(torch.float32)

    def get_key_value(self, file_path):

        data = np.load(file_path).astype(np.float32)
        points = torch.from_numpy(data).cuda()
        sdf_values = points[..., -1]
        pos_idx = sdf_values > 0
        neg_idx = sdf_values < 0
        surface_mask = (sdf_values >= -0.1) & (sdf_values <= 0.1)

        pos_points = points[pos_idx]
        neg_points = points[neg_idx]
        surface_points = points[surface_mask]

        return {'pos': pos_points, 'neg': neg_points, 'surface': surface_points}

    def __getitem__(self, idx):
        file = self.npyfiles[idx]
        file_path = os.path.join(self.npypath, file, file + '.npy')
        cube_data = self.get_key_value(file_path)

        samples_cube_points = self.get_sample(cube_data['pos'], cube_data['neg'],
                                              cube_data['surface'])
        g_file_path = os.path.join(self.npypath, file, file + '_g.npy')
        global_data = self.get_key_value(g_file_path)
        samples_global_points = self.get_sample(global_data['pos'], global_data['neg'],
                                                global_data['surface'])

        return samples_global_points.cpu(), samples_cube_points.cpu(), idx


class dfaust_data(torch.utils.data.Dataset):
    def __init__(
            self,
            data_source,
            split,
            subsample,
    ):
        self.subsample = subsample

        self.npypath = data_source
        with open(split, "r") as f:
            train_split = f.readlines()
            train_list = []
            for line in train_split:
                parts = line.strip().rsplit('_', 1)
                prefix = parts[0]
                number = int(parts[1])
                number_str = f"{number:05d}"
                path = f"{self.npypath}/{prefix}/{number_str}"
                train_list.append(path)

        self.npyfiles = train_list

        logging.debug(
            "using "
            + str(len(self.npyfiles) // 3)
            + " shapes from data source "
            + data_source
        )

    def __len__(self):
        return len(self.npyfiles)

    def get_sample(self, pos_tensor, neg_tensor, surface_points):

        half = int(self.subsample / 2)
        quater = int(self.subsample / 4)
        surface = (torch.rand(half) * surface_points.shape[0]).long().cuda()
        surface_random = torch.index_select(surface_points, 0, surface)
        pos = quater if pos_tensor.shape[0] > quater else pos_tensor.shape[0]
        neg = quater if neg_tensor.shape[0] > quater else neg_tensor.shape[0]
        random_pos = (torch.rand(pos) * pos_tensor.shape[0]).long().cuda()
        random_neg = (torch.rand(neg) * neg_tensor.shape[0]).long().cuda()
        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
        if sample_pos.shape[0] + sample_neg.shape[0] + surface_random.shape[0] == self.subsample:
            samples = torch.cat([sample_pos, sample_neg, surface_random], 0).clone().detach().to(torch.float32)
        else:
            num_points = self.subsample - (sample_pos.shape[0] + sample_neg.shape[0] + surface_random.shape[0])
            sel = (torch.rand(num_points) * surface_points.shape[0]).long().cuda()
            sel_random = torch.index_select(surface_points, 0, sel)
            samples = torch.cat([sample_pos, sample_neg, surface_random, sel_random], 0).clone().detach().to(
                torch.float32)
        return samples.to(torch.float32)

    def get_key_value(self, file_path):

        data = np.load(file_path).astype(np.float32)
        points = torch.from_numpy(data).cuda()

        sdf_values = points[..., -1]

        pos_idx = sdf_values > 0
        neg_idx = sdf_values < 0
        surface_mask = (sdf_values >= -0.1) & (sdf_values <= 0.1)

        pos_points = points[pos_idx]
        neg_points = points[neg_idx]
        surface_points = points[surface_mask]

        return {'pos': pos_points, 'neg': neg_points, 'surface': surface_points}

    def __getitem__(self, idx):
        file = self.npyfiles[idx]
        filename = os.path.basename(file)
        file_path = os.path.join(file, filename + '.npy')
        cube_data = self.get_key_value(file_path)

        samples_cube_points = self.get_sample(cube_data['pos'], cube_data['neg'],
                                              cube_data['surface'])
        g_file_path = os.path.join(file, filename + '_g.npy')
        global_data = self.get_key_value(g_file_path)
        samples_global_points = self.get_sample(global_data['pos'], global_data['neg'],
                                                global_data['surface'])

        return samples_global_points.cpu(), samples_cube_points.cpu(), idx


class stanford_data(torch.utils.data.Dataset):
    def __init__(
            self,
            data_source,
            subsample,
            cn='dragon',
    ):
        self.subsample = subsample

        self.npypath = os.path.join(data_source, cn)
        self.npyfiles = [self.npypath]
        self.cube_data = self.get_key_value(os.path.join(self.npypath, f'{cn}.npy'))
        self.global_data = self.get_key_value(os.path.join(self.npypath, f'{cn}_g.npy'))

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

    def __len__(self):
        return len(self.npyfiles)

    def get_sample(self, pos_tensor, neg_tensor, surface_points):

        half = int(self.subsample / 2)
        quater = int(self.subsample / 4)
        surface = (torch.rand(half) * surface_points.shape[0]).long().cuda()
        surface_random = torch.index_select(surface_points, 0, surface)
        pos = quater if pos_tensor.shape[0] > quater else pos_tensor.shape[0]
        neg = quater if neg_tensor.shape[0] > quater else neg_tensor.shape[0]
        random_pos = (torch.rand(pos) * pos_tensor.shape[0]).long().cuda()
        random_neg = (torch.rand(neg) * neg_tensor.shape[0]).long().cuda()
        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
        if sample_pos.shape[0] + sample_neg.shape[0] + surface_random.shape[0] == self.subsample:
            samples = torch.cat([sample_pos, sample_neg, surface_random], 0).clone().detach().to(torch.float32)
        else:
            num_points = self.subsample - (sample_pos.shape[0] + sample_neg.shape[0] + surface_random.shape[0])
            sel = (torch.rand(num_points) * surface_points.shape[0]).long().cuda()
            sel_random = torch.index_select(surface_points, 0, sel)
            samples = torch.cat([sample_pos, sample_neg, surface_random, sel_random], 0).clone().detach().to(
                torch.float32)
        return samples.to(torch.float32)

    def get_key_value(self, file_path):

        data = np.load(file_path).astype(np.float32)
        points = torch.from_numpy(data).cuda()

        sdf_values = points[..., -1]

        pos_idx = sdf_values > 0
        neg_idx = sdf_values < 0
        surface_mask = (sdf_values >= -0.1) & (sdf_values <= 0.1)

        pos_points = points[pos_idx]
        neg_points = points[neg_idx]
        surface_points = points[surface_mask]

        return {'pos': pos_points, 'neg': neg_points, 'surface': surface_points}

    def __getitem__(self, idx):

        samples_cube_points = self.get_sample(self.cube_data['pos'], self.cube_data['neg'],
                                              self.cube_data['surface'])
        samples_global_points = self.get_sample(self.global_data['pos'], self.global_data['neg'],
                                                self.global_data['surface'])
        return samples_global_points.cpu(), samples_cube_points.cpu(), idx



# reconstruction data

def read_sdf_samples_from_gl_data(filename):
    data = np.load(filename).astype(np.float32)
    points = torch.from_numpy(data).cuda()
    sdf_values = points[..., -1]
    pos_idx = sdf_values > 0
    neg_idx = sdf_values < 0
    surface_mask = (sdf_values >= -0.1) & (sdf_values <= 0.1)

    pos_points = points[pos_idx]
    neg_points = points[neg_idx]
    surface_points = points[surface_mask]
    half = True
    if half:
        axis = 2

        mask = pos_points[:, axis] > 0
        pos_points = pos_points[mask]
        mask = neg_points[:, axis] > 0
        neg_points = neg_points[mask]
        mask = surface_points[:, axis] > 0
        surface_points = surface_points[mask]

    return [pos_points, neg_points, surface_points]


def unpack_sdf_samples_from_gl_data(data, subsample=None):
    half = int(subsample / 2)
    quater = int(subsample / 4)
    surface_points = data[2]
    pos_tensor = data[0]
    neg_tensor = data[1]
    surface = (torch.rand(half) * surface_points.shape[0]).long().cuda()
    surface_random = torch.index_select(surface_points, 0, surface)
    pos = quater if pos_tensor.shape[0] > quater else pos_tensor.shape[0]
    neg = quater if neg_tensor.shape[0] > quater else neg_tensor.shape[0]
    random_pos = (torch.rand(pos) * pos_tensor.shape[0]).long().cuda()
    random_neg = (torch.rand(neg) * neg_tensor.shape[0]).long().cuda()
    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    if sample_pos.shape[0] + sample_neg.shape[0] + surface_random.shape[0] == subsample:
        samples = torch.cat([sample_pos, sample_neg, surface_random], 0).clone().detach().to(torch.float32)
    else:
        num_points = subsample - (sample_pos.shape[0] + sample_neg.shape[0] + surface_random.shape[0])
        sel = (torch.rand(num_points) * surface_points.shape[0]).long().cuda()
        sel_random = torch.index_select(surface_points, 0, sel)
        samples = torch.cat([sample_pos, sample_neg, surface_random, sel_random], 0).clone().detach().to(
            torch.float32)
    return samples.to(torch.float32)