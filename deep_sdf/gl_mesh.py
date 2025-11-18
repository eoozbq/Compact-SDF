#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import os.path
import numpy as np
import plyfile
import skimage.measure
import time
import torch
import mcubes
import deep_sdf.utils
from scipy.spatial import KDTree

_tree_cache = None


def get_tree():
    global _tree_cache
    if _tree_cache is None:
        print("Building voxel tree...")
        NN = 512
        voxel_centers = np.linspace(-1, 1, NN + 1)
        mesh = np.meshgrid(voxel_centers, voxel_centers, voxel_centers, indexing='ij')
        samples = np.vstack(mesh).reshape(3, -1).T
        _tree_cache = KDTree(samples)
        print("Voxel tree built.")
    return _tree_cache


def mesh_out(decoder, N, max_batch, latent_vec, filename, offset, scale):
    decoder.eval()
    voxel_size = 2.0 / N
    voxel_origin = [-1, -1, -1]
    tree = get_tree()
    voxel_centers = np.linspace(-1, 1, N + 1)
    mesh = np.meshgrid(voxel_centers, voxel_centers, voxel_centers, indexing='ij')
    samples = np.vstack(mesh).reshape(3, -1).T
    samples = np.concatenate([samples, np.zeros((samples.shape[0], 1))], axis=1)

    samples = torch.from_numpy(samples).to(torch.float32)
    num_samples = (N + 1) ** 3

    samples.requires_grad = False
    if 'deepsdf' in os.path.basename(filename):
        low_tag = True
    else:
        low_tag = False

    head = 0
    while head < num_samples:
        sample_subset = samples[head: min(head + max_batch, num_samples), 0:3].cuda()

        samples[head: min(head + max_batch, num_samples), 3] = (
            deep_sdf.utils.gl_sdf(decoder, latent_vec, sample_subset, low_tag=low_tag)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N + 1, N + 1, N + 1)
    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        filename,
        offset,
        scale,
    )
    return samples, sdf_values


def create_mesh(
        decoder, latent_vec, filename, N=256, max_batch=32 ** 3, offset=None, scale=None
):
    start = time.time()
    ply_filename = filename
    tree = get_tree()
    decoder.eval()
    deepsdf_out, sdf_values = mesh_out(decoder, N, max_batch, latent_vec, ply_filename + '-deepsdf' + ".ply", offset,
                                       scale)

    _, _ = mesh_out(decoder, N, max_batch, latent_vec, ply_filename + '-cube_code' + ".ply", offset, scale)

    voxel_size = 2.0 / N
    voxel_origin = [-1, -1, -1]

    num_samples = (N + 1) ** 3

    mesh_points = get_face_vector(sdf_values.data.cpu(), voxel_origin, voxel_size, offset, scale)
    grid_indices = get_grid_indices(mesh_points, voxel_size, N)
    grid_indices = expand_grids(grid_indices, N, voxel_size, expand_by=3)

    grid_samples = np.concatenate([grid_indices, np.zeros((grid_indices.shape[0], 1))], axis=1)
    grid_samples = torch.from_numpy(grid_samples).to(torch.float32)

    head = 0
    while head < grid_samples.shape[0]:
        sample_subset = grid_samples[head: min(head + max_batch, grid_samples.shape[0]), 0:3].cuda()

        grid_samples[head: min(head + max_batch, num_samples), 3] = (
            deep_sdf.utils.gl_sdf(decoder, latent_vec, sample_subset, low_tag=False)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    samples_points = deepsdf_out[:, :3]
    samples_points_sdf = deepsdf_out[:, 3]
    grid_point = grid_samples[:, :3]
    grid_point_sdf = grid_samples[:, 3]

    _, ind = tree.query(grid_point.cpu().detach().numpy())
    samples_points_sdf[ind] = grid_point_sdf

    sdf_values = samples_points_sdf.reshape(N + 1, N + 1, N + 1)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )


def convert_sdf_samples_to_ply(
        pytorch_3d_sdf_tensor,
        voxel_grid_origin,
        voxel_size,
        ply_filename_out,
        offset=None,
        scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )


def get_face_vector(
        pytorch_3d_sdf_tensor,
        voxel_grid_origin,
        voxel_size,
        offset=None,
        scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )
    np.savetxt('vectors.txt', verts)
    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]
    return mesh_points


def get_grid_indices(points, grid_size, n):
    grid_indices = np.floor((points + 1) / grid_size).astype(int)
    grid_indices = np.clip(grid_indices, 0, n)
    return grid_indices


def get_expand_grid_indices_to_point(points, grid_size, n):
    min_grid_indices = points
    offsets = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    grid_vertices = min_grid_indices[:, np.newaxis, :] + offsets
    grid_vertices = np.clip(grid_vertices, 0, n)
    grid_vertices = grid_vertices.reshape(-1, 3)
    linear_indices = np.ravel_multi_index(grid_vertices.T, (n, n, n))
    unique_indices = np.unique(linear_indices)
    expanded_indices = np.stack(np.unravel_index(unique_indices, (n, n, n)), axis=-1)
    grid_vertices_coords = -1 + grid_size * expanded_indices
    return grid_vertices_coords


def expand_grids(grid_indices, n, grid_size, expand_by=3):
    offsets = np.arange(-expand_by, expand_by + 2)
    offsets = np.array(np.meshgrid(offsets, offsets, offsets)).T.reshape(-1, 3)
    expanded_indices = grid_indices[:, np.newaxis, :] + offsets
    expanded_indices = np.clip(expanded_indices, 0, n)
    expanded = expanded_indices.reshape(-1, 3)
    linear_indices = np.ravel_multi_index(expanded.T, (n + 1, n + 1, n + 1))
    unique_indices = np.unique(linear_indices)
    expanded_indices = np.stack(np.unravel_index(unique_indices, (n + 1, n + 1, n + 1)), axis=-1)
    grid_vertices_coords = -1 + grid_size * expanded_indices
    return grid_vertices_coords
