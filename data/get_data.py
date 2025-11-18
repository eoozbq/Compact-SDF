import igl
import numpy as np
import trimesh
import os
import shutil
import json
import shutil
from concurrent.futures import ThreadPoolExecutor


def divide_space(n):
    grid_size = 2.0 / n
    return grid_size


def get_grid_indices(points, grid_size, n):
    grid_indices = np.floor((points + 1) / grid_size).astype(int)
    grid_indices = np.clip(grid_indices, 0, n)
    print('grid_indices shape: ', grid_indices.shape)
    grid_indices = np.unique(grid_indices, axis=0)
    print('grid_indices unique shape: ', grid_indices.shape)
    return grid_indices


def expand_grids(grid_indices, n, expand_by=3):
    offsets = np.arange(-expand_by, expand_by + 2)
    offsets = np.array(np.meshgrid(offsets, offsets, offsets)).T.reshape(-1, 3)
    expanded_indices = grid_indices[:, np.newaxis, :] + offsets
    expanded_indices = np.clip(expanded_indices, 0, n)
    expanded = expanded_indices.reshape(-1, 3)
    linear_indices = np.ravel_multi_index(expanded.T, (n + 1, n + 1, n + 1))
    unique_indices = np.unique(linear_indices)
    expanded_indices = np.stack(np.unravel_index(unique_indices, (n + 1, n + 1, n + 1)), axis=-1)
    return expanded_indices


def read_obj_vertices(file_path):
    mesh = trimesh.load(file_path)
    vertices = mesh.vertices
    return vertices


def get_bandwidth_sample(file_path, outpath, file_name):
    n = 512
    expand_by = 6
    grid_size = divide_space(n)
    mesh_vertices = read_obj_vertices(file_path)
    grid_indices = get_grid_indices(mesh_vertices, grid_size, n)

    covered_grids = expand_grids(grid_indices, n, expand_by=expand_by)
    samples_surface_grid = -1 + grid_size * covered_grids
    print(file_name + ' samples_surface_grid shape', samples_surface_grid.shape)
    voxel_centers = np.linspace(-1, 1, n // 4 + 1)
    mesh = np.meshgrid(voxel_centers, voxel_centers, voxel_centers, indexing='ij')
    samples_grid = np.vstack(mesh).reshape(3, -1).T
    print(file_name + ' samples_grid shape', samples_grid.shape)

    samples = np.concatenate([samples_surface_grid, samples_grid], axis=0)
    samples = np.unique(samples, axis=0)

    return samples


def get_sdf_sample(file_path, out_path, file_name, samples):
    mesh = trimesh.load(file_path, force='mesh')
    V = mesh.vertices
    F = mesh.faces
    S, I, C = igl.signed_distance(samples, V, F, return_normals=False)

    surface_sigma = True
    if surface_sigma:
        num = 360000
        points = np.random.rand(num // 2, 3) * 2 - 1
        points1 = trimesh.sample.sample_surface(mesh, count=num)[0]
        offset2 = np.random.normal(scale=0.025, size=(num, 3))
        near_samples2 = points1 + offset2
        near_samples_0 = np.concatenate([near_samples2, points], axis=0)
        near_samples = np.unique(near_samples_0, axis=0)
        samples_sigma = np.clip(near_samples, -1, 1)

        S_surface_sigma, I_surface_sigma, C_surface_sigma = igl.signed_distance(samples_sigma, V, F,
                                                                                return_normals=False)

    current_dir = os.getcwd()
    out_temp = os.path.join(current_dir, 'temp')
    if not os.path.exists(out_temp):
        os.makedirs(out_temp)
    print(out_temp)
    out_tt = file_name.split('.')[0]
    print(out_tt)
    out_sdf = os.path.join(out_path, out_tt)
    out_temp_file = os.path.join(out_temp, out_tt)
    if not os.path.exists(out_sdf):
        os.makedirs(out_sdf)
        os.makedirs(out_temp_file)

    try:
        org_out_name = os.path.join(out_temp_file, out_tt + '.npy')
        tar_out_name = os.path.join(out_sdf, out_tt + '.npy')
        np.save(org_out_name, np.concatenate((samples, S.reshape(-1, 1)), axis=1))
        if surface_sigma:
            org_out_surface_sigma_name = os.path.join(out_temp_file, out_tt + '_g.npy')
            tar_out_surface_sigma_name = os.path.join(out_sdf, out_tt + '_g.npy')
            np.save(org_out_surface_sigma_name, np.concatenate((samples_sigma, S_surface_sigma.reshape(-1, 1)), axis=1))

        np.savetxt(org_out_name.replace('.npy', '.txt'), np.concatenate((samples, S.reshape(-1, 1)), axis=1))
        if os.path.exists(org_out_name):
            shutil.move(org_out_name, tar_out_name)
            shutil.move(org_out_surface_sigma_name, tar_out_surface_sigma_name)
            print(f"Successfully moved file to {tar_out_name}")
        else:
            print(f"File {org_out_name} does not exist after saving, unable to move.")
    except Exception as e:
        print(f"Error occurred: {e}")

    shutil.rmtree(out_temp)


file_path = 'stanford_mesh'
out_path = r'stanford'
if not os.path.exists(out_path):
    os.makedirs(out_path)

train_list = ['Armadillo.obj', ]


def process_file(file_name):
    file_name = file_name
    obj_path = os.path.join(file_path, file_name)
    print(obj_path)
    obj_samples = get_bandwidth_sample(obj_path, out_path, file_name)
    if obj_samples is not None:
        try:
            get_sdf_sample(obj_path, out_path, file_name, obj_samples)
        except Exception as e:
            print(f"Error occurred: {e}")
    print(file_name)


with ThreadPoolExecutor(max_workers=1) as executor:
    executor.map(process_file, train_list)
