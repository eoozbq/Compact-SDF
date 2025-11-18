#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch

import deep_sdf
import deep_sdf.workspace as ws


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def reconstruct(
        decoder,
        num_iterations,
        latent_size,
        data_sdf_deepsdf,
        data_sdf_cubecode,
        npz,
        stat,
        clamp_dist,
        num_samples=30000,
        lr=5e-4,
        l2reg=False,
):
    def adjust_learning_rate(
            initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()

    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss()

    for e in range(num_iterations):

        decoder.eval()
        global_data = deep_sdf.gl_data.unpack_sdf_samples_from_gl_data(
            data_sdf_deepsdf, num_samples
        ).cuda()
        cube_data = deep_sdf.gl_data.unpack_sdf_samples_from_gl_data(
            data_sdf_cubecode, num_samples
        ).cuda()
        save_base_name = '_'.join(npz.split("/")[-2:])
        outtxt = f'{os.path.basename(args.experiment_directory)}/Reconstructions/2000/{save_base_name}_sample.txt'

        if not os.path.exists(outtxt):
            sdf_save_data = cube_data.cpu().detach().numpy()
            if not os.path.exists(outtxt):
                np.savetxt(outtxt, sdf_save_data)
        xyz_g = global_data[:, 0:3]
        xyz_l = cube_data[:, 0:3]
        sdf_g_gt = global_data[:, 3].unsqueeze(1)
        sdf_l_gt = cube_data[:, 3].unsqueeze(1)

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)

        pred_g_sdf, pred_l_sdf, features = decoder(latent_inputs.cuda(), xyz_g.cuda(), xyz_l.cuda())

        sdf_g_loss = loss_l1(pred_g_sdf, sdf_g_gt.cuda())
        sdf_l_loss = loss_l1(pred_l_sdf, sdf_l_gt.cuda())

        loss = sdf_g_loss + sdf_l_loss

        if l2reg:
            loss += 1e-4 * torch.mean(latent.pow(2))
        loss.backward()
        print(loss.item())
        optimizer.step()

        if e % 50 == 0:
            logging.debug(loss.cpu().data.numpy())
            logging.debug(e)
            logging.debug(latent.norm())
        loss_num = loss.cpu().data.numpy()

    return loss_num, latent


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
                    + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
             + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
             + "or 'latest' for the latest weights (this is the default)",
    )

    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=2000,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)


    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var


    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["*"])

    latent_size = specs["CodeLength"]

    decoder = arch.gl_decoder(latent_size, world_size=128).cuda()
    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()

    npypath = r'data/watertight_f_and_m_hdf5_train_50025'
    with open(r'examples/dfaust_split/split_train_val/train/50025.txt', "r") as f:
        train_split = f.readlines()
        train_list = []

        for line in train_split:
            parts = line.strip().rsplit('_', 1)
            prefix = parts[0]
            number = int(parts[1])
            number_str = f"{number:05d}"
            path = f"{npypath}/{prefix}/{number_str}"
            train_list.append(path)

    npz_filenames = train_list

    random.shuffle(npz_filenames)

    logging.debug(decoder)

    err_sum = 0.0
    repeat = 1
    save_latvec_only = False
    rerun = 0

    reconstruction_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, str(saved_model_epoch)
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)

    for ii, npz in enumerate(npz_filenames):

        filename = os.path.basename(npz)

        full_filename = os.path.join(npz, filename + '_g.npy')
        logging.debug("loading {}".format(npz))

        data_sdf_deepsdf = deep_sdf.data.read_sdf_samples_from_gl_data(full_filename)
        data_sdf_cubecode = deep_sdf.data.read_sdf_samples_from_gl_data(os.path.join(npz, filename + '.npy'))

        for k in range(repeat):

            if rerun > 1:
                mesh_filename = os.path.join(
                    reconstruction_meshes_dir, npz[:-4] + "-" + str(k + rerun)
                )
                latent_filename = os.path.join(
                    reconstruction_codes_dir, npz[:-4] + "-" + str(k + rerun) + ".pth"
                )
            else:
                parts = npz.strip().rsplit('/', 2)
                mesh_name = '_'.join(parts[-2:])
                mesh_filename = os.path.join(reconstruction_meshes_dir, mesh_name)
                latent_filename = os.path.join(
                    reconstruction_codes_dir, npz + ".pth"
                )

            if (
                    args.skip
                    and os.path.isfile(mesh_filename + ".ply")
                    and os.path.isfile(latent_filename)
            ):
                print('skip', mesh_filename + ".ply")
                continue

            logging.info("reconstructing {}".format(npz))

            data_sdf_deepsdf[0] = data_sdf_deepsdf[0][torch.randperm(data_sdf_deepsdf[0].shape[0])]
            data_sdf_deepsdf[1] = data_sdf_deepsdf[1][torch.randperm(data_sdf_deepsdf[1].shape[0])]
            data_sdf_deepsdf[2] = data_sdf_deepsdf[2][torch.randperm(data_sdf_deepsdf[2].shape[0])]

            data_sdf_cubecode[0] = data_sdf_cubecode[0][torch.randperm(data_sdf_cubecode[0].shape[0])]
            data_sdf_cubecode[1] = data_sdf_cubecode[1][torch.randperm(data_sdf_cubecode[1].shape[0])]
            data_sdf_cubecode[2] = data_sdf_cubecode[2][torch.randperm(data_sdf_cubecode[2].shape[0])]

            start = time.time()
            err, latent = reconstruct(
                decoder,
                int(args.iterations),
                latent_size,
                data_sdf_deepsdf,
                data_sdf_cubecode,
                npz,
                0.01,  # [emp_mean,emp_var],
                0.1,
                num_samples=4000,
                lr=5e-3,
                l2reg=True,
            )
            logging.debug("reconstruct time: {}".format(time.time() - start))
            err_sum += err
            logging.debug("current_error avg: {}".format((err_sum / (ii + 1))))
            logging.debug(ii)

            logging.debug("latent: {}".format(latent.detach().cpu().numpy()))

            decoder.eval()

            if not os.path.exists(os.path.dirname(mesh_filename)):
                os.makedirs(os.path.dirname(mesh_filename))
            print(mesh_filename)
            if not save_latvec_only:
                start = time.time()
                with torch.no_grad():
                    deep_sdf.gl_mesh.create_mesh(
                        decoder, latent, mesh_filename, N=256, max_batch=int(2 ** 18)
                    )
                logging.debug("total time: {}".format(time.time() - start))

            if not os.path.exists(os.path.dirname(latent_filename)):
                os.makedirs(os.path.dirname(latent_filename))

            torch.save(latent.unsqueeze(0), latent_filename)
