#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import numpy as np
import os
import torch

import deep_sdf
import deep_sdf.workspace as ws


def code_to_mesh(experiment_directory, checkpoint, keep_normalized=False):
    specs_filename = os.path.join(experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["*"])

    latent_size = specs["CodeLength"]

    decoder = arch.ours_decoder(latent_size, world_size=128).cuda()
    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(experiment_directory, ws.model_params_subdir, checkpoint + ".pth")
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()

    decoder.eval()

    latent_vectors = ws.load_latent_vectors(experiment_directory, checkpoint)

    train_split_file = specs["TrainSplit"]

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    data_source = specs["DataSource"]
    kv = {'benches': '02828884', 'chairs': '03001627', 'planes': '02691156', 'tables': '04379243',
          'lamps': '03636649',
          'sofas': '04256520'}
    class_id = kv[split.split('_')[1]]
    with open(train_split_file, "r") as f:
        train_split = json.load(f)
        train_list = train_split['ShapeNetV2'][class_id]
    instance_filenames = train_list

    print(len(instance_filenames), " vs ", len(latent_vectors))

    for i, latent_vector in enumerate(latent_vectors):

        dataset_name, class_name, instance_name = 'ShapeNetV2', class_id, instance_filenames[i]

        print("{} {} {}".format(dataset_name, class_name, instance_name))

        mesh_dir = os.path.join(
            experiment_directory,
            ws.training_meshes_subdir,
            str(saved_model_epoch),
            dataset_name,
            class_name,
        )
        print(mesh_dir)

        if not os.path.isdir(mesh_dir):
            os.makedirs(mesh_dir)

        mesh_filename = os.path.join(mesh_dir, instance_name)

        print(instance_filenames[i])

        with torch.no_grad():
            deep_sdf.gl_mesh.create_mesh(
                decoder,
                latent_vector,
                mesh_filename,
                N=256,
                max_batch=int(2 ** 18 + 2 ** 16 + 2 ** 12 + 2 ** 10),
                offset=None,
                scale=None,
            )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to generate a mesh given a latent code."
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

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    code_to_mesh(args.experiment_directory, args.checkpoint, args.keep_normalized)
