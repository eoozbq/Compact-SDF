#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import json
import os
import torch

model_params_subdir = "ModelParameters"
optimizer_params_subdir = "OptimizerParameters"
latent_codes_subdir = "LatentCodes"
logs_filename = "Logs.pth"
reconstructions_subdir = "Reconstructions"
reconstruction_meshes_subdir = "Meshes"
reconstruction_codes_subdir = "Codes"
specifications_filename = "specs.json"

evaluation_subdir = "Evaluation"
training_meshes_subdir = "TrainingMeshes"


def load_experiment_specifications(experiment_directory):
    filename = os.path.join(experiment_directory, specifications_filename)

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            + '"specs.json"'.format(experiment_directory)
        )

    return json.load(open(filename))


def load_model_parameters(experiment_directory, checkpoint, decoder):
    filename = os.path.join(
        experiment_directory, model_params_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))

    data = torch.load(filename)
    model_state = decoder.state_dict()
    checkpoint = data["model_state_dict"]
    if 'state_dict' in checkpoint:
        pth_state = checkpoint['state_dict']
    elif 'model' in checkpoint:
        pth_state = checkpoint['model']
    else:
        pth_state = checkpoint
    print(f"🔍 Model param count: {sum(p.numel() for p in model_state.values())}")
    print(f"📂 PTH param count: {sum(p.numel() for p in pth_state.values())}")
    print(f"🔑 Model keys: {len(model_state.keys())}, PTH keys: {len(pth_state.keys())}")

    model_keys = set(model_state.keys())
    pth_keys = set(pth_state.keys())

    missing_in_pth = model_keys - pth_keys
    extra_in_pth = pth_keys - model_keys

    if len(missing_in_pth) == 0 and len(extra_in_pth) == 0:
        print("\n✅ Model and .pth parameter names completely match!\n")
    else:
        if len(missing_in_pth) > 0:
            print("\n⚠️ Parameters in MODEL but missing in PTH:")
            for key in missing_in_pth:
                print(f" - {key}, shape: {model_state[key].shape}, numel: {model_state[key].numel()}")

        if len(extra_in_pth) > 0:
            print("\n⚠️ Extra parameters in PTH not in MODEL:")
            total_extra = 0
            for key in extra_in_pth:
                n = pth_state[key].numel()
                total_extra += n
                print(f" - {key}, shape: {pth_state[key].shape}, numel: {n}")
            print(f"\n✨ Total extra parameters in pth: {total_extra}\n")

    for key in model_keys & pth_keys:
        if model_state[key].shape != pth_state[key].shape:
            print(f"❌ Shape mismatch at {key}: Model {model_state[key].shape}, Pth {pth_state[key].shape}")

    decoder.load_state_dict(data["model_state_dict"], strict=True)
    return data["epoch"]


def build_decoder(experiment_directory, experiment_specs):
    arch = __import__(
        "networks." + experiment_specs["NetworkArch"], fromlist=["Decoder"]
    )

    latent_size = experiment_specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **experiment_specs["NetworkSpecs"]).cuda()

    return decoder


def load_decoder(
        experiment_directory, experiment_specs, checkpoint, data_parallel=True
):
    decoder = build_decoder(experiment_directory, experiment_specs)

    if data_parallel:
        decoder = torch.nn.DataParallel(decoder)

    epoch = load_model_parameters(experiment_directory, checkpoint, decoder)

    return (decoder, epoch)


def load_latent_vectors(experiment_directory, checkpoint):
    filename = os.path.join(
        experiment_directory, latent_codes_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include a latent code file"
            + " for checkpoint '{}'".format(experiment_directory, checkpoint)
        )

    data = torch.load(filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        num_vecs = data["latent_codes"].size()[0]

        lat_vecs = []
        for i in range(num_vecs):
            lat_vecs.append(data["latent_codes"][i].cuda())

        return lat_vecs

    else:

        num_embeddings, embedding_dim = data["latent_codes"]["weight"].shape

        lat_vecs = torch.nn.Embedding(num_embeddings, embedding_dim)

        lat_vecs.load_state_dict(data["latent_codes"])

        return lat_vecs.weight.data.detach()


def get_reconstructed_mesh_filename(
        experiment_dir, epoch, dataset, class_name, instance_name
):
    return os.path.join(
        experiment_dir,
        reconstructions_subdir,
        str(epoch),
        reconstruction_meshes_subdir,
        dataset,
        class_name,
        instance_name + ".ply",
    )


def get_training_mesh_filename(
        experiment_dir, epoch, dataset, class_name, instance_name
):
    return os.path.join(
        experiment_dir,
        training_meshes_subdir,
        str(epoch),
        dataset,
        class_name,
        instance_name + ".ply",
    )


def get_reconstructed_code_filename(
        experiment_dir, epoch, dataset, class_name, instance_name
):
    return os.path.join(
        experiment_dir,
        reconstructions_subdir,
        str(epoch),
        reconstruction_codes_subdir,
        dataset,
        class_name,
        instance_name + ".pth",
    )


def get_evaluation_dir(experiment_dir, checkpoint, create_if_nonexistent=False):
    dir = os.path.join(experiment_dir, evaluation_subdir, checkpoint)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_model_params_dir(experiment_dir, create_if_nonexistent=False):
    dir = os.path.join(experiment_dir, model_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_optimizer_params_dir(experiment_dir, create_if_nonexistent=False):
    dir = os.path.join(experiment_dir, optimizer_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_latent_codes_dir(experiment_dir, create_if_nonexistent=False):
    dir = os.path.join(experiment_dir, latent_codes_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_normalization_params_filename(
        data_dir, dataset_name, class_name, instance_name
):
    return os.path.join(
        data_dir,
        normalization_param_subdir,
        dataset_name,
        class_name,
        instance_name + ".npz",
    )
