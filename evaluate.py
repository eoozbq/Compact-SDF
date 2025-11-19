#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import logging
import json
import numpy as np
import os
import trimesh
from multiprocessing import Pool
import deep_sdf
import deep_sdf.workspace as ws


def evaluate(experiment_directory, checkpoint, gt_objpath='', gen_mesh=None):
    chamfer_results = []
    num_mesh_samples = 100000
    instance_name = gen_mesh
    logging.debug(
        "evaluating " + os.path.join(instance_name)
    )

    reconstructed_mesh_filename = os.path.join(recon_mesh_path, instance_name)
    logging.debug(
        'reconstructed mesh is "' + reconstructed_mesh_filename + '"'
    )

    ground_truth_samples_filename = os.path.join(
        gt_objpath, instance_name.split('-')[0] + '.obj')

    logging.debug(
        "ground truth samples are " + ground_truth_samples_filename
    )
    ground_truth_points = trimesh.load(ground_truth_samples_filename)

    reconstruction = trimesh.load(reconstructed_mesh_filename)
    sample_points_path = os.path.join(
        ws.get_evaluation_dir(experiment_directory, checkpoint, True), instance_name
    )
    print(sample_points_path)
    chamfer_gen_dist = deep_sdf.metrics.chamfer.compute_trimesh_chamfer(
        ground_truth_points,
        reconstruction,
        0,
        1,
        num_mesh_samples=num_mesh_samples,
        name=sample_points_path
    )

    reconstruction = trimesh.load(reconstructed_mesh_filename.replace('-gen', '-cube_code'))
    chamfer_cubesdf_dist = deep_sdf.metrics.chamfer.compute_trimesh_chamfer(
        ground_truth_points,
        reconstruction,
        0,
        1,
        num_mesh_samples=num_mesh_samples,
        name=sample_points_path
    )

    reconstruction = trimesh.load(reconstructed_mesh_filename.replace('-gen.ply', '.ply'))
    chamfer_combine_dist = deep_sdf.metrics.chamfer.compute_trimesh_chamfer(
        ground_truth_points,
        reconstruction,
        0,
        1,
        num_mesh_samples=num_mesh_samples,
        name=sample_points_path
    )

    chamfer_results.append(
        (
            os.path.join(instance_name.split('-')[0]), chamfer_gen_dist, chamfer_cubesdf_dist,
            chamfer_combine_dist)
    )
    print(chamfer_gen_dist, chamfer_cubesdf_dist, chamfer_combine_dist, instance_name)

    return chamfer_results, ws.get_evaluation_dir(experiment_directory, checkpoint, True)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Evaluate a ours autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include experiment specifications in "
             + '"specs.json", and logging will be done in this directory as well.',
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint to test.",
    )

    arg_parser.add_argument(
        "--recon_mesh_path",
        "-recon_path",
        dest="recon_mesh_path",
    )

    arg_parser.add_argument(
        "--gt_mesh_path",
        "-gt_path",
        dest="gt_mesh_path",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    recon_mesh_path = args.recon_mesh_path

    recon_mesh_all = os.listdir(recon_mesh_path)

    gt_objpath = args.gt_mesh_path

    recon_mesh = []
    for mesh in recon_mesh_all:
        if 'gen' and 'cube' not in mesh:
            recon_mesh.append(mesh)

    def parallel_evaluate(recon_mesh):
        chamfer_results, cd_path = evaluate(
            args.experiment_directory,
            args.checkpoint,
            gt_objpath=gt_objpath,
            gen_mesh=recon_mesh
        )
        return chamfer_results, cd_path


    with Pool(processes=100) as p:
        chamfer_results = p.map(parallel_evaluate, gen_mesh)

    cd_path = chamfer_results[0][1]
    with open(
            os.path.join(
                cd_path, "chamfer.csv"
            ),
            "w",
    ) as f:
        f.write("shape, chamfer_dist\n")
        for result in chamfer_results:
            result = result[0][0]
            f.write("{}, {}, {}, {}\n".format(result[0], result[1], result[2], result[3]))
            print(result[0], result[1], result[2], result[3])
