# Project RoboSplatter
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


import argparse
import ast
import logging
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from tqdm import tqdm
from robo_splatter.models.basic import RenderConfig
from robo_splatter.models.camera import Camera
from robo_splatter.models.gaussians import RigidsGaussians, VanillaGaussians
from robo_splatter.render.scenes import (
    RenderCoordSystem,
    RenderResult,
    Scene,
    SceneRenderType,
)
from robo_splatter.utils.helper import create_mp4_from_images

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Render GS scene")
    parser.add_argument(
        "--data_file",
        default="robo_splatter/config/gs_data_basic.yaml",
        help="Path to GS data config file",
        type=str,
    )
    parser.add_argument(
        "--camera_extrinsic",
        type=str,
        required=True,
        help="String of camera poses lists, e.g., '[[x, y, z, qx, qy, qz, qw], ...]'",  # noqa
    )
    parser.add_argument(
        "--camera_intrinsic",
        type=str,
        required=True,
        help="Camera intrinsic matrix",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        required=True,
        help="Height of the image, e.g., 480",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        required=True,
        help="Width of the image, e.g., 640",
    )
    parser.add_argument(
        "--instances_poses",
        type=str,
        default=None,
        help="String of instances poses lists, e.g., '[[id, x, y, z, qx, qy, qz, qw], ...]'",  # noqa
    )
    parser.add_argument(
        "--device", default="cuda", help="Device to run the code", type=str
    )
    parser.add_argument(
        "--output_dir", default="./output", help="Render Output dir", type=str
    )
    parser.add_argument(
        "--gen_mp4_path",
        type=str,
        default=None,
        help="Output path of the generated mp4 video",
    )
    parser.add_argument(
        "--coord_system",
        type=str,
        default="GAUSSIAN",
        help="see `RenderCoordSystem`, enumerated in `MUJOCO`, `GAUSSIAN`, `ISAAC`",  # noqa
    )
    parser.add_argument(
        "--scene_type",
        type=str,
        default="SCENE",
        help="see `SceneRenderType`, enumerated in `SCENE`, `FOREGROUND`, `BACKGROUND`",  # noqa
    )

    return parser.parse_args()


def entrypoint() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    bg = VanillaGaussians.init_from_yaml(args.data_file)
    fg = RigidsGaussians.init_from_yaml(args.data_file)
    scene = Scene(
        render_config=RenderConfig(),
        background_models=bg,
        foreground_models=fg,
        device=args.device,
    )

    camera_intrinsic = np.array(ast.literal_eval(args.camera_intrinsic))
    if args.camera_extrinsic.endswith(".txt"):
        camera_extrinsic = np.loadtxt(
            args.camera_extrinsic, delimiter=",", dtype=float
        )
    else:
        camera_extrinsic = np.array(ast.literal_eval(args.camera_extrinsic))

    n_frame = len(camera_extrinsic)
    if args.instances_poses is None:
        instances_poses = None
    else:
        if args.instances_poses.endswith(".txt"):
            data = np.loadtxt(args.instances_poses, delimiter=",", dtype=float)
        else:
            data = np.array(ast.literal_eval(args.instances_poses))

        instance_ids = data[:, 0].astype(int)
        poses = data[:, 1:]
        unique_ids, indices = np.unique(instance_ids, return_inverse=True)
        pose_group = {
            uid: poses[indices == i] for i, uid in enumerate(unique_ids)
        }
        instances_poses = defaultdict(dict)
        for uid, pose in pose_group.items():
            assert (
                len(pose) == n_frame
            ), f"Camera instances Pose length mismatch."
            for fid in range(n_frame):
                instances_poses[fid][uid] = torch.Tensor(pose[fid])

    images_cache = []
    depth_global_min, depth_global_max = float("inf"), -float("inf")
    for idx in tqdm(range(n_frame), desc="Rendering"):
        camera = Camera.init_from_pose_list(
            pose_list=camera_extrinsic[idx],
            camera_intrinsic=camera_intrinsic,
            image_height=args.image_height,
            image_width=args.image_width,
            device=args.device,
        )

        render_result: RenderResult = scene.render(
            camera,
            instances_poses[idx] if instances_poses is not None else None,
            render_type=args.scene_type,
            coord_system=args.coord_system,
        )
        for rgb, depth in zip(render_result.rgb, render_result.depth):
            images_cache.append([rgb, depth])
            depth_global_min = min(depth_global_min, depth.min())
            depth_global_max = max(depth_global_max, depth.max())

    images = []
    for idx, (rgb, depth) in enumerate(images_cache):
        depth_normalized = np.clip(
            (depth - depth_global_min) / (depth_global_max - depth_global_min),
            0,
            1,
        )
        depth_normalized = (depth_normalized * 255).astype(np.uint8)
        depth_map = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        image = np.concatenate([rgb, depth_map], axis=1)
        output_path = f"{args.output_dir}/image_{idx:04d}.png"
        cv2.imwrite(output_path, image)
        images.append(image)

    if args.gen_mp4_path is not None:
        os.makedirs(os.path.dirname(args.gen_mp4_path), exist_ok=True)
        create_mp4_from_images(
            images,
            args.gen_mp4_path,
            to_uint8=False,
            fps=10,
            cvt_mode=cv2.COLOR_BGR2RGB,
        )

    logger.info(f"Render GS scene successfully in {args.output_dir}")


if __name__ == "__main__":
    entrypoint()
