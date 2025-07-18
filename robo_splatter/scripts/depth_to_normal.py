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


# TODO: import from gsplat instead of copying the code.

import cv2
import numpy as np
import torch
from einops import rearrange


def ndc_2_cam(ndc_xyz, intrinsic, W, H):
    inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
    cam_z = ndc_xyz[..., 2:3]
    cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
    cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
    cam_xyz = cam_xyz @ torch.inverse(intrinsic[0, ...].t())

    return cam_xyz


def depth2point_cam(sampled_depth, ref_intrinsic):
    B, N, C, H, W = sampled_depth.shape
    valid_z = sampled_depth
    valid_x = torch.arange(
        W, dtype=torch.float32, device=sampled_depth.device
    ) / (W - 1)
    valid_y = torch.arange(
        H, dtype=torch.float32, device=sampled_depth.device
    ) / (H - 1)
    valid_y, valid_x = torch.meshgrid(valid_y, valid_x)
    # B,N,H,W
    valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
    valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
    ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(
        B, N, C, H, W, 3
    )  # 1, 1, 5, 512, 640, 3
    cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H)  # 1, 1, 5, 512, 640, 3

    return ndc_xyz, cam_xyz


def depth2normal(
    intrinsic_matrix, extrinsic_matrix, depth, offset=None, scale=1
):
    if len(depth.shape) > 2:
        depth = depth.squeeze()
    if len(intrinsic_matrix.shape) > 2:
        intrinsic_matrix = intrinsic_matrix.squeeze()
    if len(extrinsic_matrix.shape) > 2:
        extrinsic_matrix = extrinsic_matrix.squeeze()
    st = max(int(scale / 2) - 1, 0)
    if offset is not None:
        offset = offset[st::scale, st::scale]
    normal_ref = normal_from_depth_image(
        depth[st::scale, st::scale],
        intrinsic_matrix.to(depth.device),
        extrinsic_matrix.to(depth.device),
        offset,
    )
    normal_ref = normal_ref.permute(2, 0, 1)

    normal_image = rearrange(normal_ref, "c h w -> h w c")
    normal_image = (normal_image + 1) / 2 * 255
    normal_image = normal_image.numpy().astype(np.uint8)

    return normal_image[..., ::-1]


def normal_from_depth_image(
    depth, intrinsic_matrix, extrinsic_matrix, offset=None, gt_image=None
):
    # depth: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    # xyz_normal: (H, W, 3)
    xyz_world = depth2point_world(
        depth, intrinsic_matrix, extrinsic_matrix
    )  # (HxW, 3)
    xyz_world = xyz_world.reshape(*depth.shape, 3)
    xyz_normal = depth_pcd2normal(xyz_world, offset, gt_image)

    return xyz_normal


def depth2point_world(depth_image, intrinsic_matrix, extrinsic_matrix):
    # depth_image: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    _, xyz_cam = depth2point_cam(
        depth_image[None, None, None, ...], intrinsic_matrix[None, ...]
    )
    xyz_cam = xyz_cam.reshape(-1, 3)

    return xyz_cam


def depth_pcd2normal(xyz, offset=None, gt_image=None):
    hd, wd, _ = xyz.shape
    if offset is not None:
        ix, iy = torch.meshgrid(
            torch.arange(wd), torch.arange(hd), indexing="xy"
        )
        xy = (torch.stack((ix, iy), dim=-1)[1:-1, 1:-1]).to(xyz.device)
        p_offset = (
            torch.tensor([[0, 1], [0, -1], [1, 0], [-1, 0]])
            .float()
            .to(xyz.device)
        )
        new_offset = (
            p_offset[None, None] + offset.reshape(hd, wd, 4, 2)[1:-1, 1:-1]
        )
        xys = xy[:, :, None] + new_offset
        xys[..., 0] = 2 * xys[..., 0] / (wd - 1) - 1.0
        xys[..., 1] = 2 * xys[..., 1] / (hd - 1) - 1.0
        sampled_xyzs = torch.nn.functional.grid_sample(
            xyz.permute(2, 0, 1)[None], xys.reshape(1, -1, 1, 2)
        )
        sampled_xyzs = sampled_xyzs.permute(0, 2, 3, 1).reshape(
            hd - 2, wd - 2, 4, 3
        )
        bottom_point = sampled_xyzs[:, :, 0]
        top_point = sampled_xyzs[:, :, 1]
        right_point = sampled_xyzs[:, :, 2]
        left_point = sampled_xyzs[:, :, 3]
    else:
        bottom_point = xyz[..., 2:hd, 1 : wd - 1, :]
        top_point = xyz[..., 0 : hd - 2, 1 : wd - 1, :]
        right_point = xyz[..., 1 : hd - 1, 2:wd, :]
        left_point = xyz[..., 1 : hd - 1, 0 : wd - 2, :]
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = torch.nn.functional.pad(
        xyz_normal.permute(2, 0, 1), (1, 1, 1, 1), mode="constant"
    ).permute(1, 2, 0)

    return xyz_normal


if __name__ == "__main__":
    import os

    data_root = "/horizon-bucket/robot_lab/datasets/recons_sim/data_process/resolution_384/scan1"  # noqa
    camera_dict = np.load(os.path.join(data_root, "cameras.npz"))
    intrinsics = camera_dict["intrinsic"]
    c2ws = [
        camera_dict["c2w_mat_%d" % idx].astype(np.float32) for idx in range(60)
    ]
    image_idx = "000000"
    depth_image = os.path.join(data_root, f"{image_idx}_depth.npy")
    depth_image = np.load(depth_image)
    intrinsics = torch.tensor(intrinsics[None, :3, :3]).float()
    extrinsics = torch.tensor(camera_dict[f"c2w_mat_{int(image_idx)}"])[
        None, ...
    ].float()
    depth_image = torch.tensor(depth_image[None, ..., None]).float()

    normal_image = depth2normal(intrinsics, extrinsics, depth_image)
    normal_image = rearrange(normal_image, "c h w -> h w c")
    normal_image = (normal_image + 1) / 2 * 255
    normal_image = normal_image.numpy().astype(np.uint8)
    cv2.imwrite(f"depth2normal.png", normal_image[..., ::-1])
