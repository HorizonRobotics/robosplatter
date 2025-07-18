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


import logging
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from gsplat.cuda._wrapper import spherical_harmonics
from plyfile import PlyData, PlyElement
from robo_splatter.models.basic import (
    SH2RGB,
    GaussianData,
    GSInstance,
    gamma_shs,
    log_init_status,
    quat_mult,
    quat_to_rotmat,
    rotation_matrix_to_quaternion,
)
from robo_splatter.models.camera import BaseCamera

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


__all__ = [
    "VanillaGaussians",
    "RigidsGaussians",
]


class VanillaGaussians(nn.Module):
    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda",
        detach_keys: list[str] = [
            "activated_opacities",
            "means",
            "colors",
            "scales",
            "quats",
        ],
    ) -> None:
        super().__init__()

        self.device = device
        self.model_path = model_path
        self.detach_keys = detach_keys
        self.sh_degree = 0
        if model_path is not None:
            self.load(model_path)

    @classmethod
    @log_init_status
    def init_from_yaml(cls, yaml_path: str) -> None:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        if "background" not in config:
            return None

        return cls(
            model_path=config["background"]["model_path"],
            device=config["background"].get("device", "cuda"),
            detach_keys=config["background"].get("detach_keys", []),
        )

    def to(self, device: str) -> None:
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device)

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found.")

        if path.endswith(".ply"):
            self.load_ply(path)
        elif path.endswith(".ckpt"):
            self.load_splat_ckpt(path)
        else:
            raise NotImplementedError(f"Loading from {path} is not supported.")

    def load_ply(
        self,
        path: str,
        gamma: float = 1.0,
        apply_activate: bool = False,
    ) -> None:
        plydata = PlyData.read(path)
        xyz = torch.stack(
            (
                torch.tensor(plydata.elements[0]["x"], dtype=torch.float32),
                torch.tensor(plydata.elements[0]["y"], dtype=torch.float32),
                torch.tensor(plydata.elements[0]["z"], dtype=torch.float32),
            ),
            dim=1,
        )

        opacities = torch.tensor(
            plydata.elements[0]["opacity"], dtype=torch.float32
        ).unsqueeze(-1)
        features_dc = torch.zeros((xyz.shape[0], 3), dtype=torch.float32)
        features_dc[:, 0] = torch.tensor(
            plydata.elements[0]["f_dc_0"], dtype=torch.float32
        )
        features_dc[:, 1] = torch.tensor(
            plydata.elements[0]["f_dc_1"], dtype=torch.float32
        )
        features_dc[:, 2] = torch.tensor(
            plydata.elements[0]["f_dc_2"], dtype=torch.float32
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = torch.zeros(
            (xyz.shape[0], len(scale_names)), dtype=torch.float32
        )
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = torch.tensor(
                plydata.elements[0][attr_name], dtype=torch.float32
            )

        rot_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("rot_")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = torch.zeros((xyz.shape[0], len(rot_names)), dtype=torch.float32)
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = torch.tensor(
                plydata.elements[0][attr_name], dtype=torch.float32
            )

        rots = rots / torch.norm(rots, dim=-1, keepdim=True)

        if apply_activate:
            scales = torch.exp(scales)
            opacities = torch.sigmoid(opacities)

        # extra features
        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(
            extra_f_names, key=lambda x: int(x.split("_")[-1])
        )

        max_sh_degree = int(np.sqrt((len(extra_f_names) + 3) / 3) - 1)
        if max_sh_degree != 0:
            features_extra = torch.zeros(
                (xyz.shape[0], len(extra_f_names)), dtype=torch.float32
            )
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = torch.tensor(
                    plydata.elements[0][attr_name], dtype=torch.float32
                )

            features_extra = features_extra.view(
                (features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1)
            )
            features_extra = features_extra.permute(0, 2, 1)

            if abs(gamma - 1.0) > 1e-3:
                features_dc = gamma_shs(features_dc, gamma)
                features_extra[..., :] = 0.0
                opacities *= 0.8

            shs = torch.cat(
                [
                    features_dc.reshape(-1, 3),
                    features_extra.reshape(len(features_dc), -1),
                ],
                dim=-1,
            )
        else:
            # sh_dim is 0, only dc features
            shs = features_dc
            features_extra = None

        self._means = xyz
        self._opacities = opacities
        self._rgbs = shs[None, ...]  # (1, N, D)
        self._scales = scales
        self._quats = rots
        self.sh_degree = max(self.sh_degree, max_sh_degree)
        self._features_dc = features_dc
        self._features_rest = features_extra
        self.to(self.device)

    @staticmethod
    def filter_gs_points(filter_mask: torch.Tensor, data: dict) -> dict:
        for key in [
            "_means",
            "_rgbs",
            "_opacities",
            "_scales",
            "_quats",
            "_features_dc",
            "_features_rest",
        ]:
            if key not in data or data[key] is None:
                continue
            if key == "_rgbs":
                data[key] = data[key][..., filter_mask, :]
            else:
                data[key] = data[key][filter_mask]

        return data

    def get_gaussian_data(
        self,
        apply_activate: bool = True,
        filter_mask: torch.Tensor = None,
        **kwargs,
    ) -> GaussianData:
        """Get raw GaussianData object from the current model.

        Args:
            apply_activate: whether to activate the gaussians.
            filter_mask: mask for filtering gaussians.
        """
        gs_dict = dict(
            _means=self._means,
            _opacities=(
                self.get_opacity if apply_activate else self._opacities
            ),
            _scales=(self.get_scaling if apply_activate else self._scales),
            _quats=(self.get_quats if apply_activate else self._quats),
            _rgbs=getattr(self, "_rgbs", None),
            _features_dc=getattr(self, "_features_dc", None),
            _features_rest=getattr(self, "_features_rest", None),
            detach_keys=self.detach_keys,
            sh_degree=self.sh_degree,
            instance_ids=getattr(self, "instance_ids", None),
        )
        for key in kwargs:
            if key in gs_dict:
                gs_dict[key] = kwargs[key]

        if filter_mask is not None:
            gs_dict = self.filter_gs_points(filter_mask, gs_dict)

        return GaussianData(**gs_dict)

    def load_splat_ckpt(
        self, ckpt_path: str, apply_activate: bool = False
    ) -> None:
        ckpt = torch.load(ckpt_path, map_location="cuda")["splats"]
        self._means = ckpt["means"]
        self._quats = torch.nn.functional.normalize(ckpt["quats"], p=2, dim=-1)
        if apply_activate:
            self._scales = torch.exp(ckpt["scales"])
            self._opacities = torch.sigmoid(ckpt["opacities"])
        else:
            self._scales = ckpt["scales"]
            self._opacities = ckpt["opacities"]
        self._rgbs = torch.cat([ckpt["sh0"], ckpt["shN"]], dim=-2)
        self._rgbs = self._rgbs[None, ...]  # (1, N, D)
        self.sh_degree = ckpt["sh_degree"]
        self._features_dc = ckpt["sh0"]
        self._features_rest = ckpt["shN"]
        self.to(self.device)

    @property
    def colors(self):
        if self.sh_degree > 0:
            return SH2RGB(self._features_dc)
        else:
            return torch.sigmoid(self._features_dc)

    @property
    def shs_0(self):
        return self._features_dc

    @property
    def shs_rest(self):
        return self._features_rest

    @property
    def num_points(self):
        return self._means.shape[0]

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacities)

    @property
    def get_quats(self):
        return self.quat_norm(self._quats)

    @property
    def get_scaling(self):
        return torch.exp(self._scales)

    def quat_norm(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True)

    def _compute_gs_rgb(
        self, c2w: torch.Tensor, means: torch.Tensor
    ) -> torch.Tensor:
        colors = self._rgbs.reshape(
            1, len(means), -1, 3
        )  # (1, N, D) -> (1, N, K, 3)
        colors = colors.repeat(len(c2w), 1, 1, 1)  # (n_cam, N, K, 3)
        if self.sh_degree > 0:
            viewdirs = means[None, ...] - c2w[:, None, :3, 3]  # (n_cam, N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            rgbs = spherical_harmonics(self.sh_degree, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[..., 0, :])

        return rgbs  # (n_cam, N, 3)

    def get_gaussians(
        self,
        c2w: torch.Tensor,
        filter_mask: torch.Tensor = None,
        apply_activate: bool = True,
    ) -> GaussianData:
        """Get GaussianData object for input cam.

        Args:
            c2w: camera to world transform. (n_cam, 4 ,4)
            filter_mask: mask for filtering gaussians.
            apply_activate: whether to activate the gaussians.
        """

        # get colors of gaussians
        rgbs = self._compute_gs_rgb(c2w, self._means)

        return self.get_gaussian_data(apply_activate, filter_mask, _rgbs=rgbs)

    @property
    def device_memory_usage(self) -> int:
        mem = 0
        for tensor in [
            self._opacities,
            self._means,
            self._rgbs,
            self._scales,
            self._quats,
        ]:
            if tensor is not None:
                mem += tensor.element_size() * tensor.nelement()

        if self.extras:
            for t in self.extras.values():
                if isinstance(t, torch.Tensor):
                    mem += t.element_size() * t.nelement()

        return mem


class RigidsGaussians(VanillaGaussians):
    def __init__(
        self,
        instances: dict[int, GSInstance],
        device: str = "cuda",
        detach_keys: list[str] = [
            "activated_opacities",
            "means",
            "colors",
            "scales",
            "quats",
        ],
    ) -> None:
        super().__init__(detach_keys=detach_keys, device=device)
        self.instances = instances
        self.load_gs_models(instances)

    @classmethod
    @log_init_status
    def init_from_yaml(cls, yaml_path: str) -> None:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        if "foreground" not in config:
            return None

        instances = dict()
        for id, instance in config["foreground"]["instances"].items():
            instances[int(id)] = GSInstance.from_dict(instance)

        device = config["foreground"].get("device", "cuda")
        detach_keys = config["foreground"].get("detach_keys", [])

        return cls(instances, device, detach_keys)

    def load_gs_models(self, instances: dict[int, GSInstance]) -> None:
        """Load the GS models of the instances.

        Args:
            instances: dict of GSInstance objects.
        """
        all_means = []
        all_scales = []
        all_quats = []
        all_features_dc = []
        all_features_rest = []
        all_opacities = []

        all_instance_ids = []
        max_feat_rest_c = 0
        for id in instances.keys():
            instance: GSInstance = instances[id]
            self.load(instance.gs_model_path)
            data = self.get_gaussian_data(apply_activate=False)
            if instance.scale != 1.0:
                _scale = torch.tensor(instance.scale)
                data._means *= _scale
                data._scales += torch.log(_scale)

            all_means.append(data._means)
            all_scales.append(data._scales)
            all_quats.append(data._quats)
            all_features_dc.append(data._features_dc)
            if data._features_rest is not None:
                _features_rest = data._features_rest
            else:
                _features_rest = torch.zeros((len(data._means), 1, 3))
                _features_rest = _features_rest.to(data._features_dc)
            max_feat_rest_c = max(max_feat_rest_c, _features_rest.shape[1])
            all_features_rest.append(_features_rest)
            all_opacities.append(data._opacities)
            all_instance_ids.append(torch.full((data._means.shape[0], 1), id))

        self._means = torch.cat(all_means, dim=0)  # (N, 3)
        self._scales = torch.cat(all_scales, dim=0)  # (N, 3)
        self._quats = torch.cat(all_quats, dim=0)  # (N, 4)
        self._features_dc = torch.cat(all_features_dc, dim=0)  # (N, 3)
        padded = []
        for _features_rest in all_features_rest:
            current_dim = _features_rest.shape[1]
            if current_dim == max_feat_rest_c:
                padded.append(_features_rest)
                continue
            pad = (0, 0, 0, max_feat_rest_c - current_dim, 0, 0)
            padded_tensor = F.pad(
                _features_rest, pad, mode="constant", value=0
            )
            padded.append(padded_tensor)
        self._features_rest = torch.cat(padded, dim=0)  # (N, 1, 3)
        self._opacities = torch.cat(all_opacities, dim=0)  # (N, 1)
        self.instance_ids = torch.cat(all_instance_ids, dim=0)  # (N, 1)

        shs = torch.cat(
            [
                self._features_dc.reshape(-1, 3),
                self._features_rest.reshape(len(self._features_dc), -1),
            ],
            dim=-1,
        )
        self._rgbs = shs[None, ...]  # (1, N, D)

        self.to(self.device)

    @property
    def num_instances(self):
        return len(self.instances)

    def get_instances_quats(
        self, instances_pose: torch.Tensor
    ) -> torch.Tensor:
        """Get the quaternions of the instances."""

        num_instances = instances_pose.shape[1]
        quats = torch.zeros(num_instances, 4, device=self.device)

        poses = instances_pose[..., :3, :3].view(-1, 3, 3)

        # valid_mask = self.instances_fv.view(-1)
        # For unvisible instances:, keep all the instances are visible now
        valid_mask = torch.ones(
            num_instances, dtype=torch.bool, device=self.device
        )
        _quats = rotation_matrix_to_quaternion(poses[valid_mask])
        _quats = self.quat_norm(_quats)

        quats[valid_mask] = _quats
        quats[~valid_mask, 0] = 1.0

        return quats.reshape(num_instances, 4)

    def _compute_transform(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        instances_pose: dict[int, torch.Tensor],
    ):
        """Compute the transform of the GS models.

        Args:
            means: tensor of gs means.
            quats: tensor of gs quaternions.
            instances_pose: dict of instances poses.

        """
        instances_pose = torch.stack(list(instances_pose.values()), dim=0)
        # (x y z qx qy qz qw) -> (x y z qw qx qy qz)
        instances_pose = instances_pose[:, [0, 1, 2, 6, 3, 4, 5]]

        # TODO: design a dataclass for style transfom of instances_pose
        if instances_pose.shape[1] == 7:
            cur_instances_quats = self.quat_norm(instances_pose[..., 3:])
            cur_instances_trans = instances_pose[..., :3]
        else:
            raise ValueError(
                "The shape of instances_pose should be (num_instances, 7) or (num_instances, 4, 4)"  # noqa
            )

        rot_cur = quat_to_rotmat(cur_instances_quats, mode="wxyz")

        # update the means
        trans_per_pts = cur_instances_trans[
            self.instance_ids[..., 0]
        ]  # (num_gs, 3)
        quat_per_pts = cur_instances_quats[
            self.instance_ids[..., 0]
        ]  # (num_gs, 4)
        rot_per_pts = rot_cur[self.instance_ids[..., 0]]  # (num_gs, 3, 3)

        # update the means
        cur_means = (
            torch.bmm(rot_per_pts, means.unsqueeze(-1)).squeeze(-1)
            + trans_per_pts
        )

        # update the quats
        _quats = self.quat_norm(quats)
        cur_quats = quat_mult(quat_per_pts, _quats)

        return cur_means, cur_quats

    def get_gaussians(
        self,
        c2w: torch.Tensor,
        instances_pose: dict[int, torch.Tensor] = None,
        apply_activate: bool = True,
        filter_mask: torch.Tensor = None,
    ) -> GaussianData:
        """Get GaussianData from current model with given instances_pose."""
        if instances_pose is None or len(instances_pose) == 0:
            instances_pose = {
                key: torch.tensor(instance.init_pose).to(self.device)
                for key, instance in self.instances.items()
            }

        # For no instances_pose, use the init pose in config file.
        _instances_pose = dict()
        for uid, instance in self.instances.items():
            if instances_pose is None or uid not in instances_pose:
                pose = torch.tensor(instance.init_pose)
            else:
                pose = instances_pose[uid]

            _instances_pose[uid] = pose.clone().to(self.device)

        # compute the transformed gs means and quats
        world_means, world_quats = self._compute_transform(
            self._means, self._quats, _instances_pose
        )

        # get colors of gaussians
        rgbs = self._compute_gs_rgb(c2w, world_means)

        return self.get_gaussian_data(
            apply_activate,
            filter_mask,
            _rgbs=rgbs,
            _means=world_means,
            _quats=self.quat_norm(world_quats),
        )

    def remove_instances(self, ids: List[int]) -> None:
        """Remove instances from the RigidsGaussians model.

        Args:
            ids: list of instance ids to be removed.
        """
        for ins_ids in ids:
            mask = ~(self.instance_ids[..., 0] == ins_ids)
            self._means = self._means[mask]
            self._scales = self._scales[mask]
            self._quats = self._quats[mask]
            self._features_dc = self._features_dc[mask]
            self._features_rest = self._features_rest[mask]
            self._opacities = self._opacities[mask]
            self.instance_ids = self.instance_ids[mask]

    def collect_gaussians_from_ids(
        self, ids: List[int]
    ) -> dict[int, GaussianData]:
        """Collect GaussianData objects from the given instance ids.

        Args:
            ids: list of instance ids.
        """
        gaussians = {}
        for id in ids:
            if id not in gaussians:
                instance_raw_dict = {
                    "_means": self._means[self.instance_ids[..., 0] == id],
                    "_scales": self._scales[self.instance_ids[..., 0] == id],
                    "_quats": self._quats[self.instance_ids[..., 0] == id],
                    "_features_dc": self._features_dc[
                        self.instance_ids[..., 0] == id
                    ],
                    "_features_rest": self._features_rest[
                        self.instance_ids[..., 0] == id
                    ],
                    "_opacities": self._opacities[
                        self.instance_ids[..., 0] == id
                    ],
                    "instance_ids": self.instance_ids[
                        self.instance_ids[..., 0] == id
                    ],
                }
                gaussians[id] = GaussianData(**instance_raw_dict)

        return gaussians

    def replace_instances(self, replace_map: Dict[int, int]) -> None:
        """Replace instances from the RigidsGaussians model.

        Args:
            replace_map: {
                ins_id(to be replaced): ins_id(replace with)
                ...
            }
        """
        new_gaussians = self.collect_gaussians_from_ids(replace_map.values())
        for ins_id, new_id in replace_map.items():
            self.remove_instances([ins_id])
            new_gaussian = new_gaussians[new_id]
            self._means = torch.cat([self._means, new_gaussian._means], dim=0)
            self._scales = torch.cat(
                [self._scales, new_gaussian._scales], dim=0
            )
            self._quats = torch.cat([self._quats, new_gaussian._quats], dim=0)
            self._features_dc = torch.cat(
                [self._features_dc, new_gaussian._features_dc], dim=0
            )
            self._features_rest = torch.cat(
                [self._features_rest, new_gaussian._features_rest], dim=0
            )
            self._opacities = torch.cat(
                [self._opacities, new_gaussian._opacities], dim=0
            )
            self.instance_ids = torch.cat(
                [
                    self.instance_ids,
                    torch.full_like(new_gaussian.instance_ids, ins_id),
                ],
                dim=0,
            )


if __name__ == "__main__":
    background_model = VanillaGaussians.init_from_yaml(
        "robo_splatter/config/gs_data_basic.yaml"
    )
    foreground_model = RigidsGaussians.init_from_yaml(
        "robo_splatter/config/gs_data_basic.yaml"
    )
