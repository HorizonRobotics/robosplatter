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
from dataclasses import dataclass


import cv2
import numpy as np
import torch
import torch.nn as nn
from gsplat.rendering import rasterization
from robo_splatter.models.basic import GaussianData, RenderConfig
from robo_splatter.models.camera import Camera
from robo_splatter.models.gaussians import RigidsGaussians, VanillaGaussians

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    "Scene",
    "RenderResult",
    "SceneRenderType",
]


@dataclass
class RenderResult:
    rgb: np.ndarray
    depth: np.ndarray
    opacity: np.ndarray
    bgr2rgb: bool = False

    def __post_init__(
        self,
    ):
        if isinstance(self.depth, torch.Tensor):
            self.depth = self.depth.detach().cpu().numpy()
        if isinstance(self.opacity, torch.Tensor):
            opacity = self.opacity.detach().cpu().numpy()
            self.opacity = (opacity * 255).astype(np.uint8)
        if isinstance(self.rgb, torch.Tensor):
            rgb = self.rgb.detach().cpu().numpy()
            self.rgb = (rgb * 255).astype(np.uint8)
        if self.bgr2rgb:
            self.rgb = self.rgb[..., ::-1]


@dataclass
class SceneRenderType(str):
    BACKGROUND: str = "BACKGROUND"
    FOREGROUND: str = "FOREGROUND"
    SCENE: str = "SCENE"


@dataclass
class RenderCoordSystem(str):
    MUJOCO: str = "MUJOCO"
    GAUSSIAN: str = "GAUSSIAN"
    ISAAC: str = "ISAAC"


class Scene(nn.Module):
    def __init__(
        self,
        render_config: RenderConfig,
        background_models: VanillaGaussians = None,
        foreground_models: RigidsGaussians = None,
        device: str = "cuda",
    ) -> None:
        super().__init__()

        self.foreground_models = foreground_models
        self.background_models = background_models
        self.render_cfg = render_config
        self.device = device
        if self.foreground_models is not None:
            self.foreground_models.to(device)
        if self.background_models is not None:
            self.background_models.to(device)

        self.set_eval()

    def set_train(self):
        if self.foreground_models is not None:
            self.foreground_models.train()
        if self.background_models is not None:
            self.background_models.train()
        self.train()

    def set_eval(self):
        if self.foreground_models is not None:
            self.foreground_models.eval()
        if self.background_models is not None:
            self.background_models.eval()
        self.eval()

    def _render_fn(
        self,
        gs: GaussianData,
        camera: Camera,
        opaticy_mask: torch.Tensor = None,
        **kwargs,
    ):
        assert (
            self.render_cfg.batch_size == 1
        ), "batch size must be 1, will support batch size > 1 in the future"

        renders, alphas, info = rasterization(
            means=gs.means,
            quats=gs.quats,
            scales=gs.scales,
            opacities=(
                gs.opacities.squeeze() * opaticy_mask
                if opaticy_mask is not None
                else gs.opacities.squeeze()
            ),
            colors=gs.rgbs,
            viewmats=torch.linalg.inv(camera.c2w),  # [C, 4, 4]
            Ks=camera.Ks,  # [C, 3, 3]
            width=camera.image_width,
            height=camera.image_height,
            packed=self.render_cfg.packed,
            absgrad=self.render_cfg.absgrad,
            sparse_grad=self.render_cfg.sparse_grad,
            rasterize_mode=(
                "antialiased" if self.render_cfg.antialiased else "classic"
            ),
            **kwargs,
        )

        assert renders.shape[-1] == 4, f"Must render rgb, depth and alpha"
        rendered_bgr, rendered_depth = torch.split(renders, [3, 1], dim=-1)

        return (
            rendered_bgr,
            rendered_depth,
            alphas,
            info,
        )

    def render_gaussians(
        self,
        gs: GaussianData,
        camera: Camera,
        **kwargs,
    ) -> RenderResult:
        bgr, depth, opacity, render_info = self._render_fn(
            gs, camera, **kwargs
        )

        if self.training:
            render_info["means2d"].retain_grad()

        return RenderResult(bgr, depth, opacity, bgr2rgb=True)

    def collect_gaussians(
        self,
        c2w: torch.Tensor,
        instances_pose: dict[int, torch.Tensor] = None,
        render_type: SceneRenderType = SceneRenderType.BACKGROUND,
    ) -> GaussianData:
        if render_type == SceneRenderType.BACKGROUND:
            assert self.background_models is not None
            gaussians = self.background_models.get_gaussians(c2w)
        elif render_type == SceneRenderType.FOREGROUND:
            assert self.foreground_models is not None
            gaussians = self.foreground_models.get_gaussians(
                c2w, instances_pose
            )
        elif render_type == SceneRenderType.SCENE:
            assert self.background_models is not None
            assert self.foreground_models is not None

            foreground_gs = self.foreground_models.get_gaussians(
                c2w, instances_pose
            )
            background_gs = self.background_models.get_gaussians(c2w)
            gaussians = GaussianData(
                _means=torch.cat(
                    [foreground_gs._means, background_gs._means], dim=0
                ),
                _scales=torch.cat(
                    [foreground_gs._scales, background_gs._scales], dim=0
                ),
                _quats=torch.cat(
                    [foreground_gs._quats, background_gs._quats], dim=0
                ),
                _rgbs=torch.cat(
                    [foreground_gs._rgbs, background_gs._rgbs], dim=1
                ),
                _opacities=torch.cat(
                    [foreground_gs._opacities, background_gs._opacities], dim=0
                ),
                detach_keys=background_gs.detach_keys,
                extras=background_gs.extras,
            )
        else:
            raise ValueError(f"Unknown render type: {render_type}")

        return gaussians

    def render(
        self,
        camera: Camera,
        instances_pose: dict[int, torch.Tensor] = None,
        render_type: SceneRenderType = SceneRenderType.BACKGROUND,
        coord_system: RenderCoordSystem = RenderCoordSystem.GAUSSIAN,
    ) -> RenderResult:
        gs_model = self.collect_gaussians(
            c2w=camera.c2w,
            instances_pose=instances_pose,
            render_type=render_type,
        )

        if coord_system == RenderCoordSystem.MUJOCO:
            camera.c2w = camera.mojuco_c2w
        elif coord_system == RenderCoordSystem.ISAAC:
            camera.c2w = camera.isaac_c2w

        outputs = self.render_gaussians(
            gs=gs_model,
            camera=camera,
            near_plane=self.render_cfg.near_plane,
            far_plane=self.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=getattr(self.render_cfg, "radius_clip", 0.0),
        )

        return outputs
