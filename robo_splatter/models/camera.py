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


from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

__all__ = [
    "Camera",
]


@dataclass
class BaseCamera:
    """Will be deprecated in next version."""

    # (4, 4) Camera-to-world matrix
    c2w: torch.Tensor
    # (3, 3) Intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    Ks: torch.Tensor
    image_height: int
    image_width: int
    device: str = "cpu"

    def __post_init__(self) -> None:
        if isinstance(self.c2w, np.ndarray):
            self.c2w = torch.from_numpy(self.c2w).float()
        if isinstance(self.Ks, np.ndarray):
            self.Ks = torch.from_numpy(self.Ks).float()

        self.to(self.device)

    @classmethod
    def init_from_params(
        cls,
        R: np.ndarray,
        T: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        image_height: int,
        image_width: int,
        device: str = "cpu",
    ) -> None:
        c2w = torch.eye(4, dtype=torch.float32)
        c2w[:3, :3] = torch.from_numpy(R) @ cls.rot_aux.T
        c2w[:3, 3] = torch.from_numpy(T)
        Ks = torch.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32
        )

        return cls(
            c2w=c2w,
            Ks=Ks,
            image_height=image_height,
            image_width=image_width,
            device=device,
        )

    @classmethod
    def init_from_pose_list(
        cls,
        pose_list: np.ndarray,
        camera_intrinsic: np.ndarray,
        image_height: int,
        image_width: int,
        device: str = "cpu",
    ) -> None:
        # pose_list: [x, y, z, qx, qy, qz, qw]
        T_cam2world = np.eye(4)
        T_cam2world[:3, :3] = Rotation.from_quat(pose_list[3:]).as_matrix()
        T_cam2world[:3, 3] = pose_list[:3]

        return cls(
            c2w=T_cam2world,
            Ks=camera_intrinsic,
            image_height=image_height,
            image_width=image_width,
            device=device,
        )

    def to(self, device: str) -> None:
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device)

    @property
    def rot_aux(self):
        return torch.tensor(
            [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=torch.float32
        )

    @property
    def fovx(self):
        half_w = torch.tensor(
            self.image_width / 2, dtype=self.Ks.dtype, device=self.device
        )
        return 2 * torch.atan2(half_w, self.Ks[0, 0])

    @property
    def fovy(self):
        half_h = torch.tensor(
            self.image_height / 2, dtype=self.Ks.dtype, device=self.device
        )
        return 2 * torch.atan2(half_h, self.Ks[1, 1])

    @property
    def euler(self):
        R = self.c2w[:3, :3].detach()
        with torch.no_grad():
            R_np = R.cpu().numpy()
            c2w_rot = R_np @ self.rot_aux.numpy().T
            rotation = Rotation.from_matrix(c2w_rot)
            roll, pitch, yaw = rotation.as_euler("xyz")

        return (
            torch.tensor(yaw, device=self.device, dtype=R.dtype),
            torch.tensor(pitch, device=self.device, dtype=R.dtype),
            torch.tensor(roll, device=self.device, dtype=R.dtype),
        )

    @property
    def mojuco_c2w(self) -> torch.Tensor:
        # TODO: Support multi simulator refactor `mojuco_c2w` in the future
        coord_align = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).to(self.c2w)

        return self.c2w @ coord_align


@dataclass
class Camera:
    """A class for batch processing of camera parameters.

    Attributes:
        c2w: Camera-to-world transformation matrix, shape (batch, 4, 4).
        Ks: Camera intrinsic matrix, shape (batch, 3, 3).
        image_height: Height of the image in pixels.
        image_width: Width of the image in pixels.
        device: Device to store tensors (e.g., 'cpu', 'cuda').
    """

    c2w: torch.Tensor
    Ks: torch.Tensor
    image_height: int
    image_width: int
    device: str = "cuda"

    MOJUCO_COORD_ALIGN = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    ISAAC_COORD_ALIGN = torch.tensor(
        [
            [0.0, 0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    def __post_init__(self) -> None:
        if isinstance(self.c2w, np.ndarray):
            self.c2w = torch.from_numpy(self.c2w).float()
        if isinstance(self.Ks, np.ndarray):
            self.Ks = torch.from_numpy(self.Ks).float()

        if self.c2w.ndim == 2:
            self.c2w = self.c2w.unsqueeze(0)
        if self.Ks.ndim == 2:
            self.Ks = self.Ks.unsqueeze(0)

        self.to(self.device)

    @classmethod
    def init_from_pose_list(
        cls,
        pose_list: np.ndarray,
        camera_intrinsic: np.ndarray,
        image_height: int,
        image_width: int,
        device: str = "cpu",
    ) -> "Camera":
        """Initialize from a list of poses and camera intrinsics.

        Args:
            pose_list: shape (batch, 7) with [x, y, z, qx, qy, qz, qw].
            camera_intrinsic: Array of shape (3, 3) or (batch, 3, 3).
            image_height: Image height in pixels.
            image_width: Image width in pixels.
            device: Device to store tensors.

        Returns:
            BatchCamera instance.
        """
        pose_list = np.asarray(pose_list)
        if pose_list.ndim == 1:
            pose_list = pose_list[np.newaxis]
        assert (
            pose_list.shape[-1] == 7
        ), f"Expected pose_list shape (*, 7), got {pose_list.shape}"
        batch_size = pose_list.shape[0]

        c2w = torch.eye(4, dtype=torch.float32).repeat(batch_size, 1, 1)
        rotations = Rotation.from_quat(
            pose_list[:, 3:]
        ).as_matrix()  # (batch, 3, 3)
        c2w[:, :3, :3] = torch.from_numpy(rotations).float()
        c2w[:, :3, 3] = torch.from_numpy(pose_list[:, :3]).float()

        camera_intrinsic = np.asarray(camera_intrinsic)
        if camera_intrinsic.ndim == 2:
            camera_intrinsic = np.repeat(
                camera_intrinsic[np.newaxis], batch_size, axis=0
            )
        assert camera_intrinsic.shape == (
            batch_size,
            3,
            3,
        ), f"Expected camera_intrinsic ({batch_size}, 3, 3), got {camera_intrinsic.shape}"  # noqa
        Ks = torch.from_numpy(camera_intrinsic).float()

        return cls(
            c2w=c2w,
            Ks=Ks,
            image_height=image_height,
            image_width=image_width,
            device=device,
        )

    def to(self, device: str) -> "Camera":
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")

        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device)
        self.device = device

        return self

    @property
    def mojuco_c2w(self) -> torch.Tensor:
        coord_align = (
            self.MOJUCO_COORD_ALIGN.to(self.c2w)
            .unsqueeze(0)
            .repeat(self.c2w.shape[0], 1, 1)
        )
        return self.c2w @ coord_align

    @property
    def isaac_c2w(self) -> torch.Tensor:
        coord_align = (
            self.ISAAC_COORD_ALIGN.to(self.c2w)
            .unsqueeze(0)
            .repeat(self.c2w.shape[0], 1, 1)
        )
        return self.c2w @ coord_align
