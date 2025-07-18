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
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from dataclasses_json import DataClassJsonMixin
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
from torch.nn import functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


__all__ = [
    "GaussianData",
    "GSInstance",
    "RenderConfig",
    "log_init_status",
]


def gamma_shs(shs: torch.Tensor, gamma: float) -> torch.Tensor:
    C0 = 0.28209479177387814  # Constant for normalization in spherical harmonics  # noqa
    # Clip to the range [0.0, 1.0], apply gamma correction, and then un-clip back  # noqa
    new_shs = torch.clip(shs * C0 + 0.5, 0.0, 1.0)
    new_shs = (torch.pow(new_shs, gamma) - 0.5) / C0
    return new_shs


@dataclass
class GaussianData:
    _opacities: torch.Tensor
    _means: torch.Tensor
    _scales: torch.Tensor
    _quats: torch.Tensor
    _rgbs: Optional[torch.Tensor] = None
    _features_dc: Optional[torch.Tensor] = None
    _features_rest: Optional[torch.Tensor] = None
    sh_degree: Optional[int] = 0
    detach_keys: List[str] = field(default_factory=list)
    extras: Optional[Dict[str, torch.Tensor]] = None
    instance_ids: Optional[torch.Tensor] = None

    def to(self, device: str) -> None:
        for k, v in self.__dict__.items():
            if not isinstance(v, torch.Tensor):
                continue
            self.__dict__[k] = v.to(device)

    def _apply_detach(
        self, tensor_name: str, tensor: torch.Tensor
    ) -> torch.Tensor:
        if tensor_name in self.detach_keys:
            return tensor.detach()
        return tensor

    @property
    def opacities(self):
        return self._apply_detach("activated_opacities", self._opacities)

    @property
    def means(self):
        return self._apply_detach("means", self._means)

    @property
    def rgbs(self):
        return self._apply_detach("colors", self._rgbs)

    @property
    def scales(self):
        return self._apply_detach("scales", self._scales)

    @property
    def quats(self):
        return self._apply_detach("quats", self._quats)

    @property
    def sh_dim(self):
        return self._rgbs.shape[-1]

    def __len__(self):
        return len(self.means)

    def save_to_ply(
        self, path: str, colors: torch.Tensor = None, enable_mask: bool = False
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        numpy_data = self.get_numpy_data()
        means = numpy_data["_means"]
        scales = numpy_data["_scales"]
        quats = numpy_data["_quats"]
        opacities = numpy_data["_opacities"]
        sh0 = numpy_data["_features_dc"]
        shN = numpy_data.get("_features_rest", np.zeros((means.shape[0], 0)))
        shN = shN.reshape(means.shape[0], -1)

        # Create a mask to identify rows with NaN or Inf in any of the numpy_data arrays  # noqa
        if enable_mask:
            invalid_mask = (
                np.isnan(means).any(axis=1)
                | np.isinf(means).any(axis=1)
                | np.isnan(scales).any(axis=1)
                | np.isinf(scales).any(axis=1)
                | np.isnan(quats).any(axis=1)
                | np.isinf(quats).any(axis=1)
                | np.isnan(opacities).any(axis=0)
                | np.isinf(opacities).any(axis=0)
                | np.isnan(sh0).any(axis=1)
                | np.isinf(sh0).any(axis=1)
                | np.isnan(shN).any(axis=1)
                | np.isinf(shN).any(axis=1)
            )

            # Filter out rows with NaNs or Infs from all data arrays
            means = means[~invalid_mask]
            scales = scales[~invalid_mask]
            quats = quats[~invalid_mask]
            opacities = opacities[~invalid_mask]
            sh0 = sh0[~invalid_mask]
            shN = shN[~invalid_mask]

        num_points = means.shape[0]

        with open(path, "wb") as f:
            # Write PLY header
            f.write(b"ply\n")
            f.write(b"format binary_little_endian 1.0\n")
            f.write(f"element vertex {num_points}\n".encode())
            f.write(b"property float x\n")
            f.write(b"property float y\n")
            f.write(b"property float z\n")
            f.write(b"property float nx\n")
            f.write(b"property float ny\n")
            f.write(b"property float nz\n")

            if colors is not None:
                for j in range(colors.shape[1]):
                    f.write(f"property float f_dc_{j}\n".encode())
            else:
                for i, data in enumerate([sh0, shN]):
                    prefix = "f_dc" if i == 0 else "f_rest"
                    for j in range(data.shape[1]):
                        f.write(f"property float {prefix}_{j}\n".encode())

            f.write(b"property float opacity\n")

            for i in range(scales.shape[1]):
                f.write(f"property float scale_{i}\n".encode())
            for i in range(quats.shape[1]):
                f.write(f"property float rot_{i}\n".encode())

            f.write(b"end_header\n")

            # Write vertex data
            for i in range(num_points):
                f.write(struct.pack("<fff", *means[i]))  # x, y, z
                f.write(struct.pack("<fff", 0, 0, 0))  # nx, ny, nz (zeros)

                if colors is not None:
                    color = colors.detach().cpu().numpy()
                    for j in range(color.shape[1]):
                        f_dc = (color[i, j] - 0.5) / 0.2820947917738781
                        f.write(struct.pack("<f", f_dc))
                else:
                    for data in [sh0, shN]:
                        for j in range(data.shape[1]):
                            f.write(struct.pack("<f", data[i, j]))

                f.write(struct.pack("<f", opacities[i]))  # opacity

                for data in [scales, quats]:
                    for j in range(data.shape[1]):
                        f.write(struct.pack("<f", data[i, j]))


@dataclass
class GSInstance(DataClassJsonMixin):
    gs_model_path: str
    mesh_model_path: Optional[str] = None
    init_pose: list[float] = (0, 0, 0, 0, 0, 0, 1)  # [x, y, z, qx, qy, qz, qw]
    instance_name: Optional[str] = None
    class_name: Optional[str] = None
    scale: float = 1.0

    def __post_init__(self):
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = Rotation.from_quat(
            self.init_pose[3:]
        ).as_matrix()
        pose_matrix[:3, 3] = self.init_pose[:3]
        self.pose_matrix = pose_matrix


@dataclass
class RenderConfig:
    near_plane: float = 0.01  # Near plane for rendering
    far_plane: float = 10000000000.0  # Far plane for rendering
    antialiased: bool = (
        False  # Whether to use antialiasing for gaussian rendering  # noqa
    )
    packed: bool = False  # Whether to use packed rendering
    absgrad: bool = True  # Whether to use absolute gradient for rendering
    sparse_grad: bool = False  # Whether to use sparse gradient for rendering
    batch_size: int = 1  # Batch size for rendering, currently only support 1


def log_init_status(func):
    def wrapper(cls, yaml_path, *args, **kwargs):
        cls_object = func(cls, yaml_path, *args, **kwargs)
        logger.info(f"Successfully init {cls.__name__} from {yaml_path}")

        return cls_object

    return wrapper


@dataclass
class MeshData:
    vertices: torch.Tensor  # Vertices of the mesh (shape: [N, 3])
    faces: torch.Tensor  # Faces (shape: [M, 3] for triangular meshes)
    normals: Optional[torch.Tensor] = None  # Normals (shape: [N, 3]), optional
    texture_coords: Optional[torch.Tensor] = (
        None  # Texture coordinates (shape: [N, 2]), optional
    )

    def __post_init__(self):
        # Ensure vertices, faces, normals, and texture_coords are tensors
        self.vertices = self._to_tensor(self.vertices)
        self.faces = self._to_tensor(self.faces)
        if self.normals is not None:
            self.normals = self._to_tensor(self.normals)
        if self.texture_coords is not None:
            self.texture_coords = self._to_tensor(self.texture_coords)

    def _to_tensor(self, data):
        """Converts data to tensor if it's not already a tensor."""
        if isinstance(data, torch.Tensor):
            return data
        return torch.tensor(data, dtype=torch.float32)

    @classmethod
    def from_file(cls, filepath: str):
        vertices = []
        faces = []
        normals = []
        texture_coords = []

        with open(filepath, "r") as file:
            for line in file:
                parts = line.strip().split()
                if not parts:
                    continue

                if parts[0] == "v":
                    vertices.append(
                        [float(val) for val in parts[1:4]]
                    )  # Vertex (x, y, z)
                elif parts[0] == "vn":
                    normals.append(
                        [float(val) for val in parts[1:4]]
                    )  # Normal (nx, ny, nz)
                elif parts[0] == "vt":
                    texture_coords.append(
                        [float(val) for val in parts[1:3]]
                    )  # Texture coordinates (u, v)
                elif parts[0] == "f":
                    face = []
                    for vertex in parts[1:]:
                        indices = vertex.split("/")
                        face.append(
                            int(indices[0]) - 1
                        )  # Indices are 1-based in .obj files
                    faces.append(face)

        # Convert to tensors and return an instance of MeshData
        return cls(
            vertices=torch.tensor(vertices, dtype=torch.float32),
            faces=torch.tensor(faces, dtype=torch.long),
            normals=(
                torch.tensor(normals, dtype=torch.float32) if normals else None
            ),
            texture_coords=(
                torch.tensor(texture_coords, dtype=torch.float32)
                if texture_coords
                else None
            ),
        )

    def simplify(self, factor: float):
        num_vertices_to_keep = int(len(self.vertices) * factor)
        simplified_vertices = self.vertices[:num_vertices_to_keep]

        # For simplicity, we just reduce the vertices without altering faces
        return MeshData(
            vertices=simplified_vertices,
            faces=self.faces,
            normals=self.normals,
            texture_coords=self.texture_coords,
        )

    def apply_transformation(self, transformation_matrix: torch.Tensor):
        homogenous_vertices = torch.cat(
            (
                self.vertices,
                torch.ones((self.vertices.shape[0], 1), dtype=torch.float32),
            ),
            dim=1,
        )
        transformed_vertices = torch.matmul(
            homogenous_vertices, transformation_matrix.T
        )[:, :3]
        return MeshData(
            vertices=transformed_vertices,
            faces=self.faces,
            normals=self.normals,
            texture_coords=self.texture_coords,
        )

    def get_vertex_normals(self):
        if self.normals is None:
            normals = torch.zeros_like(self.vertices)
            for i, face in enumerate(self.faces):
                v0, v1, v2 = self.vertices[face]
                # Calculate the normal of the face using cross product
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = torch.cross(edge1, edge2)
                normal = normal / normal.norm()  # Normalize the normal
                # Assign the computed normal to all vertices of the face
                for idx in face:
                    normals[idx] += normal

            # Normalize the normals for each vertex
            normals = normals / normals.norm(dim=1, keepdim=True)
            self.normals = normals

        return self.normals


def k_nearest_sklearn(x: torch.Tensor, k: int):
    # Convert tensor to numpy array
    x_np = x.cpu().numpy()

    # Build the nearest neighbors model
    nn_model = NearestNeighbors(
        n_neighbors=k + 1, algorithm="auto", metric="euclidean"
    ).fit(x_np)

    # Find the k-nearest neighbors
    distances, indices = nn_model.kneighbors(x_np)

    # Exclude the point itself from the result and return
    return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(
        np.float32
    )


def interpolate_quats(q1, q2, fraction=0.5):
    q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)
    q2 = q2 / torch.norm(q2, dim=-1, keepdim=True)

    dot = (q1 * q2).sum(dim=-1)
    dot = torch.clamp(dot, -1, 1)

    neg_mask = dot < 0
    q2[neg_mask] = -q2[neg_mask]
    dot[neg_mask] = -dot[neg_mask]

    similar_mask = dot > 0.9995
    q_interp_similar = q1 + fraction * (q2 - q1)

    theta_0 = torch.acos(dot)
    theta = theta_0 * fraction

    sin_theta = torch.sin(theta)
    sin_theta_0 = torch.sin(theta_0)

    s1 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0

    q_interp = (s1[..., None] * q1) + (s2[..., None] * q2)

    final_q_interp = torch.zeros_like(q1)
    final_q_interp[similar_mask] = q_interp_similar[similar_mask]
    final_q_interp[~similar_mask] = q_interp[~similar_mask]
    return final_q_interp


def quat_mult(q1, q2):
    # NOTE:
    # Q1 is the quaternion that rotates the vector from the original position to the final position  # noqa
    # Q2 is the quaternion that been rotated
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def multiple_quaternion_vector3d(qwxyz, vxyz, last_scalar=False):
    if last_scalar:
        # qxyzw -> qwxyz
        qwxyz = qwxyz[..., [3, 0, 1, 2]]
    qw = qwxyz[..., 0]
    qx = qwxyz[..., 1]
    qy = qwxyz[..., 2]
    qz = qwxyz[..., 3]
    vx = vxyz[..., 0]
    vy = vxyz[..., 1]
    vz = vxyz[..., 2]
    qvw = -vx * qx - vy * qy - vz * qz
    qvx = vx * qw - vy * qz + vz * qy
    qvy = vx * qz + vy * qw - vz * qx
    qvz = -vx * qy + vy * qx + vz * qw
    vx_ = qvx * qw - qvw * qx + qvz * qy - qvy * qz
    vy_ = qvy * qw - qvz * qx - qvw * qy + qvx * qz
    vz_ = qvz * qw + qvy * qx - qvx * qy - qvw * qz
    return torch.stack([vx_, vy_, vz_], dim=-1).cuda().requires_grad_(False)


def multiple_quaternions(qwxyz1, qwxyz2, last_scalar=False):
    if last_scalar:
        # qxyzw -> qwxyz
        qwxyz1 = qwxyz1[..., [3, 0, 1, 2]]
        qwxyz2 = qwxyz2[..., [3, 0, 1, 2]]

    q1w = qwxyz1[..., 0]
    q1x = qwxyz1[..., 1]
    q1y = qwxyz1[..., 2]
    q1z = qwxyz1[..., 3]

    q2w = qwxyz2[..., 0]
    q2x = qwxyz2[..., 1]
    q2y = qwxyz2[..., 2]
    q2z = qwxyz2[..., 3]

    qw_ = q1w * q2w - q1x * q2x - q1y * q2y - q1z * q2z
    qx_ = q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y
    qy_ = q1w * q2y - q1x * q2z + q1y * q2w + q1z * q2x
    qz_ = q1w * q2z + q1x * q2y - q1y * q2x + q1z * q2w

    return (
        torch.stack([qw_, qx_, qy_, qz_], dim=-1).cuda().requires_grad_(False)
    )


def rotation_matrix_to_quaternion(
    rotation_matrices: torch.Tensor,
) -> torch.Tensor:
    """Converts rotation matrices to quaternions.

    Args:
        rotation_matrices: A tensor of shape (N, 3, 3) representing rotation matrices.  # noqa
    Returns:
        A tensor of shape (N, 4) representing quaternions, listed as (w, x, y, z).  # noqa
    """
    if len(rotation_matrices.shape) == 2:
        rotation_matrices = rotation_matrices.unsqueeze(0)
    # 提取旋转矩阵的旋转部分
    rotation_part = rotation_matrices[:, :3, :3]

    # 计算四元数的分量
    trace = (
        rotation_part[:, 0, 0]
        + rotation_part[:, 1, 1]
        + rotation_part[:, 2, 2]
    )
    qw = torch.sqrt(torch.clamp(1.0 + trace, min=1e-12)) / 2.0
    s = 4.0 * qw
    qx = (rotation_part[:, 2, 1] - rotation_part[:, 1, 2]) / s
    qy = (rotation_part[:, 0, 2] - rotation_part[:, 2, 0]) / s
    qz = (rotation_part[:, 1, 0] - rotation_part[:, 0, 1]) / s

    # 将四元数堆叠成一个张量
    quaternions = torch.stack((qw, qx, qy, qz), dim=1)
    return quaternions


def quat_to_rotmat(quats: torch.Tensor, mode="wxyz") -> torch.Tensor:
    """Convert quaternion to rotation matrix."""
    quats = F.normalize(quats, p=2, dim=-1)

    if mode == "xyzw":
        x, y, z, w = torch.unbind(quats, dim=-1)
    elif mode == "wxyz":
        w, x, y, z = torch.unbind(quats, dim=-1)
    else:
        raise ValueError(f"Invalid mode: {mode}.")

    R = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )

    return R.reshape(quats.shape[:-1] + (3, 3))


def RGB2SH(rgb):
    """Converts from RGB values [0,1] to 0th spherical harmonic coeff."""
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """Converts from 0th spherical harmonic coefficient to RGB values [0,1]."""
    C0 = 0.28209479177387814
    return sh * C0 + 0.5
