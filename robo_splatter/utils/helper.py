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
import cv2
import imageio
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mp4_from_images(
    images: list[np.ndarray],
    output_path: str,
    to_uint8: bool = False,
    fps: float = 10,
    prompt: str = None,
    cvt_mode: int = None,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX  # 字体样式
    font_scale = 0.5  # 字体大小
    font_thickness = 1  # 字体粗细
    color = (255, 255, 255)  # 文字颜色（白色）
    position = (20, 25)  # 左上角坐标 (x, y)

    with imageio.get_writer(output_path, fps=fps) as writer:
        for image in images:
            if to_uint8:
                image = image.clip(min=0, max=1)
                image = (255.0 * image).astype(np.uint8)

            if cvt_mode is not None:
                image = cv2.cvtColor(image, cvt_mode)

            if prompt is not None:
                cv2.putText(
                    image,
                    prompt,
                    position,
                    font,
                    font_scale,
                    color,
                    font_thickness,
                )

            writer.append_data(image)

    logger.info(f"MP4 video saved to {output_path}")
