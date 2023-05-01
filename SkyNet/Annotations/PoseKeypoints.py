"""
SkyNet - Detecção, Rastreamento e Classificação de Pose utilizando TensorFlow

Copyright 2023 Augusto Mathias Adams <augusto.adams@ufpr.br>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.
"""
import numpy as np
import matplotlib.colors as colors
import cv2 as cv

POINT_COLOR = [int(255 * color) for color in colors.to_rgb('mediumblue')]

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): [int(255 * color) for color in colors.to_rgb('m')],
    (0, 2): [int(255 * color) for color in colors.to_rgb('c')],
    (1, 3): [int(255 * color) for color in colors.to_rgb('m')],
    (2, 4): [int(255 * color) for color in colors.to_rgb('c')],
    (0, 5): [int(255 * color) for color in colors.to_rgb('m')],
    (0, 6): [int(255 * color) for color in colors.to_rgb('c')],
    (5, 7): [int(255 * color) for color in colors.to_rgb('m')],
    (7, 9): [int(255 * color) for color in colors.to_rgb('m')],
    (6, 8): [int(255 * color) for color in colors.to_rgb('c')],
    (8, 10): [int(255 * color) for color in colors.to_rgb('c')],
    (5, 6): [int(255 * color) for color in colors.to_rgb('y')],
    (5, 11): [int(255 * color) for color in colors.to_rgb('m')],
    (6, 12): [int(255 * color) for color in colors.to_rgb('c')],
    (11, 12): [int(255 * color) for color in colors.to_rgb('y')],
    (11, 13): [int(255 * color) for color in colors.to_rgb('m')],
    (13, 15): [int(255 * color) for color in colors.to_rgb('m')],
    (12, 14): [int(255 * color) for color in colors.to_rgb('c')],
    (14, 16): [int(255 * color) for color in colors.to_rgb('c')]
}


def draw_keypoints(frame, keypoints, confidence_threshold):
    """
    Marca os pontos de postura na imagem
    :param frame: imagem
    :param keypoints: os pontos de postura
    :param confidence_threshold: o limite minimo de confiança
    :return: None
    """
    y, x, c = frame.shape
    # shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in keypoints:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv.circle(frame, (int(kx), int(ky)), 4, POINT_COLOR, -1)


def draw_connections(frame,
                     keypoints,
                     confidence_threshold):
    """
    Anota as linhas de postura
    :param frame: a imagem
    :param keypoints: os pontos da postura
    :param confidence_threshold: o nível de confiança da predição
    :return: None
    """
    y, x, c = frame.shape
    # shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        p1, p2 = edge
        y1, x1, c1 = keypoints[p1]
        y2, x2, c2 = keypoints[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
