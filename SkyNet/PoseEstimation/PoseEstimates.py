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
from SkyNet.Annotations.PoseKeypoints import KEYPOINT_DICT, KEYPOINT_EDGE_INDS_TO_COLOR

class PoseEstimates:
    """
    Conteiner dos pontos chave e da caixa delimitadora
    """

    def __init__(self,
                 keypoints_with_scores,
                 offset_width,
                 offset_height,
                 image_width,
                 image_height):
        scores = []
        points = []
        points_with_scores = np.reshape(keypoints_with_scores, (17, 3))
        for i in range(0, len(points_with_scores)):
            y_p = offset_height + int(image_height * points_with_scores[i][0])
            x_p = offset_width + int(image_width * points_with_scores[i][1])
            score = points_with_scores[i][2]
            points.append([y_p, x_p, score])
            scores.append(score)
        self.__scores = np.array(scores)
        self.__points = np.array(points)

    def get_points(self):
        return self.__points
