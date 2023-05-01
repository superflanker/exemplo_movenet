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

from SkyNet.PoseEstimation.PoseEstimation import PoseEstimation
from SkyNet.Annotations.PoseKeypoints import draw_keypoints, draw_connections, KEYPOINT_EDGE_INDS_TO_COLOR
from SkyNet.ObjectDetection.ObjectDetector import ObjectDetector
from SkyNet.ObjectTracking.CentroidTracker import CentroidTracker
from SkyNet.Utils import crop_bb
import cv2 as cv
import numpy as np


class SkyNet:

    def __init__(self,
                 capture_device=0,
                 pose_input_size=256,
                 detector_input_size=300,
                 pose_interpreter_file='models/singlepose_movenet.tflite',
                 detector_interpreter_file='models/ssd_mobilenet_v2.tflite',
                 use_deep_sort=True):

        self.__capture_device = cv.VideoCapture(capture_device)

        self.__pose_estimator = PoseEstimation(pose_input_size, pose_interpreter_file)

        self.__object_detector = ObjectDetector(detector_input_size,
                                                detector_interpreter_file)

        self.__tracker = CentroidTracker(10)

    def run(self):

        while self.__capture_device.isOpened():
            # Lendo o frame atual
            ret, frame = self.__capture_device.read()
            if not ret:
                break
            width = int(self.__capture_device.get(cv.CAP_PROP_FRAME_WIDTH))  # float `width`
            height = int(self.__capture_device.get(cv.CAP_PROP_FRAME_HEIGHT))  # float `height`
            # Convertendo o frame para RGB

            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            classes, names, centroids, bboxes, scores = self.__object_detector.run_detector(frame.copy(),
                                                                                            width,
                                                                                            height)

            image_crops = crop_bb(frame, bboxes)

            tracks = self.__tracker.update_tracks(bboxes)

            # estimação de postura - por objeto

            for i in range(0, len(image_crops)):

                image = image_crops[i]

                bbox = bboxes[i]

                offset_width = bbox[0]

                offset_height = bbox[1]

                im_height, im_width = image.shape[:2]

                pose = self.__pose_estimator.run_estimator(image,
                                                           offset_width,
                                                           offset_height,
                                                           im_width,
                                                           im_height)

                draw_keypoints(frame, pose.get_points(), 0.1)

                draw_connections(frame, pose.get_points(), 0.1)

            for (objectID, centroid) in tracks.items():
                text = "ID {}".format(objectID)

                cv.putText(frame,
                           text,
                           (centroid[0] - 10, centroid[1] - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5,
                           (0, 255, 0), 2)

                cv.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            # Convertendo o frame de volta para BGR
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            # Mostrando o frame processado

            cv.imshow('Video', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
