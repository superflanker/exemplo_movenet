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

import tensorflow as tf
from SkyNet.Utils import preprocess


def detect(interpreter, input_tensor):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    is_dynamic_shape_model = input_details[0]['shape_signature'][2] == -1
    if is_dynamic_shape_model:
        input_tensor_index = input_details[0]['index']
        input_shape = input_tensor.shape
        interpreter.resize_tensor_input(
            input_tensor_index, input_shape, strict=True)
    interpreter.allocate_tensors()

    interpreter.set_tensor(input_details[0]['index'], input_tensor.numpy())

    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num_detections = interpreter.get_tensor(output_details[3]['index'])

    return classes, boxes, scores


class ObjectDetector:
    def __init__(self,
                 input_size,
                 interpreter_file='models/ssd_mobilenet_v2.tflite'):
        """
        Inicialização da classe
        :param input_size: o tamanho do quadro tratado pela Rede neural de classificação
        :param interpreter_file: o arquivo da cnn classificadora
        """
        self.__input_size = input_size
        self.__interpreter = tf.lite.Interpreter(model_path=interpreter_file)

    def __classify(self, frame):
        """
        detecção de objetos
        :param frame: imagem
        :return: classes, caixas delimitadoras e scores
        """
        classes, boxes, scores = detect(self.__interpreter,
                                        frame)

        classes = classes[0]

        boxes = boxes[0]

        scores = scores[0]

        return classes, boxes, scores

    def __detection_cleanup(self,
                            width,
                            height,
                            classes,
                            boxes,
                            scores,
                            confidence_threshold=0.5):
        """
        Limpeza de objetos detectados
        :param classes: classes detectadas
        :param boxes: caixas detectadas
        :param scores: scores obtidos
        :param confidence_threshold: nível de confiança mínima
        :return: new_classes, new_boxes, new_scores
        """

        new_classes = list()

        new_classnames = list()

        new_centroids = list()

        new_boxes = list()

        new_scores = list()

        for i in range(0, len(classes)):
            detection_class = classes[i]
            bbox = boxes[i]
            if detection_class == 0: # pessoa
                if scores[i] >= confidence_threshold:
                    x_min, y_min, x_max, y_max = int(width * bbox[1]), int(height * bbox[0]), int(width * bbox[3]), int(height * bbox[2])
                    new_classes.append(classes[i])
                    new_boxes.append(([x_min, y_min, x_max, y_max]))
                    new_centroids.append([(x_max + x_min)/2.0, (y_max+y_min)/2.0])
                    new_scores.append(scores[i])
                    new_classnames.append("Pessoa")
        return new_classes, new_classnames, new_centroids, new_boxes, new_scores

    def run_detector(self,
                     frame,
                     width,
                     height):

        """
        Roda o estimador
        :param height: altura da imagem originanl
        :param width: largura da imagem original
        :param frame: a imagem original
        :return: PoseEstimation[] array com as pessoas e posturas estimadas
        """
        img = preprocess(frame, self.__input_size)
        classes, boxes, scores = self.__classify(img)
        classes, classnames, centroids,  boxes, scores = self.__detection_cleanup(width,
                                                                                  height,
                                                                                  classes,
                                                                                  boxes,
                                                                                  scores)
        return classes, classnames, centroids, boxes, scores
