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
import tensorflow as tf
import cv2 as cv
from .PoseEstimates import PoseEstimates
from SkyNet.Utils import preprocess


def detect(interpreter, input_tensor):
    """Runs detection on an input image.

  Args:
    interpreter: tf.lite.Interpreter
    input_tensor: A [1, input_height, input_width, 3] Tensor of type tf.float32.
      input_size is specified when converting the model to TFLite.

  Returns:
    A tensor of shape [1, 6, 56].
  """

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

    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores


class PoseEstimation:
    """
    Estimação de Postura utilizando a Movenet Multipose

    """

    def __init__(self,
                 input_size,
                 interpreter_file='models/singlepose_movenet.tflite'):
        """
        Inicialização da classe
        :param input_size: o tamanho do quadro tratado pela Rede neural de classificação
        :param interpreter_file: o arquivo
        """
        self.__input_size = input_size
        self.__interpreter = tf.lite.Interpreter(model_path=interpreter_file)

    def __classify(self, frame):
        """
        Estimação de pose
        :param frame: imagem
        :return: os pontos chave e a caixa delimitadora
        """
        keypoints_with_scores = detect(self.__interpreter,
                                       frame)

        return keypoints_with_scores

    def run_estimator(self,
                      frame,
                      offset_width,
                      offset_height,
                      width,
                      height):
        """
        Roda o estimador
        :param frame: a imagem original
        :param offset_height: offset do quadro selecionado
        :param offset_width: offset do quadro selecionado
        :param height: altura da imagem original
        :param width: largura da imagem original
        :return: a postura da pessoa
        """
        img = preprocess(frame, self.__input_size)
        keypoints = self.__classify(img)
        pose = PoseEstimates(keypoints,
                             offset_width=offset_width,
                             offset_height=offset_height,
                             image_width=width,
                             image_height=height)
        return pose
