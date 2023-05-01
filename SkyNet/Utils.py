import tensorflow as tf
import cv2 as cv
import numpy as np


def preprocess(frame, img_size):
    """
    Pré-Processamento da imagem do opencv
    :param frame: quadro do opencv
    :param img_size: tamanho final do quadro
    :return: o quadro redimensionado e com a cor corrigida
    """

    resized_frame = cv.resize(frame, (img_size, img_size))

    # Converte de RGB para BGR
    bgr_frame = cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB)

    # Opcional, mas recimendado
    rgb_frame = tf.convert_to_tensor(bgr_frame, dtype=tf.uint8)

    # Adiciona dimensão - importante!!!

    image_tensor = tf.expand_dims(rgb_frame, 0)

    return image_tensor


def crop_bb(frame, raw_dets):
    crops = []
    im_height, im_width = frame.shape[:2]
    for i, detection in enumerate(raw_dets):
        l, t, r, b = [int(x) for x in detection]
        crop_l = max(0, l)
        crop_r = min(im_width, r)
        crop_t = max(0, t)
        crop_b = min(im_height, b)
        crops.append(frame[crop_t:crop_b, crop_l:crop_r])

    return crops