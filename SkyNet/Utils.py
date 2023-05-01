import tensorflow as tf
import cv2 as cv
import numpy as np
from collections import OrderedDict


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
    crops = OrderedDict()
    im_height, im_width = frame.shape[:2]
    for i in raw_dets.keys():
        detection = raw_dets[i]
        l, t, r, b = [int(x) for x in detection]
        crop_l = max(0, l)
        crop_r = min(im_width, r)
        crop_t = max(0, t)
        crop_b = min(im_height, b)
        crops[i] = frame[crop_t:crop_b, crop_l:crop_r]

    return crops


def non_max_suppression(boxes,
                        max_bbox_overlap,
                        scores=None):
    """Suppress overlapping detections.
    Original code from [1]_ has been adapted to include confidence score.
    .. [1] http://www.pyimagesearch.com/2015/02/16/
           faster-non-maximum-suppression-python/
    Examples
    --------
        >>> boxes = [d.roi for d in detections]
        >>> scores = [d.confidence for d in detections]
        >>> indices = non_max_suppression(boxes, max_bbox_overlap, scores)
        >>> detections = [detections[i] for i in indices]
    Parameters
    ----------
    boxes : ndarray
        Array of ROIs (x, y, width, height).
    max_bbox_overlap : float
        ROIs that overlap more than this values are suppressed.
    scores : Optional[array_like]
        Detector confidence score.
    Returns
    -------
    List[int]
        Returns indices of detections that have survived non-maxima suppression.
    """
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float32)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > max_bbox_overlap)[0]))
        )
    return pick