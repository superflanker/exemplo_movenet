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

import cv2 as cv


def draw_rectangle(l, t, r, b, image, label=None, thickness=2, color=(0, 0, 255)):
    """Draw a rectangle.

    Parameters
    ----------
    x : float | int
        Top left corner of the rectangle (x-axis).
    y : float | int
        Top let corner of the rectangle (y-axis).
    w : float | int
        Width of the rectangle.
    h : float | int
        Height of the rectangle.
    label : Optional[str]
        A text label that is placed at the top left corner of the
        rectangle.

    """
    pt1 = int(l), int(t)
    pt2 = int(r), int(b)
    cv.rectangle(image, pt1, pt2, color, thickness)
    if label is not None:
        text_size = cv.getTextSize(
            label, cv.FONT_HERSHEY_PLAIN, 1, thickness)

        center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
        pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + text_size[0][1]
        cv.rectangle(image, pt1, pt2, color, -1)
        cv.putText(image, label, center, cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), thickness)
        # cv.imshow("frame", image)

