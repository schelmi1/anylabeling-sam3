import os.path

import cv2
import numpy as np
import qimage2ndarray
from PyQt6 import QtGui
from PyQt6.QtGui import QImage


def qt_img_to_rgb_cv_img(qt_img, img_path=None):
    """
    Convert 8bit/16bit RGB image or 8bit/16bit Gray image to 8bit RGB image
    """
    cv_image = None
    if img_path is not None and os.path.exists(img_path):
        # Load image from path directly when possible.
        # Some files may fail to decode (unsupported/corrupt/empty); fallback below.
        raw = np.fromfile(img_path, dtype=np.uint8)
        if raw.size > 0:
            cv_image = cv2.imdecode(raw, -1)

        if cv_image is not None and cv_image.size > 0:
            if len(cv_image.shape) == 2:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
            elif cv_image.shape[2] == 4:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGB)
            else:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    if cv_image is None:
        if qt_img is None or qt_img.isNull():
            raise ValueError(f"Could not decode image from path: {img_path}")
        if (
            qt_img.format() == QImage.Format.Format_RGB32
            or qt_img.format() == QImage.Format.Format_ARGB32
            or qt_img.format() == QImage.Format.Format_ARGB32_Premultiplied
        ):
            cv_image = qimage2ndarray.rgb_view(qt_img)
        else:
            cv_image = qimage2ndarray.raw_view(qt_img)
    # To uint8
    if cv_image.dtype != np.uint8:
        cv2.normalize(cv_image, cv_image, 0, 255, cv2.NORM_MINMAX)
        cv_image = np.array(cv_image, dtype=np.uint8)
    # To RGB
    if len(cv_image.shape) == 2 or cv_image.shape[2] == 1:
        cv_image = cv2.merge([cv_image, cv_image, cv_image])
    return cv_image


def qt_img_to_cv_img(in_image):
    return qimage2ndarray.rgb_view(in_image)


def cv_img_to_qt_img(in_mat):
    return QtGui.QImage(qimage2ndarray.array2qimage(in_mat))
