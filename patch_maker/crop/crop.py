import numpy as np
from enum import Enum


class Padding(Enum):
    MIRROR = 0
    SAME = 1
    VALID = 2


class Crop:
    """画像配列から指定された領域を切り出す."""

    def __init__(self, padding=Padding.MIRROR):
        _check_padding(padding)
        self._padding = padding

    def holizontal(self, array, x1, x2):
        _check_array(array)
        height, width = array.shape[:2]
        if not (- width <= x1 < 2 * width and - width <= x2 < 2 * width):
            raise ValueError('can\'t mirror (x1, x2)=%s' % str((x1, x2)))

        if x1 >= x2:
            shape = [array.shape[0], 0] + list(array.shape[2:])
            return np.array([], dtype=array.dtype).reshape(shape)
        center = array[:, min(max(x1, 0), width):max(min(x2, width), 0)]
        if self._padding == Padding.VALID:
            return center
        if self._padding == Padding.MIRROR:
            source = array
        elif self._padding == Padding.SAME:
            source = np.zeros(
                (height, width + max(0, 0 - x1) - min(0, width - x2)) + array.shape[2:])
        left = np.flipud(source[:, abs(min(0, x2)):abs(min(0, x1))])
        right = np.flipud(
            source[:, min(2 * width - x2, width):min(2 * width - x1, width)])

        return np.concatenate((left, center, right), axis=1).astype(array.dtype)

    def vertical(self, array, y1, y2):
        _check_array(array)
        height, width = array.shape[:2]
        if not(- height <= y1 < 2 * height and - height <= y2 < 2 * height):
            raise ValueError('can\'t mirror (y1, y2)=%s' % str((y1, y2)))

        if y1 >= y2:
            shape = [0] + list(array.shape[1:])
            return np.array([], dtype=array.dtype).reshape(shape)
        center = array[min(max(y1, 0), height):max(min(y2, height), 0), :]
        if self._padding == Padding.VALID:
            return center
        if self._padding == Padding.MIRROR:
            source = array
        elif self._padding == Padding.SAME:
            source = np.zeros(
                (height + max(0, - y1) - min(0, height - y2), width) + array.shape[2:])
        up = np.flipud(source[abs(min(0, y2)):abs(min(0, y1)), :])
        down = np.flipud(
            source[min(2 * height - y2, height):min(2 * height - y1, height), :])

        return np.concatenate((up, center, down)).astype(array.dtype)

    def center(self, array, box):
        _check_array(array)
        _check_box(box)
        x1, y1, x2, y2 = box
        return self.vertical(self.holizontal(array, x1, x2), y1, y2)


def _check_array(array):
    if not isinstance(array, np.ndarray):
        raise ValueError('invalid array %s' % type(array))
    if len(array.shape) < 2:
        raise ValueError('invalid array shape %s.' % (str(array.shape)))


def _check_box(box):
    if not (isinstance(box, tuple) or isinstance(box, list)):
        raise ValueError('invalid box %s' % type(box))
    if len(box) != 4:
        raise ValueError('invalid box %s' % str(box))


def _check_padding(padding):
    if padding not in Padding:
        raise ValueError('invalid padding %s' % str(padding))
