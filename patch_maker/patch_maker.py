# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image


class PatchMaker:
    def __init__(self, patch_size):
        if patch_size % 2 != 0:
            raise ValueError('IllegalArgumentError: patch_size(%d) must be an even number.' % patch_size)

        self._patch_size = patch_size
        self.half_size = patch_size // 2

    @property
    def patch_size(self):
        return self._patch_size

    def get_patches(self, image):
        array = _convert_to_array(image)
        height, width = array.shape[:2]
        if width < self.patch_size // 2 or height < self.patch_size // 2:
            raise ValueError(
                'IllegalArgumentError: patch_size(%d) is more than twice as big as height(%d) and width(%d).' % (
                    self.patch_size, height, width))

        return [self._get_patch(array, x, y) for y in range(height) for x in range(width)]

    def get_patch(self, image, x, y):
        return self._get_patch(_convert_to_array(image), x, y)

    def _get_patch(self, array, x, y):
        height, width = array.shape[:2]
        if width < self.patch_size // 2 or height < self.patch_size // 2:
            raise ValueError(
                'IllegalArgumentError: patch_size(%d) is more than twice as big as height(%d) and width(%d).' % (
                    self.patch_size, height, width))
        xlim = [x - self.half_size, x + self.half_size]
        ylim = [y - self.half_size, y + self.half_size]

        left = xlim[0] < 0
        right = width <= xlim[1]
        xcenter = not left and not right
        up = ylim[0] < 0
        down = height <= ylim[1]
        ycenter = not up and not down

        # center
        if xcenter and ycenter:
            return array[ylim[0]:ylim[1], xlim[0]:xlim[1]]
        # left
        if left and ycenter:
            xlim[0] = self.half_size - x

            l = array[ylim[0]:ylim[1], xlim[0] - 1::-1]
            r = array[ylim[0]:ylim[1], :xlim[1]]
            return np.concatenate((l, r), axis=1)
        # right
        if right and ycenter:
            xlim[1] = width - (x + self.half_size - width)
            l = array[ylim[0]:ylim[1], xlim[0]:]
            r = array[ylim[0]:ylim[1], width - 1:xlim[1] - 1:-1]
            return np.concatenate((l, r), axis=1)
        # up
        if up and xcenter:
            ylim[0] = self.half_size - y
            u = array[ylim[0] - 1::-1, xlim[0]:xlim[1]]
            d = array[:ylim[1], xlim[0]:xlim[1]]
            return np.concatenate((u, d))
        # down
        if down and xcenter:
            ylim[1] = height - (y + self.half_size - height)
            u = array[ylim[0]:, xlim[0]:xlim[1]]
            d = array[height - 1:ylim[1] - 1:-1, xlim[0]:xlim[1]]
            return np.concatenate((u, d))
        # left up
        if left and up:
            xlim[0] = self.half_size - x
            ylim[0] = self.half_size - y

            ul = array[ylim[0] - 1::-1, xlim[0] - 1::-1]
            dl = array[:ylim[1], xlim[0] - 1::-1]

            ur = array[ylim[0] - 1::-1, :xlim[1]]
            dr = array[:ylim[1], :xlim[1]]

            l = np.concatenate((ul, dl))
            r = np.concatenate((ur, dr))

            return np.concatenate((l, r), axis=1)
        # left down
        if left and down:
            xlim[0] = self.half_size - x
            ylim[1] = height - (y + self.half_size - height)
            ul = array[ylim[0]:, xlim[0] - 1::-1]
            dl = array[height - 1:ylim[1] - 1:-1, xlim[0] - 1::-1]

            ur = array[ylim[0]:, :xlim[1]]
            dr = array[height - 1:ylim[1] - 1:-1, :xlim[1]]
            l = np.concatenate((ul, dl))
            r = np.concatenate((ur, dr))
            return np.concatenate((l, r), axis=1)

        # right up
        if right and up:
            xlim[1] = width - (x + self.half_size - width)
            ylim[0] = self.half_size - y
            ul = array[ylim[0] - 1::-1, xlim[0]:]
            dl = array[:ylim[1], xlim[0]:]

            ur = array[ylim[0] - 1::-1, width - 1:xlim[1] - 1:-1]
            dr = array[:ylim[1], width - 1:xlim[1] - 1:-1]
            l = np.concatenate((ul, dl))
            r = np.concatenate((ur, dr))
            return np.concatenate((l, r), axis=1)

        # right down
        if right and down:
            xlim[1] = width - (x + self.half_size - width)
            ylim[1] = height - (y + self.half_size - height)
            ul = array[ylim[0]:, xlim[0]:]
            dl = array[height - 1:ylim[1] - 1:-1, xlim[0]:]

            ur = array[ylim[0]:, width - 1:xlim[1] - 1:-1]
            dr = array[height - 1:ylim[1] - 1:-1, width - 1:xlim[1] - 1:-1]
            l = np.concatenate((ul, dl))
            r = np.concatenate((ur, dr))
            return np.concatenate((l, r), axis=1)


# FIXME: 速度改善の為に作成したが、PatchMakerよりも遅い。速度改善が必要。
class _PatchMaker2:
    def __init__(self, height, width, ch, patch_size):
        if patch_size % 2 != 0:
            raise ValueError('IllegalArgumentError: patch_size(%d) must be an even number.' % patch_size)
        if width < patch_size // 2 or height < patch_size // 2:
            raise ValueError(
                'IllegalArgumentError: patch_size(%d) is more than twice as big as height(%d) and width(%d).' % (
                    patch_size, height, width))
        self.width = width
        self.height = height
        self.ch = ch
        self.shape = np.array([width, height, ch])
        self._patch_size = patch_size
        self.half_size = patch_size // 2

        shape = [self.height, self.width, self.patch_size, self.patch_size]
        self.x = np.fromfunction(self._make_x, shape)
        self.y = np.fromfunction(self._make_y, shape)

    @property
    def size(self):
        return self.height, self.width, self.ch

    @property
    def patch_size(self):
        return self.patch_size

    def get_patches(self, image):
        array = _convert_to_array(image)
        if np.array_equal(array.shape, self.shape):
            raise ValueError('IllegalArgumentError: input image shape %r must be equal to %r' % array.shape, self.shape)

        return array[self.y, self.x]

    def get_patch(self, image, x, y):
        array = _convert_to_array(image)
        if np.array_equal(array.shape, self.shape):
            raise ValueError('IllegalArgumentError: input image shape %r must be equal to %r' % array.shape, self.shape)

        shape = [self.patch_size, self.patch_size]
        px = np.fromfunction(lambda y_, x_: self._make_x(y, x, y_, x_), shape)
        py = np.fromfunction(lambda y_, x_: self._make_y(y, x, y_, x_), shape)
        return array[py, px]

    def _make_x(self, y, x, y_, x_):
        nx = x + x_ - self.half_size
        i = nx >= self.width
        nx[i] = (2 * self.width - 1) - nx[i]
        j = nx < 0
        nx[j] = np.abs(nx[j] + 1)
        return nx.astype(np.int64)

    def _make_y(self, y, x, y_, x_):
        ny = y + y_ - self.half_size
        i = ny >= self.height
        ny[i] = (2 * self.height - 1) - ny[i]
        j = ny < 0
        ny[j] = np.abs(ny[j] + 1)
        return ny.astype(np.int64)


def _convert_to_array(image):
    image_array = np.array(image)
    if len(image_array.shape) == 2:
        height, width = image_array.shape
        image_array = image_array.reshape((height, width, 1))
    return image_array


def _convert_to_image(array):
    if array.shape[2] == 1:
        height, width, _ = array.shape
        array = array.reshape((height, width))
    return Image.fromarray(array)
