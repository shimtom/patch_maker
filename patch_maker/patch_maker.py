# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import numpy as np
from PIL import Image
from three_dimensional_image import ThreeDimensionalSlicer

Logger = logging.getLogger(__name__)


class PatchMaker3D:
    def __init__(self, patch_size, logger=Logger):
        self._patch_maker = PatchMaker(patch_size)
        self._logger = logger

    def get_patch(self, images, x, y, z):
        slicer = ThreeDimensionalSlicer(images)
        return self._get_patch(slicer, x, y, z)

    def get_patches(self, images):
        slicer = ThreeDimensionalSlicer(images)

        depth, height, width = slicer.size

        self._check(depth, height, width)
        slicer.optimize()

        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    yield self._get_patch(slicer, x, y, z)

    def get_next_patches_memory(self, images, n, save_dir):
        slicer = ThreeDimensionalSlicer(images)
        depth, height, width = slicer.size
        self._check(depth, height, width)

        self._logger.debug('save xy slices')
        for z, slice_xy in enumerate(slicer.get_slices_xy(convert_image=False)):
            np.save(os.path.join(save_dir, 'xy-%d' % z), np.array(self._patch_maker.get_patches(slice_xy)))
        slicer.close_xy()

        self._logger.debug('save zx slices')
        for y, slice_zx in enumerate(slicer.get_slices_zx(convert_image=False)):
            np.save(os.path.join(save_dir, 'zx-%d' % y), np.array(self._patch_maker.get_patches(slice_zx)))
        slicer.close_zx()

        self._logger.debug('save yz slices')
        for x, slice_yz in enumerate(slicer.get_slices_yz(convert_image=False)):
            np.save(os.path.join(save_dir, 'yz-%d' % x), np.array(self._patch_maker.get_patches(slice_yz)))
        slicer.close_yz()

        slicer.close()

        patches = []
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    if len(patches) >= n:
                        yield patches
                        patches = []
                    patch = list(map(lambda name: np.load(os.path.join(save_dir, name)),
                                     ['xy-%d.npy' % z, 'zx-%d.npy' % y, 'yz-%d.npy' % x]))
                    patches.append(self._make_patch(*patch))

        yield patches

    def get_next_patches(self, images, n):
        slicer = ThreeDimensionalSlicer(images)

        depth, height, width = slicer.size

        self._check(depth, height, width)
        slicer.optimize()

        patches = []
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    if len(patches) >= n:
                        yield patches
                        patches = []
                    patches.append(self._get_patch(slicer, x, y, z))

        yield patches

    def _check(self, depth, height, width):
        half_size = self._patch_maker.patch_size // 2
        if width < half_size or height < half_size or depth < half_size:
            raise ValueError('IllegalArgumentError: '
                             'depth(%d), height(%d) and width(%d) must be more than twice as big as patch_size(%d).' % (
                                 depth, height, width, self._patch_maker.patch_size))

    def _make_patch(self, patch_xy, patch_zx, patch_yz):
        patch = np.array([patch_xy, patch_zx, patch_yz])
        patch = np.transpose(patch, axes=(1, 2, 0, 3))
        patch = patch.reshape([self._patch_maker.patch_size, self._patch_maker.patch_size, -1])
        return patch

    def _get_patch(self, slicer, x, y, z):
        depth, height, width = slicer.size

        if x < 0 or x > width or y < 0 or y > height or z < 0 or z > depth:
            raise ValueError('IndexOutOfBoundsError: (x, y, z)=(%d,%d,%d) must be within (%d, %d, %d)' % (
                x, y, z, width, height, depth))

        slice_xy, slice_zx, slice_yz = slicer.get_slice(x, y, z, immediately=True, convert_image=False)
        patch_xy = self._patch_maker.get_patch(slice_xy, x, y)
        patch_zx = self._patch_maker.get_patch(slice_zx, z, x)
        patch_yz = self._patch_maker.get_patch(slice_yz, y, z)

        return self._make_patch(patch_xy, patch_zx, patch_yz)


class PatchMaker:
    def __init__(self, patch_size):
        if patch_size % 2 != 0:
            raise ValueError('IllegalArgumentError: patch_size(%d) must be an even number.' % patch_size)

        self._patch_size = patch_size
        self._half_size = patch_size // 2

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

    def generate_get_next_patches(self, image, n):
        array = _convert_to_array(image)
        height, width = array.shape[:2]
        if width < self.patch_size // 2 or height < self.patch_size // 2:
            raise ValueError(
                'IllegalArgumentError: patch_size(%d) is more than twice as big as height(%d) and width(%d).' % (
                    self.patch_size, height, width))

        patches = []
        for y in range(height):
            for x in range(width):
                if len(patches) >= n:
                    yield patches
                    patches = []
                patches.append(self._get_patch(array, x, y))

        yield patches

    def get_n_patches(self, image, x, y, n):
        array = _convert_to_array(image)
        height, width = array.shape[:2]
        if width < self.patch_size // 2 or height < self.patch_size // 2:
            raise ValueError(
                'IllegalArgumentError: patch_size(%d) is more than twice as big as height(%d) and width(%d).' % (
                    self.patch_size, height, width))
        if n <= 0:
            return []

        patches = [None for _ in range(n)]
        count = 0
        for _y in range(y, height):
            for _x in range(x, width):
                if count == n:
                    return patches
                patches[count] = self._get_patch(array, _x, _y)
                count += 1
        return patches

    def get_patch(self, image, x, y):
        return self._get_patch(_convert_to_array(image), x, y)

    def _get_patch(self, array, x, y):
        height, width = array.shape[:2]
        if width < self.patch_size // 2 or height < self.patch_size // 2:
            raise ValueError(
                'IllegalArgumentError: patch_size(%d) is more than twice as big as height(%d) and width(%d).' % (
                    self.patch_size, height, width))
        xlim = [x - self._half_size, x + self._half_size]
        ylim = [y - self._half_size, y + self._half_size]

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
            xlim[0] = self._half_size - x

            l = array[ylim[0]:ylim[1], xlim[0] - 1::-1]
            r = array[ylim[0]:ylim[1], :xlim[1]]
            return np.concatenate((l, r), axis=1)
        # right
        if right and ycenter:
            xlim[1] = width - (x + self._half_size - width)
            l = array[ylim[0]:ylim[1], xlim[0]:]
            r = array[ylim[0]:ylim[1], width - 1:xlim[1] - 1:-1]
            return np.concatenate((l, r), axis=1)
        # up
        if up and xcenter:
            ylim[0] = self._half_size - y
            u = array[ylim[0] - 1::-1, xlim[0]:xlim[1]]
            d = array[:ylim[1], xlim[0]:xlim[1]]
            return np.concatenate((u, d))
        # down
        if down and xcenter:
            ylim[1] = height - (y + self._half_size - height)
            u = array[ylim[0]:, xlim[0]:xlim[1]]
            d = array[height - 1:ylim[1] - 1:-1, xlim[0]:xlim[1]]
            return np.concatenate((u, d))
        # left up
        if left and up:
            xlim[0] = self._half_size - x
            ylim[0] = self._half_size - y

            ul = array[ylim[0] - 1::-1, xlim[0] - 1::-1]
            dl = array[:ylim[1], xlim[0] - 1::-1]

            ur = array[ylim[0] - 1::-1, :xlim[1]]
            dr = array[:ylim[1], :xlim[1]]

            l = np.concatenate((ul, dl))
            r = np.concatenate((ur, dr))

            return np.concatenate((l, r), axis=1)
        # left down
        if left and down:
            xlim[0] = self._half_size - x
            ylim[1] = height - (y + self._half_size - height)
            ul = array[ylim[0]:, xlim[0] - 1::-1]
            dl = array[height - 1:ylim[1] - 1:-1, xlim[0] - 1::-1]

            ur = array[ylim[0]:, :xlim[1]]
            dr = array[height - 1:ylim[1] - 1:-1, :xlim[1]]
            l = np.concatenate((ul, dl))
            r = np.concatenate((ur, dr))
            return np.concatenate((l, r), axis=1)

        # right up
        if right and up:
            xlim[1] = width - (x + self._half_size - width)
            ylim[0] = self._half_size - y
            ul = array[ylim[0] - 1::-1, xlim[0]:]
            dl = array[:ylim[1], xlim[0]:]

            ur = array[ylim[0] - 1::-1, width - 1:xlim[1] - 1:-1]
            dr = array[:ylim[1], width - 1:xlim[1] - 1:-1]
            l = np.concatenate((ul, dl))
            r = np.concatenate((ur, dr))
            return np.concatenate((l, r), axis=1)

        # right down
        if right and down:
            xlim[1] = width - (x + self._half_size - width)
            ylim[1] = height - (y + self._half_size - height)
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
