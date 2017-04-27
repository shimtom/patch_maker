# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import logging
import os

import numpy as np
from PIL import Image, ImageOps
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

    def get_patches_memory(self, images, save_dir, gray_scale=False):
        slicer = ThreeDimensionalSlicer(images)
        depth, height, width = slicer.size
        self._check(depth, height, width)

        if not os.path.isdir(save_dir):
            self._logger.debug(' make dir %s' % save_dir)
            os.makedirs(save_dir)

        def convert_to_gray(image):
            if gray_scale:
                return ImageOps.grayscale(image)
            return image

        self._logger.debug('save xy slices')
        for z, slice_xy in enumerate(slicer.get_slices_xy(convert_image=False)):
            np.save(os.path.join(save_dir, 'xy-%d' % z),
                    np.array(self._patch_maker.get_patches(convert_to_gray(slice_xy))))
        slicer.close_xy()

        self._logger.debug('save zx slices')
        for y, slice_zx in enumerate(slicer.get_slices_zx(convert_image=False)):
            np.save(os.path.join(save_dir, 'zx-%d' % y),
                    np.array(self._patch_maker.get_patches(convert_to_gray(slice_zx))))
        slicer.close_zx()

        self._logger.debug('save yz slices')
        for x, slice_yz in enumerate(slicer.get_slices_yz(convert_image=False)):
            np.save(os.path.join(save_dir, 'yz-%d' % x),
                    np.array(self._patch_maker.get_patches(convert_to_gray(slice_yz))))
        slicer.close_yz()

        slicer.close()

        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    slice_xy, slice_zx, slice_yz = list(map(lambda name: np.load(os.path.join(save_dir, name)),
                                                            ['xy-%d.npy' % z, 'zx-%d.npy' % y, 'yz-%d.npy' % x]))
                    patch = self._make_patch(slice_xy[y][x], slice_zx[x][z], slice_yz[z][y])
                    yield patch
                    del slice_xy
                    del slice_zx
                    del slice_yz
                    gc.collect()

    def get_next_patches_memory(self, images, n, save_dir, gray_scale=False):
        slicer = ThreeDimensionalSlicer(images)
        depth, height, width = slicer.size
        self._check(depth, height, width)

        if not os.path.isdir(save_dir):
            self._logger.debug(' make dir %s' % save_dir)
            os.makedirs(save_dir)

        def convert_to_gray(image):
            if gray_scale:
                return ImageOps.grayscale(image)
            return image

        self._logger.debug('save xy slices')
        for z, slice_xy in enumerate(slicer.get_slices_xy(convert_image=False)):
            np.save(os.path.join(save_dir, 'xy-%d' % z),
                    np.array(self._patch_maker.get_patches(convert_to_gray(slice_xy))))
        slicer.close_xy()

        self._logger.debug('save zx slices')
        for y, slice_zx in enumerate(slicer.get_slices_zx(convert_image=False)):
            np.save(os.path.join(save_dir, 'zx-%d' % y),
                    np.array(self._patch_maker.get_patches(convert_to_gray(slice_zx))))
        slicer.close_zx()

        self._logger.debug('save yz slices')
        for x, slice_yz in enumerate(slicer.get_slices_yz(convert_image=False)):
            np.save(os.path.join(save_dir, 'yz-%d' % x),
                    np.array(self._patch_maker.get_patches(convert_to_gray(slice_yz))))
        slicer.close_yz()

        slicer.close()

        patches = []
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    if len(patches) >= n:
                        yield patches
                        patches = []
                    slice_xy, slice_zx, slice_yz = list(map(lambda name: np.load(os.path.join(save_dir, name)),
                                                            ['xy-%d.npy' % z, 'zx-%d.npy' % y, 'yz-%d.npy' % x]))
                    patches.append(self._make_patch(slice_xy[y][x], slice_zx[x][z], slice_yz[z][y]))
                    del slice_xy
                    del slice_zx
                    del slice_yz
                    gc.collect()

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


def generate_patch(image, point, size, padding='MIRROR', to_image=True):
    """画像の指定された座標からパッチを切り出す.
    parameters:
    returns:
    """
    _check_image(image)
    _check_point(point, image.width, image.height)
    _check_size(size, image.width, image.height)
    _check_padding(padding)
    _check_to_image(to_image)

    patch = _generate_patch(np.array(image), point, size, padding=padding)
    return Image.fromarray(patch) if to_image else patch


def generate_patches(image, size, interval=1, padding='MIRROR', to_image=True):
    """画像からパッチを連続して切り出すジェネレータ.
    :param image: PIL.Image オブジェクト.元となる画像.
    :param size: (width, height). パッチのサイズ.
    :param interval: パッチを切り出す間隔
    :param padding: 'MIRROR','SAME','VALID'のどれか.パッチのパディングの仕方.
    :param to_image: パッチをPIL.Image オブジェクトに変換するか.Falseの場合,numpy.ndarrayオブジェクト
    :return: パッチ.
    """
    _check_image(image)
    _check_size(size, image.width, image.height)
    _check_interval(interval)
    _check_padding(padding)
    _check_to_image(to_image)
    width, height = image.size
    array = np.array(image)

    for i in range(0, width * height, interval):
        x, y = i % width, i // width
        patch = _generate_patch(array, (x, y), size, padding=padding)
        yield Image.fromarray(patch) if to_image else patch

def _check_image(image):
    if not isinstance(image, Image.Image):
        raise ValueError('image must be PIL.Image object but %s.' % str(type(image)))

def _check_point(point, width, height):
    if not (isinstance(point, tuple) or isinstance(point, list)) or len(point) != 2:
        raise ValueError('point must be 2 length tuple or list but %s' % str(point))
    if not isinstance(point[0], int) or not isinstance(point[1], int) or not(0 <= point[0] < width) or not(0 <= point[1] < height):
        raise ValueError('invalid coordinate %s' % str(point))

def _check_size(size, width, height):
    if not (isinstance(size, tuple) or isinstance(size, list)) or len(size) != 2:
        raise ValueError('size must be 2 length tuple or list but %s' % str(size))
    if not isinstance(size[0], int) or not isinstance(size[0], int) or not (0 < size[0] < width * 2) or not( 0 < size[1] < height * 2):
        raise ValueError('invalid size %s' % str(size))

def _check_interval(interval):
    if not (isinstance(interval, int)) or interval <= 0:
        raise ValueError('invalid interval %s' % str(interval))

def _check_padding(padding):
    if not (padding == 'MIRROR' or padding == 'SAME' or padding == 'VALID'):
        raise ValueError('invalid padding %s' % str(padding))

def _check_to_image(to_image):
    if not isinstance(to_image, bool):
        raise ValueError('to_image must be bool object but %s' % str(type(to_image)))

def _generate_patch(array, point, size, padding='MIRROR', to_image=True):
    # TODO: padding='SAME'とpadding='VALID'の対応
    width, height = array.shape[:2]
    patch_width = (size[0] // 2, size[0] - size[0] // 2)
    patch_height = (size[1] // 2, size[1] - size[1] // 2)

    x1, x2 = point[0] - patch_width[0], point[0] + patch_width[1]
    y1, y2 = point[1] - patch_height[0], point[1] + patch_height[1]

    left, right = x1 < 0, width < x2
    up, down = y1 < 0, height < y2
    xcenter, ycenter = not left and not right, not up and not down

    # center
    if xcenter and ycenter:
        return array[y1:y2, x1:x2]
    # left
    if left and ycenter:
        l = np.fliplr(array[y1:y2, :abs(x1)])
        r = array[y1:y2, :x2]
        return np.concatenate((array[y1:y2, 0:x1].fliplr(), r), axis=1)
    # right
    if right and ycenter:
        l = array[y1:y2, x1:]
        r = np.fliplr(array[y1:y2, 2 * width - x2 - 1:])
        return np.concatenate((l, r), axis=1)
    # up
    if up and xcenter:
        u = np.fliplr(array[:abs(y1), x1:x2])
        d = array[:y2, x1:x2]
        return np.concatenate((u, d))
    # down
    if down and xcenter:
        u = array[y1:, x1:x2]
        d = np.flipud(array[2 * height - y2 - 1:, x1:x2])
        return np.concatenate((u, d))

    if left and right and up and down:
        ul = np.fliplr(np.flipud(array[:abs(y1), :abs(x1)]))
        dl = np.fliplr(np.flipud(array[2 * height - y2 - 1:, :abs(x1)]))
        ur = np.fliplr(np.flipud(array[:abs(y1), 2 * width - x2 - 1:]))
        dr = np.fliplr(np.flipud(array[2 * height - y2 - 1:, 2 * width - x2 - 1:]))

        l = np.fliplr(array[y1:y2, :abs(x1)])
        r = np.fliplr(array[y1:y2, 2 * width - x2 - 1:])
        u = np.fliplr(array[:abs(y1), x1:x2])
        d = np.flipud(array[2 * height - y2 - 1:, x1:x2])

        c = array[:, :]

        l = np.concatenate((ul, l, dl))
        c = np.concatenate((u, c, d))
        r = np.concatenate((ur, r, dr))

        return np.concatenate((l, c, r), axis=1)

    if left and right and ycenter:
        l = np.fliplr(array[y1:y2, :abs(x1)])
        r = np.fliplr(array[y1:y2, 2 * width - x2 - 1:])
        c = array[:, :]

        return np.concatenate((l, c, r), axis=1)


    # left up
    if left and up:
        ul = np.fliplr(np.flipud(array[:abs(y1), :abs(x1)]))
        dl = np.fliplr(array[:y2, :abs(x1)])

        ur = np.flipud(array[:abs(y1), :x2])
        dr = array[:y2, :x2]

        l = np.concatenate((ul, dl))
        r = np.concatenate((ur, dr))

        if size == (25, 25) and point == (10, 10):
            print(x1, x2)
            print(y1, y2)
            print(ul.shape, dl.shape)
            print(ur.shape, dr.shape)
        return np.concatenate((l, r), axis=1)

    # left down
    if left and down:
        ul = np.fliplr(array[y1:, :abs(x1)])
        dl = np.fliplr(np.flipud(array[2 * height - y2 - 1:, :abs(x1)]))

        ur = array[y1:, :x2]
        dr = np.flipud(array[2 * height - y2 - 1:, :x2])

        l = np.concatenate((ul, dl))
        r = np.concatenate((ur, dr))
        return np.concatenate((l, r), axis=1)

    # right up
    if right and up:
        ul = np.flipud(array[:abs(y1), x1:])
        dl = array[:y2, x1:]

        ur = np.fliplr(np.flipud(array[:abs(y1), 2 * width - x2 - 1:]))
        dr = np.fliplr(array[:y2, 2 * width - x2 - 1:])

        l = np.concatenate((ul, dl))
        r = np.concatenate((ur, dr))
        return np.concatenate((l, r), axis=1)

    # right down
    if right and down:
        ul = array[y1:, x1:]
        dl = np.flipud(array[2 * height - y2 - 1:, x1:])

        ur = np.fliplr(array[y1:, 2 * width - x2 - 1:])
        dr = np.fliplr(np.flipud(array[2 * height - y2 - 1:, 2 * width - x2 - 1:]))

        l = np.concatenate((ul, dl))
        r = np.concatenate((ur, dr))
        return np.concatenate((l, r))

def _mirror_left_center(array, box):
    x1, y1, x2, y2 = box
    return np.fliplr(array[y1:y2, :abs(x1)])


MIRROR = 'MIRROR'
SAME = 'SAME'
VALID = 'VALID'

class Clip:
    def __init__(self, width, height, padding=MIRROR):
        self._width = width
        self._height = height
        self._padding = padding

    def center(self, array, box):
        x1, y1, x2, y2 = box
        return array[y1:y2, x1:x2]

    def holizontal(self, array, x1, x2):
        center = array[:, x1:x2]
        if self._padding == VALID:
            return center
        if self._padding == MIRROR:
            source = array
        elif self._padding == SAME:
            source = np.zeros_like(array)
        left = np.fliplr(source[:, :abs(min(0, x1))])
        right = np.fliplr(source[:, min(2*self._width - x2 -1, self._width):])

        return np.concatenate((left, center, right), axis=1)

    def vertical(self, array, y1, y2):
        center = array[y1:y2, :]
        if self._padding == VALID:
            return center
        if self._padding == MIRROR:
            source = array
        elif self._padding == SAME:
            source = np.zeros_like(array)
        up = np.flipud(source[:, :abs(min(0, y1))])
        down = np.flipud(source[:, min(2 * self._height - y2 - 1, self._height):])

        return np.concatenate((up, center, down))


class PatchMaker:
    def __init__(self, patch_size, logger=Logger):
        if patch_size < 0:
            raise ValueError('IllegalArgumentError: patch_size(%d) must be more than 0.')
        if patch_size % 2 != 0:
            raise ValueError('IllegalArgumentError: patch_size(%d) must be an even number.' % patch_size)

        self._patch_size = patch_size
        self._half_size = patch_size // 2
        self._logger = logger

    @property
    def patch_size(self):
        """パッチサイズを取得する。

        :return: パッチサイズ.
        """
        return self._patch_size

    def generate_patches(self, image, padding='MIRROR'):
        width, height = image.size

        if width < self.patch_size // 2 or height < self.patch_size // 2:
            raise ValueError(
                'IllegalArgumentError: patch_size(%d) is more than twice as big as height(%d) and width(%d).' % (
                    self.patch_size, height, width))

        array = _convert_to_array(image)

        for y in range(height):
            for x in range(width):
                yield self._get_patch(array, x, y)

    def generate_next_patches(self, image, n):
        width, height = image.size
        if width < self.patch_size // 2 or height < self.patch_size // 2:
            raise ValueError(
                'IllegalArgumentError: patch_size(%d) is more than twice as big as height(%d) and width(%d).' % (
                    self.patch_size, height, width))

        array = _convert_to_array(image)

        patches = []
        for y in range(height):
            for x in range(width):
                if len(patches) >= n:
                    yield patches
                    patches = []
                patches.append(self._get_patch(array, x, y))

        yield patches

    def get_patch(self, image, x, y):
        return self._get_patch(_convert_to_array(image), x, y)

    def get_flipped_patch(self, image, x, y, direction='lr'):
        if direction == 'lr':
            return np.fliplr(self.get_patch(image, x, y))
        elif direction == 'ud':
            return np.flipud(self.get_patch(image, x, y))

    def get_rotated_patch(self, image, x, y, direction='r'):
        if direction == 'r':
            np.fliplr(np.transpose(self.get_patch(image, x, y), axes=(1, 0, 2)))
        elif direction == 'l':
            np.flipud(np.transpose(self.get_patch(image, x, y), axes=(1, 0, 2)))

    def get_n_patches(self, image, x, y, n):
        width, height = image.size

        if n <= 0:
            return []
        array = _convert_to_array(image)
        patches = []
        for _y in range(y, height):
            for _x in range(x, width):
                if len(patches) == n:
                    break
                patches.append(self.get_patch(array, _x, _y))

        return patches

    def flip_left_right(self, image):
        array = _convert_to_array(image)
        return np.fliplr(array)

    def _get_patch(self, array, x, y, padding='MIRROR'):
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
            return self._check(array[ylim[0]:ylim[1], xlim[0]:xlim[1]], x, y, key='center')
        # left
        if left and ycenter:
            xlim[0] = self._half_size - x

            l = array[ylim[0]:ylim[1], xlim[0] - 1::-1]
            r = array[ylim[0]:ylim[1], :xlim[1]]
            return self._check(np.concatenate((l, r), axis=1), x, y, key='left center')
        # right
        if right and ycenter:
            xlim[1] = width - (x + self._half_size - width)
            l = array[ylim[0]:ylim[1], xlim[0]:]
            r = array[ylim[0]:ylim[1], width - 1:xlim[1] - 1:-1]
            return self._check(np.concatenate((l, r), axis=1), x, y, key='right center')
        # up
        if up and xcenter:
            ylim[0] = self._half_size - y
            u = array[ylim[0] - 1::-1, xlim[0]:xlim[1]]
            d = array[:ylim[1], xlim[0]:xlim[1]]
            return self._check(np.concatenate((u, d)), x, y, key='up center')
        # down
        if down and xcenter:
            ylim[1] = height - (y + self._half_size - height)
            u = array[ylim[0]:, xlim[0]:xlim[1]]
            d = array[height - 1:ylim[1] - 1:-1, xlim[0]:xlim[1]]
            return self._check(np.concatenate((u, d)), x, y, key='down center')
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

            return self._check(np.concatenate((l, r), axis=1), x, y, key='left up')
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
            return self._check(np.concatenate((l, r), axis=1), x, y, key='left down')

        # right up
        if right and up:
            xlim[1] = width - (x + self._half_size - width)  # ok
            ylim[0] = self._half_size - y
            ul = array[ylim[0] - 1::-1, xlim[0]:]
            dl = array[:ylim[1], xlim[0]:]  # ok

            ur = array[ylim[0] - 1::-1, width - 1:xlim[1] - 1:-1]
            dr = array[:ylim[1], width - 1:xlim[1] - 1:-1]
            l = np.concatenate((ul, dl))
            r = np.concatenate((ur, dr))
            return self._check(np.concatenate((l, r), axis=1), x, y, key='right up')

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
            return self._check(np.concatenate((l, r), axis=1), x, y, key='right down')

    def _check(self, array, x, y, key=''):
        shape = array.shape
        if shape[0] != self._patch_size or shape[1] != self._patch_size:
            raise ValueError(
                '%s (%d, %d) array shape must be (%d, %d, ...) but %s' % (
                    key, x, y, self._patch_size, self._patch_size, str(shape)))
        return array


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
