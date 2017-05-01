# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
from PIL import Image
from .crop import Padding
from .crop import Crop
Logger = logging.getLogger(__name__)


def generate_patch(image, point, size, padding=Padding.MIRROR, to_image=True):
    """画像の指定された座標からパッチを切り出す.
    :param image: PIL.Image オブジェクト.元となる画像.
    :param point: (x, y). パッチを切り出す座標.
    :param size: (width, height). パッチのサイズ.
    :param padding: Padding.MIRROR, Padding.SAME, Padding.VALIDのどれか. パッチのパディングの仕方.    :param to_image: パッチをPIL.Image オブジェクトに変換するか.Falseの場合,numpy.ndarrayオブジェクト
    :yield: パッチ.
    """
    _check_image(image)
    _check_point(point, image.width, image.height)
    _check_size(size, image.width, image.height)
    _check_padding(padding)
    _check_to_image(to_image)

    patch = _generate_patch(np.array(image), point, size, padding=padding)
    if to_image:
        return Image.fromarray(patch)
    else:
        return patch


def generate_patches(image, size, interval=1, padding=Padding.MIRROR, to_image=True):
    """画像からパッチを連続して切り出すジェネレータ.
    :param image: PIL.Image オブジェクト.元となる画像.
    :param size: (width, height). パッチのサイズ.
    :param interval: パッチを切り出す間隔.
    :param padding: Padding.MIRROR, Padding.SAME, Padding.VALIDのどれか. パッチのパディングの仕方.
    :param to_image: パッチをPIL.Image オブジェクトに変換するか.Falseの場合,numpy.ndarrayオブジェクト
    :yield: パッチ.
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
        if to_image:
            yield Image.fromarray(patch)
        else:
            yield patch


def generate_3d_patch(images, point, size, padding=Padding.MIRROR):
    """画像配列の指定された座標から3dパッチを切り出す.
    :param images: PIL.Image オブジェクトのリスト.元となる画像群.
    :param point: (x, y, z). パッチの中心となる座標.
    :param size: (width, height, depth). パッチのサイズ.
    :param padding: Padding.MIRROR, Padding.SAME, Padding.VALIDのどれか. パッチのパディングの仕方.
    :return: パッチ. numpy.ndarrayオブジェクト.
    """
    _check_images(images)
    width, height, depth = images[0].width, images[0].height, len(images)
    _check_3d_point(point, width, height, depth)
    _check_3d_size(size, width, height, depth)
    _check_padding(padding)

    patch = _generate_3d_patch(np.array(images), point, size, padding=padding)
    return patch


def generate_3d_patches(images, size, interval=1, padding=Padding.MIRROR):
    """画像配列の指定された座標から3dパッチを切り出す.
    :param images: PIL.Image オブジェクトのリスト.元となる画像群.
    :param size: (width, height, depth). パッチのサイズ.
    :param interval: パッチを切り出す間隔.
    :param padding: Padding.MIRROR, Padding.SAME, Padding.VALIDのどれか. パッチのパディングの仕方.
    :return: パッチ. numpy.ndarrayオブジェクト.
    """
    _check_images(images)
    _check_3d_size(size, images[0].width, images[0].height, len(images))
    _check_interval(interval)
    _check_padding(padding)

    width = images[0].width
    height = images[0].height
    depth = len(images)

    for i in range(0, depth * height * width, interval):
        x = i % (height * width) % width
        y = i % (height * width) // width
        z = i // (height * width)
        yield _generate_3d_patch(images, (x, y, z), size, padding=padding)


def _generate_3d_patch(arrays, point, size, padding):
    def reshape(array):
        h, w = array.shape[:2]
        return array.reshape((h, w, -1))

    patch_xy = _generate_patch_xy(arrays, point, size, padding)
    patch_yz = _generate_patch_xy(arrays, point, size, padding)
    patch_zx = _generate_patch_zx(arrays, point, size, padding)

    patch_xy = reshape(patch_xy)
    patch_yz = reshape(patch_yz)
    patch_zx = reshape(patch_zx)

    return np.concatenate((patch_xy, patch_yz, patch_zx), axis=-1)


def _generate_patch_xy(arrays, point, size, padding):
    z, y, x = point
    patch_depth, patch_height, patch_width = size
    x1, x2 = x - patch_width // 2, x + patch_width - patch_width // 2
    y1, y2 = y - patch_height // 2, y + patch_height - patch_height // 2

    patch = Crop(padding).center(arrays[z, :, :], (x1, y1, x2, y2))

    return patch


def _generate_patch_yz(arrays, point, size, padding):
    z, y, x = point
    patch_depth, patch_height, patch_width = size
    y1, y2 = y - patch_height // 2, y + patch_height - patch_height // 2
    z1, z2 = z - patch_depth // 2, z + patch_depth - patch_depth // 2

    patch = Crop(padding).center(arrays[:, :, x], (y1, z1, y2, z2))

    return patch


def _generate_patch_zx(arrays, point, size, padding):
    z, y, x = point
    patch_depth, patch_height, patch_width = size
    z1, z2 = z - patch_depth // 2, z + patch_depth - patch_depth // 2
    x1, x2 = x - patch_width // 2, x + patch_width - patch_width // 2

    axes = (1, 0) + \
        tuple([i + 2 for i in range(len(arrays[:, y, :].shape) - 2)])
    patch = Crop(padding).center(np.transpose(
        arrays[:, y, :], axes), (z1, x1, z2, x2))

    return patch


def _generate_patch(array, point, size, padding):
    x, y = point
    height, width = array.shape[:2]
    patch_width, patch_height = size

    x1, x2 = x - patch_width // 2, x + patch_width - patch_width // 2
    y1, y2 = y - patch_height // 2, y + patch_height - patch_height // 2

    return Crop(padding=padding).center(array, (x1, y1, x2, y2))


def _check_interval(interval):
    if not (isinstance(interval, int)) or interval <= 0:
        raise ValueError('invalid interval %s' % str(interval))


def _check_padding(padding):
    if padding not in Padding:
        raise ValueError('invalid padding %s' % str(padding))


def _check_to_image(to_image):
    if not isinstance(to_image, bool):
        raise ValueError('to_image must be bool object but %s' %
                         str(type(to_image)))


def _check_image(image):
    if not isinstance(image, Image.Image):
        raise ValueError('image must be PIL.Image object but %s.' %
                         str(type(image)))


def _check_point(point, width, height):
    if not (isinstance(point, tuple) or isinstance(point, list)) or len(point) != 2:
        raise ValueError(
            'point must be 2 length tuple or list but %s' % str(point))
    if not isinstance(point[0], int) or not isinstance(point[1], int) or not(0 <= point[0] < width) or not(0 <= point[1] < height):
        raise ValueError('invalid coordinate %s' % str(point))


def _check_size(size, width, height):
    if not (isinstance(size, tuple) or isinstance(size, list)) or len(size) != 2:
        raise ValueError(
            'size must be 2 length tuple or list but %s' % str(size))
    if not (isinstance(size[0], int) and isinstance(size[0], int)) or not (0 < size[0] <= width) or not(0 < size[1] <= height):
        raise ValueError('invalid size %s. limit (%s)' %
                         (str(size), str((width * 2, height * 2))))


def _check_images(images):
    w, h = images[0].size
    for image in images:
        _check_image(image)
        if (w, h) != image.size:
            raise ValueError('invalid images')


def _check_3d_point(point, width, height, depth):
    if not (isinstance(point, tuple) or isinstance(point, list)) or len(point) != 3:
        raise ValueError(
            'point must be 3 length tuple or list but %s' % str(point))
    if not (0 <= point[0] < width and 0 <= point[1] < height and 0 <= point[2] < depth):
        raise ValueError('invalid coordinate %s' % str(point))


def _check_3d_size(size, width, height, depth):
    if not (isinstance(size, tuple) or isinstance(size, list)) or len(size) != 3:
        raise ValueError(
            'size must be 3 length tuple or list but %s' % str(size))
    if not (0 < size[0] <= width and 0 < size[1] <= height and 0 < size[2] <= depth):
        raise ValueError('invalid size %s. limit (%s)' %
                         (str(size), str((width, height, depth))))
