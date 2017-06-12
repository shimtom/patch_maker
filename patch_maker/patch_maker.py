# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
from .crop import Padding
from .crop import Crop


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


def generate_patches(image, size, strides=(1, 1), bounds=(None, None, None, None), padding=Padding.MIRROR, to_image=True):
    """画像からパッチを連続して切り出すジェネレータ.
    :param image: PIL.Image オブジェクト.元となる画像.
    :param size: (width, height). パッチのサイズ.
    :param strides: パッチを切り出す間隔.
    :param bounds: パッチを切り出す範囲.(sx, sy, ex, ey).
    :param padding: Padding.MIRROR, Padding.SAME, Padding.VALIDのどれか. パッチのパディングの仕方.
    :param to_image: パッチをPIL.Image オブジェクトに変換するか.Falseの場合,numpy.ndarrayオブジェクト
    :yield: パッチ.
    """
    _check_image(image)
    _check_size(size, image.width, image.height)
    _check_strides(strides, 2)
    _check_padding(padding)
    _check_to_image(to_image)
    width, height = image.size
    array = np.array(image).astype(np.uint8)
    sx, sy, ex, ey = bounds if bounds is not None else (0, 0, width, height)
    sx, sy = max(sx or 0, 0), max(sy or 0, 0)
    ex, ey = min(ex or width, width), min(ey or height, height)

    for y in range(sy, ey, strides[0]):
        for x in range(sx, ex, strides[1]):
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


def generate_3d_patches(images, size, strides=(1, 1, 1), padding=Padding.MIRROR):
    """画像配列の指定された座標から3dパッチを切り出す.
    :param images: PIL.Image オブジェクトのリスト.元となる画像群.
    :param size: (width, height, depth). パッチのサイズ.
    :param strides: パッチを切り出す間隔.
    :param padding: Padding.MIRROR, Padding.SAME, Padding.VALIDのどれか. パッチのパディングの仕方.
    :return: パッチ. numpy.ndarrayオブジェクト.
    """
    _check_images(images)
    _check_3d_size(size, images[0].width, images[0].height, len(images))
    _check_strides(strides, 3)
    _check_padding(padding)

    width = images[0].width
    height = images[0].height
    depth = len(images)
    arrays = np.array(images)

    for z in range(0, depth, strides[0]):
        for y in range(0, height, strides[1]):
            for x in range(0, width, strides[2]):
                yield _generate_3d_patch(arrays, (x, y, z), size, padding=padding)


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


def _check_strides(strides, dims):
    if not (isinstance(strides, tuple) or isinstance(strides, list)) or len(strides) != dims:
        raise ValueError('invalid strides %s' % str(strides))
    for stride in strides:
        if not (isinstance(stride, int)) or stride < 0:
            raise ValueError('invalid strides %s' % str(strides))


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


def main():
    import os
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('images', type=str, nargs='+', help='image. support extensions:[.png|.jpg|.tif]')
    parser.add_argument('size', type=int, nargs=2, help='patch size')
    parser.add_argument('save', type=str, help='save directory.')
    parser.add_argument('--strides', type=int, default=(1,1), nargs=2, help='strides' )
    parser.add_argument('--bounds', type=int, default=(0, 0, None, None), nargs=4, help='bounds')

    args = parser.parse_args()

    files = args.images
    size = args.size
    directory = args.save
    size = tuple(args.size)
    strides = tuple(args.strides)
    bounds = tuple(args.bounds)

    if not os.path.isdir(directory):
        print('invalid save directory: %s' % directory, file=sys.stderr)
        return

    # ファイルごとにパッチを作成し保存
    for f in files:
        if not os.path.isfile(f) or not f.split('.')[-1] in ['png', 'jpg', 'tif']:
            print('invalid file: %s' % f, file=sys.stderr)
            continue
        image = Image.open(f)
        patches = [p for p in generate_patches(image, size, strides, bounds, to_image=False)]
        patches = np.array(patches)
        name = '.'.join(f.split('/')[-1].split('.')[:-1]) + '.npz'
        np.save(os.path.join(directory, name), patches)
        image.close()
        del patches


if __name__ == '__main__':
    main()
