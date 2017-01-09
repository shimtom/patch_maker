# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

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


class PatchMaker2:
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


# def patches(image, patch_size):
#     height, width = image.size
#     get_patch = make_memory_efficiently_patch_getter(image, patch_size)
#
#     for x in range(width):
#         for y in range(height):
#             yield get_patch(x, y)


# def make_memory_efficiently_patch_getter(image, patch_size):
#     """
#     入力画像から指定された座標のパッチを取得する関数を作成する。
#     :param image:
#     :param patch_size:
#     :return:
#     """
#     half_size = patch_size // 2
#     image_array = _convert_to_array(image)
#     height, width, ch = image_array.shape
#
#     def get_patch(x, y):
#         xlim = [x - half_size, x + half_size]
#         ylim = [y - half_size, y + half_size]
#
#         left = xlim[0] < 0
#         right = width <= xlim[1]
#         xcenter = not left and not right
#         up = ylim[0] < 0
#         down = height <= ylim[1]
#         ycenter = not up and not down
#
#         # center
#         if xcenter and ycenter:
#             return image_array[ylim[0]:ylim[1], xlim[0]:xlim[1]]
#         # left
#         if left and ycenter:
#             xlim[0] = half_size - x
#
#             l = image_array[ylim[0]:ylim[1], xlim[0] - 1::-1]
#             r = image_array[ylim[0]:ylim[1], :xlim[1]]
#             return np.concatenate((l, r), axis=1)
#         # right
#         if right and ycenter:
#             xlim[1] = width - (x + half_size - width)
#             l = image_array[ylim[0]:ylim[1], xlim[0]:]
#             r = image_array[ylim[0]:ylim[1], width - 1:xlim[1] - 1:-1]
#             return np.concatenate((l, r), axis=1)
#         # up
#         if up and xcenter:
#             ylim[0] = half_size - y
#             u = image_array[ylim[0] - 1::-1, xlim[0]:xlim[1]]
#             d = image_array[:ylim[1], xlim[0]:xlim[1]]
#             return np.concatenate((u, d))
#         # down
#         if down and xcenter:
#             ylim[1] = height - (y + half_size - height)
#             u = image_array[ylim[0]:, xlim[0]:xlim[1]]
#             d = image_array[height - 1:ylim[1] - 1:-1, xlim[0]:xlim[1]]
#             return np.concatenate((u, d))
#         # left up
#         if left and up:
#             xlim[0] = half_size - x
#             ylim[0] = half_size - y
#
#             ul = image_array[ylim[0] - 1::-1, xlim[0] - 1::-1]
#             dl = image_array[:ylim[1], xlim[0] - 1::-1]
#
#             ur = image_array[ylim[0] - 1::-1, :xlim[1]]
#             dr = image_array[:ylim[1], :xlim[1]]
#
#             l = np.concatenate((ul, dl))
#             r = np.concatenate((ur, dr))
#
#             return np.concatenate((l, r), axis=1)
#         # left down
#         if left and down:
#             xlim[0] = half_size - x
#             ylim[1] = height - (y + half_size - height)
#             ul = image_array[ylim[0]:, xlim[0] - 1::-1]
#             dl = image_array[height - 1:ylim[1] - 1:-1, xlim[0] - 1::-1]
#
#             ur = image_array[ylim[0]:, :xlim[1]]
#             dr = image_array[height - 1:ylim[1] - 1:-1, :xlim[1]]
#             l = np.concatenate((ul, dl))
#             r = np.concatenate((ur, dr))
#             return np.concatenate((l, r), axis=1)
#
#         # right up
#         if right and up:
#             xlim[1] = width - (x + half_size - width)
#             ylim[0] = half_size - y
#             ul = image_array[ylim[0] - 1::-1, xlim[0]:]
#             dl = image_array[:ylim[1], xlim[0]:]
#
#             ur = image_array[ylim[0] - 1::-1, width - 1:xlim[1] - 1:-1]
#             dr = image_array[:ylim[1], width - 1:xlim[1] - 1:-1]
#             l = np.concatenate((ul, dl))
#             r = np.concatenate((ur, dr))
#             return np.concatenate((l, r), axis=1)
#
#         # right down
#         if right and down:
#             xlim[1] = width - (x + half_size - width)
#             ylim[1] = height - (y + half_size - height)
#             ul = image_array[ylim[0]:, xlim[0]:]
#             dl = image_array[height - 1:ylim[1] - 1:-1, xlim[0]:]
#
#             ur = image_array[ylim[0]:, width - 1:xlim[1] - 1:-1]
#             dr = image_array[height - 1:ylim[1] - 1:-1, width - 1:xlim[1] - 1:-1]
#             l = np.concatenate((ul, dl))
#             r = np.concatenate((ur, dr))
#             return np.concatenate((l, r), axis=1)
#
#     return get_patch


# def make_patch_getter(image, patch_size):
#     extended_image = extend_image_array(image, patch_size)
#
#     def get_patch(x, y):
#         return extended_image[y:y + patch_size, x:x + patch_size]
#
#     return get_patch
#
#
# def extend_image_array(array, patch_size):
#     height, width, ch = array.shape
#     half_size = patch_size // 2
#
#     c = array
#
#     u = array[half_size - 1::-1, :]
#     d = array[height:height - half_size - 1:-1, :]
#     l = array[:, half_size - 1::-1]
#     r = array[:, width:width - half_size - 1:-1]
#
#     ul = array[half_size - 1::-1, half_size - 1::-1]
#     ur = array[half_size - 1::-1, width:width - half_size - 1:-1]
#     dl = array[height:height - half_size - 1:-1, half_size - 1::-1]
#     dr = array[height:height - half_size - 1:-1, width:width - half_size - 1:-1]
#
#     r_l = np.concatenate((ul, l))
#     r_l = np.concatenate((r_l, dl))
#
#     r_c = np.concatenate((u, c))
#     r_c = np.concatenate((r_c, d))
#
#     r_r = np.concatenate((ur, r))
#     r_r = np.concatenate((r_r, dr))
#
#     result = np.concatenate((r_l, r_c), axis=1)
#     result = np.concatenate((result, r_r), axis=1)
#
#     return result
#
#
# def extend_image(image, patch_size):
#     result = extend_image_array(_convert_to_array(image), patch_size)
#     return _convert_to_image(result)
#
#
# def enclose_patch(image, patch_size, x, y):
#     """入力画像の指定された座標のパッチとなる部分を塗りつぶす。
#
#     :param image:
#     :param patch_size:
#     :param x:
#     :param y:
#     :return:
#     """
#     array = _convert_to_array(image)
#     ch = array.shape[2]
#     gray = [128]
#     yresource = np.array(gray * patch_size * ch).reshape((patch_size, 1, ch))
#     xresource = np.array(gray * patch_size * ch).reshape((1, patch_size, ch))
#
#     array[y:y + patch_size, x:x + 1] = yresource
#     array[y:y + patch_size, x + patch_size:x + patch_size + 1] = yresource
#     array[y:y + 1, x: x + patch_size] = xresource
#     array[y + patch_size:y + patch_size + 1, x:x + patch_size] = xresource
#
#     return _convert_to_image(array)


def make_data(input, label, patch_size, make_label):
    """入力画像からパッチを切り出しラベル画像を参考にラベルをつける。"""
    if not np.array_equal(np.array(input.size), np.array(label.size)):
        raise ValueError('differ label image size(%s) from input image size(%s)' % (str(input.size), str(label.size)))

    patch_maker = PatchMaker(patch_size)
    target_array = _convert_to_array(label)

    height, width = input.size

    data = []
    for x in range(width):
        for y in range(height):
            patch = patch_maker.get_patch(input, x, y)
            label = make_label(target_array[y][x])
            data.append((np.array(patch), np.array(label)))

    return np.array(data)


def make_data_set(input_array, target_array, cond, patch_size, indicate=False):
    """
    その内削除
    :param input_array:
    :param target_array:
    :param cond:
    :param patch_size:
    :param indicate:
    :return:
    """
    # def make_label(patch, x, y):
    #     cy,cx = patch.shape[0] //2, patch.shape[1] //2
    #     if np.array_equal(patch[cy][cx], np.array((0,0,0))):
    #         return [1, 0]
    #     else:
    #         return [0, 1]

    if not np.array_equal(input_array.shape[0], target_array.shape[0]) or not np.array_equal(input_array.shape[1],
                                                                                             input_array.shape[1]):
        raise ValueError(
            'differ target image size from input image size' + str(input_array.shape) + ' ' + str(input_array.shape))

    height, width, ch = input_array.shape
    get_patch = make_memory_efficiently_patch_getter(input_array, patch_size)
    ex_input = None
    ex_label = None
    if indicate:
        ex_input = extend_image_array(input_array, patch_size)
        ex_label = extend_image_array(target_array, patch_size)

    inputs = []
    labels = []

    for x in range(width):
        inputs2 = []
        i = 0
        for y in range(height):
            patch = get_patch(x, y)
            if cond(target_array[y][x]):
                inputs.append(patch)
                labels.append(np.array([1, 0]))
                i += 1
                if indicate:
                    enclose_patch(ex_input, patch_size, x, y)
                    enclose_patch(ex_label, patch_size, x, y)
            else:
                inputs2.append(patch)
        random.shuffle(inputs2)
        inputs = inputs + inputs2[:i]
        labels = labels + [[0, 1]] * i

    return inputs, labels, ex_input, ex_label


def main_make_data_set(images, labels, results):
    th = np.array((0, 0, 0))
    color = 'RGB'
    patch_size = 32
    for i, paths in enumerate(zip(images, labels, results)):
        print(i)
        image_array = _convert_to_array(Image.open(paths[0]).convert(color))
        label_array = _convert_to_array(Image.open(paths[1]).convert(color))
        inputs, labels, ex_input, ex_label = make_data_set(image_array, label_array, lambda c: np.array_equal(c, th),
                                                           patch_size,
                                                           True)
        np.save(paths[2] + '.input', inputs)
        np.save(paths[2] + '.label', labels)
        _convert_to_image(ex_input).save(paths[2] + '.input.jpg')
        _convert_to_image(ex_label).save(paths[2] + '.label.jpg')


if __name__ == '__main__':
    label_base = '../images/label/'
    image_base = '../images/data/'
    result_base = '../images/result'
    images = [image_base + 'image1.tif', image_base + 'image3.png', image_base + 'image4.png',
              image_base + 'image5.png', image_base + 'image6.png']
    labels = [label_base + 'image1.tif', label_base + 'image3.png', label_base + 'image4.png',
              label_base + 'image5.png', label_base + 'image6.png']
    results = [result_base + 'image1', result_base + 'image3', result_base + 'image4', result_base + 'image5',
               result_base + 'image6']
    main_make_data_set(images=images, labels=labels, results=results)
