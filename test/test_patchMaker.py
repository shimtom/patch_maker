from unittest import TestCase, skip
from patch_maker import generate_patch, generate_patches
from patch_maker import Padding
from PIL import Image
import numpy as np


class TestPatchMaker(TestCase):
    def setUp(self):
        array = np.arange(20 * 20 * 1).reshape([20, 20]) % 255
        test_gray_image = Image.fromarray(array.astype(np.uint8))
        array = np.arange(20 * 20 * 3).reshape([20, 20, -1]) % 255
        test_rgb_image = Image.fromarray(array.astype(np.uint8))
        array = np.arange(20 * 20 * 3).reshape([20, 20, -1]) % 255
        test_rgba_image = Image.fromarray(array.astype(np.uint8))
        self._test_images = [test_gray_image, test_rgb_image, test_rgba_image]

        self._points = [(0, 0), (19, 19), (0, 19), (19, 0)]
        self._points += [(10, 10), (10, 0), (10, 19), (0, 10), (19, 10)]

        self._sizes = [(1, 1), (20, 20), (1, 20), (20, 1)]

        self._strides = [(1, 1), (20, 20), (21, 21)]
        self._paddings = [Padding.VALID, Padding.MIRROR, Padding.SAME]

        self._failure_points = [(-1, -1), (-1, 0), (0, -1)]
        self._failure_points += [(21, 21), (21, 20), (20, 21)]
        self._failure_points += [(21, -1), (21, 0), (20, -1)]
        self._failure_points += [(-1, 21), (-1, 20), (0, 21)]
        self._failure_points += [(10, -1), (10, 21), (-1, 10), (21, 10)]

        self._failure_sizes = [(-1, -1), (-1, 0), (0, -1)]
        self._failure_sizes += [(21, -1), (21, 0), (21, -1)]
        self._failure_sizes += [(-1, 21), (-1, 20), (0, 21)]
        self._failure_sizes += [(21, 21), (21, 20), (21, 20)]
        self._failure_sizes += [(0, 0)]

    def test_generate_patch(self):
        """画像から指定された条件でパッチを切り出すメソッドをテストする."""
        # 画像サイズより小さいパッチサイズを使用
        def valid(point, size, width, height):
            x, y = point
            pw, ph = size
            x1, x2 = max(min(x - pw // 2, width),
                         0), max(min(x + pw - pw // 2, width), 0)
            y1, y2 = max(min(y - ph // 2, height),
                         0), max(min(y + ph - ph // 2, height), 0)
            return (x2 - x1, y2 - y1)

        for image in self._test_images:
            for point in self._points:
                for size in self._sizes:
                    for padding in self._paddings:
                        if padding == Padding.VALID:
                            answer = valid(
                                point, size, image.width, image.height)
                        else:
                            answer = size
                        message = 'point: %s, size: %s, padding: %s, answer: %s' % (
                            str(point), str(size), str(padding), str(answer))

                        try:
                            patch = generate_patch(
                                image, point, size, padding, True)
                            array = generate_patch(
                                image, point, size, padding, False)
                        except Exception as e:
                            self.fail('%s\n%s' % (str(e), message))

                        self.assertEqual(patch.size, answer, msg=message)
                        self.assertEqual(patch.mode, image.mode, msg=message)
                        self.assertEqual(
                            patch, Image.fromarray(array), msg=message)
                        self.assertTrue(np.array_equal(
                            array, np.array(array)), msg=message)
                        # TODO: ピクセルごとに値の確認

    def test_generate_patch_failing(self):
        for point in self._failure_points:
            for size in self._sizes:
                for padding in self._paddings:
                    message = 'point: %s, size: %s' % (str(point), str(size))
                    for image in self._test_images:
                        with self.assertRaises(ValueError):
                            generate_patch(image, point, size,
                                           padding, to_image=True)
                        with self.assertRaises(ValueError):
                            generate_patch(image, point, size,
                                           padding, to_image=False)

        for point in self._points:
            for size in self._failure_sizes:
                for test_padding in self._paddings:
                    message = 'point: %s, size: %s' % (str(point), str(size))
                    for image in self._test_images:
                        with self.assertRaises(ValueError, msg=message):
                            generate_patch(image, point, size,
                                           test_padding, to_image=True)
                        with self.assertRaises(ValueError):
                            generate_patch(image, point, size,
                                           test_padding, to_image=False)

        for point in self._points:
            for size in self._sizes:
                for test_padding in ['', None, 0, 0.1]:
                    for image in self._test_images:
                        with self.assertRaises(ValueError):
                            generate_patch(image, point, size,
                                           test_padding, to_image=True)
                        with self.assertRaises(ValueError):
                            generate_patch(image, point, size,
                                           test_padding, to_image=False)

    def test_generate_patches(self):
        """画像から指定された条件で、画像から連続してパッチを切り出すメソッドをテストする."""

        for image in self._test_images:
            for size in self._sizes:
                for strides in self._strides:
                    for padding in [Padding.SAME, Padding.MIRROR]:
                        try:
                            for patch in generate_patches(image, size, strides, padding=padding, to_image=True):
                                self.assertEqual(size, patch.size)
                                self.assertEqual(image.mode, patch.mode)
                                # TODO: ピクセルごとに値の確認
                        except Exception as e:
                            self.fail('%s\n' % (str(e)))
    @skip
    def test_generate_patches_failing(self):
        failure_strides = [-1, 0.1, 0]
        failure_paddings = ['', None, 0, 0.1]
        # check size
        for size in self._failure_sizes:
            for padding in self._paddings:
                for strides in self._strides:
                    for image in self._test_images:
                        gnenerator = generate_patches(
                            image, size, strides, padding=padding, to_image=True)
                        while True:
                            try:
                                with self.assertRaises(ValueError, "invalid size %s" % str(size)):
                                    next(gnenerator)
                            except StopIteration:
                                break
                        gnenerator = generate_patches(
                            image, size, strides, padding=padding, to_image=True)
                        while True:
                            try:
                                with self.assertRaises(ValueError, "invalid size %s" % str(size)):
                                    next(gnenerator)
                            except StopIteration:
                                break

        # check strides
        for size in self._sizes:
            for strides in failure_strides:
                for padding in self._paddings:
                    for image in self._test_images:
                        gnenerator = generate_patches(
                            image, size, strides, padding=padding, to_image=True)
                        while True:
                            try:
                                with self.assertRaises(ValueError, "invalid strides %s" % str(strides)):
                                    next(gnenerator)
                            except StopIteration:
                                break
                        gnenerator = generate_patches(
                            image, size, strides, padding=padding, to_image=True)
                        while True:
                            try:
                                with self.assertRaises(ValueError, "invalid strides %s" % str(strides)):
                                    next(gnenerator)
                            except StopIteration:
                                break

        # check padding
        for size in self._sizes:
            for strides in self._strides:
                for padding in failure_paddings:
                    for image in self._test_images:
                        gnenerator = generate_patches(
                            image, size, strides, padding=padding, to_image=True)
                        while True:
                            try:
                                with self.assertRaises(ValueError, "invalid padding %s" % str(padding)):
                                    next(gnenerator)
                            except StopIteration:
                                break
                        gnenerator = generate_patches(
                            image, size, strides, padding=padding, to_image=True)
                        while True:
                            try:
                                with self.assertRaises(ValueError, "invalid padding %s" % str(padding)):
                                    next(gnenerator)
                            except StopIteration:
                                break


def valid(point, size, width, height):
    x, y = point
    pw, ph = size
    x1, x2 = max(min(x - pw // 2, width),
                 0), max(min(x + pw - pw // 2, width), 0)
    y1, y2 = max(min(y - ph // 2, height),
                 0), max(min(y + ph - ph // 2, height), 0)
    return (x2 - x1, y2 - y1)
