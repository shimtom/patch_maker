from unittest import TestCase
from patch_maker import generate_patch, generate_patches
from PIL import Image
import numpy as np


class TestPatchMaker(TestCase):
    def test_constructor(self):
        array = np.arange(20 * 20 * 1).reshape([20, 20]) % 255
        test_gray_image = Image.fromarray(array.astype(np.uint8))
        array = np.arange(20 * 20 * 3).reshape([20, 20]) % 255
        test_rgb_image = Image.fromarray(array.astype(np.uint8))
        array = np.arange(20 * 20 * 3).reshape([20, 20]) % 255
        test_rgba_image = Image.fromarray(array.astype(np.uint8))
        self._test_images = [test_gray_image, test_rgb_image, test_rgba_image]

    def test_generate_patch(self):
        """画像から指定された条件でパッチを切り出すメソッドをテストする."""
        # 画像サイズより小さいパッチサイズを使用

        # 成功を確認
        # 中央, 左上, 左下, 右上, 右下, 左中央, 右中央, 下中央, 上中央
        points = [(10, 10), (0, 0), (0, 20), (20, 0), (20, 20),
                  (0, 10), (20, 10), (10, 20), (0, 10)]
        # 幅 = 高さ
        sizes = [(0, 0), (10, 10), (20, 20), (25, 25),
                 (30, 30), (35, 35), (40, 40)]
        # 幅 < 高さ
        sizes += [(0, 5), (5, 10), (15, 20), (20, 25),
                  (25, 30), (30, 35), (35, 40)]
        # 幅 > 高さ
        sizes += [(5, 0), (10, 5), (20, 15), (25, 20),
                  (30, 20), (35, 30), (40, 35)]

        test_paddings = ['MIRROR', 'SAME', 'VALID']

        for point in points:
            for size in sizes:
                for padding in test_paddings:
                    for image in self._test_images:
                        patch = generate_patch(
                            image, point, size, padding, to_image=True)
                        array = generate_patch(
                            image, point, size, padding, to_image=False)
                        self.assertEqual(patch.size, size)
                        self.assertEqual(patch.mode, image.mode)
                        self.assertEqual(patch, Image.fromarray(array))
                        self.assertTrue(np.array_equal(array, np.array(array)))
                        # TODO: ピクセルごとに値の確認

        # 失敗を確認
        failure_points = [(-1, -1), (-1, 0), (0, -1)]
        failure_points += [(21, -1), (20, -1), (21, 0)]
        failure_points += [(21, 21), (21, 20), (20, 21)]
        failure_points += [(-1, 21), (-1, 20), (0, 21)]
        failure_points += [(0.1, 0.1), (0.1, 0), (0, 0.1)]
        failure_sizes = [(-1, -1), (41, 41), (-1, 0), (0, -1),
                         (41, 40), (40, 41), (0.1, 0.1), (0.1, 0), (0, 0.1)]
        failure_paddings = ['']

        for point in failure_points:
            for size in sizes:
                for padding in test_paddings:
                    for image in self._test_images:
                        with self.assertRaises(ValueError, "invalid coordinate %s" % str(point)):
                            patch = generate_patch(
                                image, point, size, padding, to_image=True)
                        with self.assertRaises(ValueError, "invalid coordinate %s" % str(point)):
                            array = generate_patch(
                                image, point, size, padding, to_image=False)

        for point in points:
            for size in failure_sizes:
                for test_padding in test_paddings:
                    for image in self._test_images:
                        with self.assertRaises(ValueError, "invalid size %s" % str(size)):
                            patch = generate_patch(
                                image, point, size, test_padding, to_image=True)
                        with self.assertRaises(ValueError, "invalid size %s" % str(point)):
                            array = generate_patch(
                                image, point, size, test_padding, to_image=False)

        for point in points:
            for size in sizes:
                for test_padding in failure_paddings:
                    for image in self._test_images:
                        with self.assertRaises(ValueError, "invalid padding %s" % test_padding):
                            patch = generate_patch(
                                image, point, size, test_padding, to_image=True)
                        with self.assertRaises(ValueError, "invalid size %s" % test_padding):
                            array = generate_patch(
                                image, point, size, test_padding, to_image=False)

    def test_generate_patches(self):
        """画像から指定された条件で、画像から連続してパッチを切り出すメソッドをテストする."""
        # 幅 = 高さ
        sizes = [(0, 0), (10, 10), (20, 20), (25, 25),
                 (30, 30), (35, 35), (40, 40)]
        # 幅 < 高さ
        sizes += [(0, 5), (5, 10), (15, 20), (20, 25),
                  (25, 30), (30, 35), (35, 40)]
        # 幅 > 高さ
        sizes += [(5, 0), (10, 5), (20, 15), (25, 20),
                  (30, 20), (35, 30), (40, 35)]
        intervals = [0, 1, 20, 40, 1000, 1600]
        paddings = ['MIRROR', 'SAME', 'VALID']

        for size in sizes:
            for interval in intervals:
                for padding in paddings:
                    for image in self._test_images:
                        length = image.width * image.height
                        patches = []
                        for patch in generate_patches(image, size, interval, padding, to_image=True):
                            self.assertEqual(image.size, patch.size)
                            self.assertEqual(image.mode, patch.mode)
                            patches.append(patch)
                            # TODO: ピクセルごとに値の確認
                        self.assertEqual(length, len(patches))

                        count = 0
                        for i, array in enumerate(generate_patches(image, size, interval, padding, to_image=False)):
                            self.assertEqual(
                                patches[i], Image.fromarray(array))
                            self.assertTrue(np.array_equal(
                                array, np.array(patches[i])))
                            count += 1
                        self.assertEqual(length, count)

        failure_sizes = [(-1, -1), (41, 41), (-1, 0), (0, -1),
                         (41, 40), (40, 41), (0.1, 0.1), (0.1, 0), (0, 0.1)]
        failure_intervals = [-1, 0.1]
        failure_paddings = ['']

        # check size
        for size in failure_sizes:
            for padding in paddings:
                for interval in intervals:
                    for image in self._test_images:
                        gnenerator = generate_patches(
                            image, size, interval, padding, to_image=True)
                        while True:
                            try:
                                with self.assertRaises(ValueError, "invalid size %s" % str(size)):
                                    next(gnenerator)
                            except StopIteration:
                                break
                        gnenerator = generate_patches(
                            image, size, interval, padding, to_image=True)
                        while True:
                            try:
                                with self.assertRaises(ValueError, "invalid size %s" % str(size)):
                                    next(gnenerator)
                            except StopIteration:
                                break

        # check interval
        for size in sizes:
            for interval in failure_intervals:
                for padding in paddings:
                    for image in self._test_images:
                        gnenerator = generate_patches(
                            image, size, interval, padding, to_image=True)
                        while True:
                            try:
                                with self.assertRaises(ValueError, "invalid interval %s" % str(interval)):
                                    next(gnenerator)
                            except StopIteration:
                                break
                        gnenerator = generate_patches(
                            image, size, interval, padding, to_image=True)
                        while True:
                            try:
                                with self.assertRaises(ValueError, "invalid interval %s" % str(interval)):
                                    next(gnenerator)
                            except StopIteration:
                                break

        # check padding
        for size in sizes:
            for interval in intervals:
                for padding in failure_paddings:
                    for image in self._test_images:
                        gnenerator = generate_patches(
                            image, size, interval, padding, to_image=True)
                        while True:
                            try:
                                with self.assertRaises(ValueError, "invalid padding %s" % str(padding)):
                                    next(gnenerator)
                            except StopIteration:
                                break
                        gnenerator = generate_patches(
                            image, size, interval, padding, to_image=True)
                        while True:
                            try:
                                with self.assertRaises(ValueError, "invalid padding %s" % str(padding)):
                                    next(gnenerator)
                            except StopIteration:
                                break
