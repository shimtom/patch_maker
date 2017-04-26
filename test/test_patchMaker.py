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
        test_points = [(10, 10), (0, 0), (0, 20), (20, 0),
                       (20, 20), (0, 10), (20, 10), (10, 20), (0, 10)]
        # 幅 = 高さ
        test_sizes = [(0, 0), (10, 10), (20, 20), (25, 25),
                      (30, 30), (35, 35), (40, 40)]
        # 幅 < 高さ
        test_sizes += [(0, 5), (5, 10), (15, 20), (20, 25),
                       (25, 30), (30, 35), (35, 40)]
        # 幅 > 高さ
        test_sizes += [(5, 0), (10, 5), (20, 15), (25, 20),
                       (30, 20), (35, 30), (40, 35)]

        paddings = ['MIRROR', 'SAME', 'VALID']

        for test_point in test_points:
            for test_size in test_sizes:
                for padding in paddings:
                    for test_image in self._test_images:
                        patch = generate_patch(
                            self._test_image, test_point, test_size, padding, to_image=True)
                        array = generate_patch(
                            self._test_image, test_point, test_size, padding, to_image=False)
                        self.assertEqual(patch.size, test_size)
                        self.assertEqual(patch.mode, test_image.mode)
                        self.assertEqual(patch, Image.fromarray(array))
                        self.assertTrue(np.array_equal(array, np.array(array)))

        # 失敗を確認
        failure_test_points = [(-1, -1), (-1, 0), (0, -1)]
        failure_test_points += [(21, -1), (20, -1), (21, 0)]
        failure_test_points += [(21, 21), (21, 20), (20, 21)]
        failure_test_points += [(-1, 21), (-1, 20), (0, 21)]
        failure_test_points += [(0.1, 0.1), (0.1, 0), (0, 0.1)]
        failure_test_sizes = [(-1, -1), (41, 41), (-1, 0), (0, -1),
                              (41, 40), (40, 41), (0.1, 0.1), (0.1, 0), (0, 0.1)]
        failure_paddings = ['']

        for test_point in failure_test_points:
            for test_size in test_sizes:
                for padding in paddings:
                    for test_image in self._test_images:
                        with self.assertRaises(ValueError, "invalid coordinate %s" % str(test_point)):
                            patch = generate_patch(
                                self._test_image, test_point, test_size, padding, to_image=True)
                        with self.assertRaises(ValueError, "invalid coordinate %s" % str(test_point)):
                            array = generate_patch(
                                self._test_image, test_point, test_size, padding, to_image=False)

        for test_point in test_points:
            for test_size in failure_test_sizes:
                for padding in paddings:
                    for test_image in self._test_images:
                        with self.assertRaises(ValueError, "invalid size %s" % str(test_size)):
                            patch = generate_patch(
                                self._test_image, test_point, test_size, padding, to_image=True)
                        with self.assertRaises(ValueError, "invalid size %s" % str(test_point)):
                            array = generate_patch(
                                self._test_image, test_point, test_size, padding, to_image=False)

        for test_point in test_points:
            for test_size in test_sizes:
                for padding in failure_paddings:
                    for test_image in self._test_images:
                        with self.assertRaises(ValueError, "invalid padding %s" % padding):
                            patch = generate_patch(
                                self._test_image, test_point, test_size, padding, to_image=True)
                        with self.assertRaises(ValueError, "invalid size %s" % padding):
                            array = generate_patch(
                                self._test_image, test_point, test_size, padding, to_image=False)

    def test_generate_patches(self):
        """画像から指定された条件で、画像から連続してパッチを切り出すメソッドをテストする."""
