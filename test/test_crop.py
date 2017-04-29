from unittest import TestCase
import numpy as np
from patch_maker import Crop
from patch_maker import Padding


class TestCrop(TestCase):
    def setUp(self):
        array1 = np.arange(20 * 20 * 1).reshape([20, 20]) % 255
        array1 = array1.astype(np.uint8)
        array2 = np.arange(20 * 20 * 3).reshape([20, 20, -1]) % 255
        array2 = array2.astype(np.uint8)
        self._test_array = [array1, array2]

        points = [-10, -1, 0, 1, 20, 21, 30]
        self._points = [(p1, p2) for p1 in points for p2 in points]
        self._failure_points = [(p1, p2) for p1 in [-21, 41] for p2 in points]
        self._failure_points += [(p1, p2) for p1 in points for p2 in [-21, 41]]
        self._failure_arrays = ['', None, 1, 1.0, np.array([])]

    def test_constructor(self):
        paddings = [Padding.MIRROR, Padding.SAME, Padding.VALID]
        arrays = self._test_array
        arrays += [np.array([], dtype=np.uint8).reshape([1, 0])]
        arrays += [np.array([], dtype=np.uint8).reshape([1, 1, 0])]

        for padding in paddings:
            for array in self._test_array:
                try:
                    Crop(padding=padding)
                except Exception as e:
                    self.fail(e)

        for padding in ['', 0, 0.1, None, 'mirror', 'same', 'valid']:
            for array in self._test_array:
                with self.assertRaises(ValueError):
                    Crop(padding=padding)

    def test_holizontal(self):
        for array in self._test_array:
            w = array.shape[1]
            clip_valid = Crop(padding=Padding.VALID)
            clip_mirror = Crop(padding=Padding.MIRROR)
            clip_same = Crop(padding=Padding.SAME)
            for point in self._points:
                # valid
                answer = make_holizontal_answer(valid(point, w), array.shape)
                message = 'padding: %s, point: %s, answer: %s' % (
                    Padding.VALID, str(point), str(answer))
                try:
                    result = clip_valid.holizontal(array, point[0], point[1])
                except Exception as e:
                    self.fail('%s\n%s' % (str(e), message))
                self.assertEqual(result.shape, answer, msg=message)

                # mirror
                answer = make_holizontal_answer(mirror(point, w), array.shape)
                message = 'padding: %s, point: %s, answer: %s' % (
                    Padding.MIRROR, str(point), str(answer))
                try:
                    result = clip_mirror.holizontal(array, point[0], point[1])
                except Exception as e:
                    self.fail('%s\n%s' % (str(e), message))
                self.assertEqual(result.shape, answer, msg=message)

                # same
                answer = make_holizontal_answer(same(point, w), array.shape)
                message = 'padding: %s, point: %s, answer: %s' % (
                    Padding.SAME, str(point), str(answer))
                try:
                    result = clip_same.holizontal(array, point[0], point[1])
                except Exception as e:
                    self.fail('%s\n%s' % (str(e), message))
                self.assertEqual(result.shape, answer, msg=message)

    def test_holizontal_failing(self):
        for array in self._test_array:
            clip_valid = Crop(padding=Padding.VALID)
            clip_mirror = Crop(padding=Padding.MIRROR)
            clip_same = Crop(padding=Padding.SAME)
            for point in self._failure_points:
                with self.assertRaises(ValueError):
                    clip_valid.holizontal(array, point[0], point[1])
                    clip_mirror.holizontal(array, point[0], point[1])
                    clip_same.holizontal(array, point[0], point[1])

        for array in self._failure_arrays:
            clip_valid = Crop(padding=Padding.VALID)
            clip_mirror = Crop(padding=Padding.MIRROR)
            clip_same = Crop(padding=Padding.SAME)
            for point in self._points:
                with self.assertRaises(ValueError):
                    clip_valid.holizontal(array, point[0], point[1])
                    clip_mirror.holizontal(array, point[0], point[1])
                    clip_same.holizontal(array, point[0], point[1])

    def test_vertical(self):
        for array in self._test_array:
            clip_valid = Crop(padding=Padding.VALID)
            clip_mirror = Crop(padding=Padding.MIRROR)
            clip_same = Crop(padding=Padding.SAME)
            h = array.shape[0]
            for point in self._points:
                # VALID
                answer = make_vertical_answer(valid(point, h), array.shape)
                message = 'padding: %s, point: %s, answer: %s' % (
                    Padding.VALID, str(point), str(answer))
                try:
                    result = clip_valid.vertical(array, point[0], point[1])
                except Exception as e:
                    self.fail('%s\n%s' % (str(e), message))
                self.assertEqual(result.shape, answer, msg=message)

                # mirror
                answer = make_vertical_answer(mirror(point, h), array.shape)
                message = 'padding: %s, point: %s, answer: %s' % (
                    Padding.MIRROR, str(point), str(answer))
                try:
                    result = clip_mirror.vertical(array, point[0], point[1])
                except Exception as e:
                    self.fail('%s\n%s' % (str(e), message))
                self.assertEqual(result.shape, answer, msg=message)

                # same
                answer = make_vertical_answer(same(point, h), array.shape)
                message = 'padding: %s, point: %s, answer: %s' % (
                    Padding.SAME, str(point), str(answer))
                try:
                    result = clip_same.vertical(array, point[0], point[1])
                except Exception as e:
                    self.fail('%s\n%s' % (str(e), message))
                self.assertEqual(result.shape, answer, msg=message)

    def test_vertical_failing(self):
        for array in self._test_array:
            clip_valid = Crop(padding=Padding.VALID)
            clip_mirror = Crop(padding=Padding.MIRROR)
            clip_same = Crop(padding=Padding.SAME)
            for point in self._failure_points:
                with self.assertRaises(ValueError):
                    clip_valid.vertical(array, point[0], point[1])
                    clip_mirror.vertical(array, point[0], point[1])
                    clip_same.vertical(array, point[0], point[1])

        for array in self._failure_arrays:
            clip_valid = Crop(padding=Padding.VALID)
            clip_mirror = Crop(padding=Padding.MIRROR)
            clip_same = Crop(padding=Padding.SAME)
            for point in self._points:
                with self.assertRaises(ValueError):
                    clip_valid.vertical(array, point[0], point[1])
                    clip_mirror.vertical(array, point[0], point[1])
                    clip_same.vertical(array, point[0], point[1])

    def test_center(self):
        points_list = [(x, y) for x in self._points for y in self._points]

        for array in self._test_array:
            clip_valid = Crop(padding=Padding.VALID)
            clip_mirror = Crop(padding=Padding.MIRROR)
            clip_same = Crop(padding=Padding.SAME)
            h, w = array.shape[:2]
            for ps in points_list:
                # VALID
                box = (ps[0][0], ps[1][0], ps[0][1], ps[1][1])
                x1, x2 = valid(ps[0], w)
                y1, y2 = valid(ps[1], h)
                answer = make_center_answer((x1, y1, x2, y2), array.shape)
                message = 'padding: %s, box: %s, answer: %s' % (
                    Padding.VALID, str(box), str(answer))
                try:
                    result = clip_valid.center(array, box)
                except Exception as e:
                    self.fail('%s\n%s' % (str(e), message))
                self.assertEqual(result.shape, answer, msg=message)

                # mirror
                x1, x2 = mirror(ps[0], w)
                y1, y2 = mirror(ps[1], h)
                answer = make_center_answer((x1, y1, x2, y2), array.shape)
                message = 'padding: %s, box: %s, answer: %s' % (
                    Padding.MIRROR, str(box), str(answer))
                try:
                    result = clip_mirror.center(array, box)
                except Exception as e:
                    self.fail('%s\n%s' % (str(e), message))
                self.assertEqual(result.shape, answer, msg=message)

                # same
                x1, x2 = same(ps[0], w)
                y1, y2 = same(ps[1], h)
                answer = make_center_answer((x1, y1, x2, y2), array.shape)
                message = 'padding: %s, box: %s, answer: %s' % (
                    Padding.SAME, str(box), str(answer))
                try:
                    result = clip_same.center(array, box)
                except Exception as e:
                    self.fail('%s\n%s' % (str(e), message))
                self.assertEqual(result.shape, answer, msg=message)

    def test_center_failing(self):
        failure_boxes = [(x[0], y[0], x[1], y[1])
                         for x in self._failure_points for y in self._failure_points]
        failure_boxes += [(p[0], p[1], fp[0], fp[1])
                          for p in self._points for fp in self._failure_points]
        failure_boxes += [(fp[0], fp[1], p[0], p[1])
                          for p in self._points for fp in self._failure_points]

        for array in self._test_array:
            clip_valid = Crop(padding=Padding.VALID)
            clip_mirror = Crop(padding=Padding.MIRROR)
            clip_same = Crop(padding=Padding.SAME)
            for box in failure_boxes:
                with self.assertRaises(ValueError):
                    clip_valid.center(array, box)
                    clip_mirror.center(array, box)
                    clip_same.center(array, box)


def valid(point, limit=None):
    p1, p2 = point
    p1, p2 = max(min(p1, limit), 0), max(min(p2, limit), 0)
    return (p1, p2)


def mirror(point, limit=None):
    return point


def same(point, limit=None):
    return point


def make_holizontal_answer(point, shape):
    p1, p2 = point
    height, width = shape[:2]
    return (height, max(0, p2 - p1)) + shape[2:]


def make_vertical_answer(point, shape):
    p1, p2 = point
    height, width = shape[:2]
    return (max(0, p2 - p1), width) + shape[2:]


def make_center_answer(box, shape):
    x1, y1, x2, y2 = box
    height, width = shape[:2]
    return (max(0, y2 - y1), max(0, x2 - x1)) + shape[2:]
