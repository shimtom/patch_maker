from unittest import TestCase
import patch_maker as pm
import numpy as np


class TestClip(TestCase):
    def setUp(self):
        array1 = np.arange(20 * 20 * 1).reshape([20, 20]) % 255
        array1 = array1.astype(np.uint8)
        array2 = np.arange(20 * 20 * 3).reshape([20, 20, -1]) % 255
        array2 = array2.astype(np.uint8)
        self._test_array = [array1, array2]

    def test_constructor(self):
        paddings = [pm.MIRROR, pm.SAME, pm.VALID]
        arrays = self._test_array
        arrays += [np.array([], dtype=np.uint8).reshape([1, 0])]
        arrays += [np.array([], dtype=np.uint8).reshape([1, 1, 0])]

        for padding in paddings:
            for array in self._test_array:
                try:
                    pm.Clip(padding=padding)
                except ValueError as e:
                    self.self.fail(e)

        for padding in ['', 0, 0.1, None, 'mirror', 'same', 'valid']:
            for array in self._test_array:
                with self.assertRaises(ValueError):
                    pm.Clip(padding=padding)

    def test_holizontal(self):
        points = [(-10, -10), (-10, -1), (-10, 0), (-10, 1),
                  (-10, 20), (-10, 21), (-10, 30)]
        points += [(-1, -10), (-1, -1), (-1, 0), (-1, 1),
                   (-1, 20), (-1, 21), (-1, 30)]
        points += [(0, -10), (0, -1), (0, 0), (0, 1),
                   (0, 20), (0, 21), (0, 30)]
        points += [(20, -10), (20, -1), (20, 0), (20, 1),
                   (20, 20), (20, 21), (20, 30)]
        points += [(30, -10), (30, -1), (30, 0), (30, 1),
                   (30, 20), (30, 21), (30, 30)]

        for array in self._test_array:
            clip_valid = pm.Clip(padding=pm.VALID)
            clip_mirror = pm.Clip(padding=pm.MIRROR)
            clip_same = pm.Clip(padding=pm.SAME)
            for point in points:
                h, w = array.shape[:2]
                # VALID
                answer = [array.shape[0], max(
                    0, max(min(point[1], w), 0) - max(min(point[0], w), 0))]
                answer += array.shape[2:]
                answer = tuple(answer)
                result = clip_valid.holizontal(array, point[0], point[1])
                message = 'padding: %s, point: %s, answer: %s, shape: %s' % (
                    pm.VALID, str(point), str(answer), str(result.shape))
                self.assertEqual(result.shape, answer, msg=message)
                # mirror
                answer = [array.shape[0], max(0, point[1] - point[0])]
                answer += array.shape[2:]
                answer = tuple(answer)
                result = clip_mirror.holizontal(array, point[0], point[1])
                message = 'padding: %s, point: %s, answer: %s, shape: %s' % (
                    pm.MIRROR, str(point), str(answer), str(result.shape))
                self.assertEqual(result.shape, answer, msg=message)
                # same
                answer = [array.shape[0], max(0, point[1] - point[0])]
                answer += array.shape[2:]
                answer = tuple(answer)
                result = clip_same.holizontal(array, point[0], point[1])
                message = 'padding: %s, point: %s, answer: %s, shape: %s' % (
                    pm.SAME, str(point), str(answer), str(result.shape))
                self.assertEqual(result.shape, answer, msg=message)

        failure_points = [(-11, -10), (-11, -1), (-11, 0),
                          (-11, 1), (-11, 20), (-11, 21), (-11, 30)]
        failure_points += [(31, -10), (31, -1), (31, 0),
                           (31, 1), (31, 20), (31, 21), (31, 30)]
        for array in self._test_array:
            clip_valid = pm.Clip(padding=pm.VALID)
            clip_mirror = pm.Clip(padding=pm.MIRROR)
            clip_same = pm.Clip(padding=pm.SAME)
            for point in failure_points:
                with self.assertRaises(ValueError):
                    clip_valid.holizontal(array, point[0], point[1])
                    clip_mirror.holizontal(array, point[0], point[1])
                    clip_same.holizontal(array, point[0], point[1])

        failure_array = ['', None, 1, 1.0, np.array([])]
        for array in failure_array:
            clip_valid = pm.Clip(padding=pm.VALID)
            clip_mirror = pm.Clip(padding=pm.MIRROR)
            clip_same = pm.Clip(padding=pm.SAME)
            for point in points:
                with self.assertRaises(ValueError):
                    clip_valid.holizontal(array, point[0], point[1])
                    clip_mirror.holizontal(array, point[0], point[1])
                    clip_same.holizontal(array, point[0], point[1])

    def test_vertical(self):
        points = [(-10, -10), (-10, -1), (-10, 0), (-10, 1),
                  (-10, 20), (-10, 21), (-10, 30)]
        points += [(-1, -10), (-1, -1), (-1, 0), (-1, 1),
                   (-1, 20), (-1, 21), (-1, 30)]
        points += [(0, -10), (0, -1), (0, 0), (0, 1),
                   (0, 20), (0, 21), (0, 30)]
        points += [(20, -10), (20, -1), (20, 0), (20, 1),
                   (20, 20), (20, 21), (20, 30)]
        points += [(30, -10), (30, -1), (30, 0), (30, 1),
                   (30, 20), (30, 21), (30, 30)]

        for array in self._test_array:
            clip_valid = pm.Clip(padding=pm.VALID)
            clip_mirror = pm.Clip(padding=pm.MIRROR)
            clip_same = pm.Clip(padding=pm.SAME)
            for point in points:
                h, w = array.shape[:2]
                # VALID
                answer = [max(0, max(min(point[1], h), 0) -
                              max(min(point[0], h), 0)), array.shape[1]]
                answer += array.shape[2:]
                answer = tuple(answer)
                result = clip_valid.vertical(array, point[0], point[1])
                message = 'padding: %s, point: %s, answer: %s, shape: %s' % (
                    pm.VALID, str(point), str(answer), str(result.shape))
                self.assertEqual(result.shape, answer, msg=message)
                # mirror
                answer = [max(0, point[1] - point[0]), array.shape[1]]
                answer += array.shape[2:]
                answer = tuple(answer)
                result = clip_mirror.vertical(array, point[0], point[1])
                message = 'padding: %s, point: %s, answer: %s, shape: %s' % (
                    pm.MIRROR, str(point), str(answer), str(result.shape))
                self.assertEqual(result.shape, answer, msg=message)
                # same
                answer = [max(0, point[1] - point[0]), array.shape[1]]
                answer += array.shape[2:]
                answer = tuple(answer)
                result = clip_same.vertical(array, point[0], point[1])
                message = 'padding: %s, point: %s, answer: %s, shape: %s' % (
                    pm.SAME, str(point), str(answer), str(result.shape))
                self.assertEqual(result.shape, answer, msg=message)

        failure_points = [(-11, -10), (-11, -1), (-11, 0),
                          (-11, 1), (-11, 20), (-11, 21), (-11, 30)]
        failure_points += [(31, -10), (31, -1), (31, 0),
                           (31, 1), (31, 20), (31, 21), (31, 30)]

        for array in self._test_array:
            clip_valid = pm.Clip(padding=pm.VALID)
            clip_mirror = pm.Clip(padding=pm.MIRROR)
            clip_same = pm.Clip(padding=pm.SAME)
            for point in failure_points:
                with self.assertRaises(ValueError):
                    clip_valid.vertical(array, point[0], point[1])
                    clip_mirror.vertical(array, point[0], point[1])
                    clip_same.vertical(array, point[0], point[1])

        failure_arrays = ['', None, 1, 1.0, np.array([])]
        for array in failure_arrays:
            clip_valid = pm.Clip(padding=pm.VALID)
            clip_mirror = pm.Clip(padding=pm.MIRROR)
            clip_same = pm.Clip(padding=pm.SAME)
            for point in points:
                with self.assertRaises(ValueError):
                    clip_valid.vertical(array, point[0], point[1])
                    clip_mirror.vertical(array, point[0], point[1])
                    clip_same.vertical(array, point[0], point[1])

    def test_center(self):
        def _make_valid_answer(box, shape):
            height, width = shape[:2]
            x1, y1, x2, y2 = box
            yshape = max(0, max(min(y2, height), 0) - max(min(y1, height), 0))
            xshape = max(0, max(min(x2, width), 0) - max(min(x1, width), 0))
            answer = (yshape, xshape) + shape[2:]
            return answer

        def _make_mirror_answer(box, shape):
            x1, y1, x2, y2 = box
            yshape, xshape = max(0, y2 - y1), max(0, x2 - x1)
            answer = (yshape, xshape) + shape[2:]
            return answer

        def _make_same_answer(box, shape):
            x1, y1, x2, y2 = box
            yshape, xshape = max(0, y2 - y1), max(0, x2 - x1)
            answer = (yshape, xshape) + shape[2:]
            return answer
        points = [(-10, -10), (-10, -1), (-10, 0), (-10, 1),
                  (-10, 20), (-10, 21), (-10, 30)]
        points += [(-1, -10), (-1, -1), (-1, 0), (-1, 1),
                   (-1, 20), (-1, 21), (-1, 30)]
        points += [(0, -10), (0, -1), (0, 0), (0, 1),
                   (0, 20), (0, 21), (0, 30)]
        points += [(20, -10), (20, -1), (20, 0), (20, 1),
                   (20, 20), (20, 21), (20, 30)]
        points += [(30, -10), (30, -1), (30, 0), (30, 1),
                   (30, 20), (30, 21), (30, 30)]
        boxes = [(x[0], y[0], x[1], y[1]) for x in points for y in points]

        for array in self._test_array:
            clip_valid = pm.Clip(padding=pm.VALID)
            clip_mirror = pm.Clip(padding=pm.MIRROR)
            clip_same = pm.Clip(padding=pm.SAME)
            for box in boxes:
                h, w = array.shape[:2]
                # VALID
                answer = _make_valid_answer(box, array.shape)
                result = clip_valid.center(array, box)
                message = 'padding: %s, box: %s, answer: %s, shape: %s' % (
                    pm.VALID, str(box), str(answer), str(result.shape))
                self.assertEqual(result.shape, answer, msg=message)
                # mirror
                answer = _make_mirror_answer(box, array.shape)
                result = clip_mirror.center(array, box)
                message = 'padding: %s, box: %s, answer: %s, shape: %s' % (
                    pm.MIRROR, str(box), str(answer), str(result.shape))
                self.assertEqual(result.shape, answer, msg=message)
                # same
                answer = _make_mirror_answer(box, array.shape)
                result = clip_same.center(array, box)
                message = 'padding: %s, box: %s, answer: %s, shape: %s' % (
                    pm.MIRROR, str(box), str(answer), str(result.shape))
                self.assertEqual(result.shape, answer, msg=message)

        failure_points = [(-11, -10), (-11, -1), (-11, 0),
                          (-11, 1), (-11, 20), (-11, 21), (-11, 30)]
        failure_points += [(31, -10), (31, -1), (31, 0),
                           (31, 1), (31, 20), (31, 21), (31, 30)]
        failure_boxes = [(x[0], y[0], x[1], y[1])
                         for x in failure_points for y in failure_points]
        failure_boxes += [(p[0], p[1], fp[0], fp[1])
                          for p in points for fp in failure_points]
        failure_boxes += [(fp[0], fp[1], p[0], p[1])
                          for p in points for fp in failure_points]
        for array in self._test_array:
            clip_valid = pm.Clip(padding=pm.VALID)
            clip_mirror = pm.Clip(padding=pm.MIRROR)
            clip_same = pm.Clip(padding=pm.SAME)
            for box in failure_boxes:
                with self.assertRaises(ValueError):
                    clip_valid.center(array, box)
                    clip_mirror.center(array, box)
                    clip_same.center(array, box)
