from unittest import TestCase
from patch_maker import PatchMaker

class TestPatchMaker(TestCase):
    def test_constructor(self):
        patch_size1 = -1
        patch_size2 = 0
        patch_size3 = 99
        patch_size4 = 100
        PatchMaker(patch_size1)
        PatchMaker(patch_size2)
        PatchMaker(patch_size3)
        PatchMaker(patch_size4)

    def test_patch_size(self):
        self.assertEqual(PatchMaker(32).patch_size, 32)

    def test_get_patches(self):
        self.fail()

    def test_generate_get_next_patches(self):
        self.fail()

    def test_get_n_patches(self):
        self.fail()

    def test_get_patch(self):
        self.fail()

    def test__check(self):
        self.fail()

    def test__get_patch(self):
        self.fail()
