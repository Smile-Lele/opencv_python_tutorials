from unittest import TestCase

from mypackage.imUtils.icv import *


class Test(TestCase):
    def setUp(self):
        self.mshape = (9, 16)
        self.dsize = (1080, 1920)
        self.ones = np.ones(self.mshape, np.float32)
        self.mat = np.arange(1, 145).reshape(self.mshape).astype(np.float32)
        self.gray = cv.resize(self.mat, (1920, 1080), interpolation=cv.INTER_AREA)
        self.color = cv.cvtColor(self.gray, cv.COLOR_GRAY2BGR)

    def tearDown(self):
        print(f'well done!')

    def test_is_color(self):
        self.assertFalse(isColor(self.gray))
        self.assertTrue(isColor(self.color))

    def test_img2mat(self):
        realValue = img2Mat(self.color, self.mshape)
        expectedValue = self.mat
        # self.assertTrue((realValue == expectedValue).all())
        np.testing.assert_allclose(realValue, expectedValue)

    def test_mat2grid_img(self):
        realValue = mat2GridImg(self.mat, self.dsize)
        expectedValue = self.gray
        np.testing.assert_allclose(realValue, expectedValue)

    def test_cvt_bgr2gray(self):
        realValue = cvtBGR2Gray(self.color)
        expectedValue = self.gray
        np.testing.assert_allclose(realValue, expectedValue)

    def test_cvt_gray2bgr(self):
        realValue = self.color
        expectedValue = cvtGray2BGR(self.gray)
        np.testing.assert_allclose(realValue, expectedValue)

    def test_mean_filter_on_gray(self):
        realValue = meanFilterOnGray([self.gray, self.gray, self.gray])
        expectedValue = self.gray
        np.testing.assert_allclose(realValue, expectedValue)

    def test_mat2mask(self):
        realValue = mat2Mask(self.ones)
        expectedValue = self.ones * 255
        np.testing.assert_allclose(realValue, expectedValue)

    def test_scale_abs_ex(self):
        realValue = scaleAbs_ex(self.ones, 255)
        expectedValue = self.ones * 255
        np.testing.assert_allclose(realValue, expectedValue)

    def test_bitwise_mask(self):
        realValue = bitwise_mask(self.ones, mat2Mask(self.ones))
        expectedValue = self.ones
        np.testing.assert_allclose(realValue, expectedValue)
