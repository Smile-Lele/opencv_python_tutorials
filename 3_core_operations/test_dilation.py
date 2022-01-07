import cv2 as cv


img = cv.imread('copy_15-15-10.png', cv.IMREAD_GRAYSCALE)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
print(kernel)
img = cv.morphologyEx(img, cv.MORPH_DILATE, kernel, iterations=1)
cv.imwrite('img1.png', img)
img = cv.morphologyEx(img, cv.MORPH_DILATE, kernel, iterations=1)
cv.imwrite('img2.png', img)

"""
Summary:
1. In term of non-curve structure elements,
once dilation only effects one pixel around it.
2. Strongly recommend using odd kernel
"""