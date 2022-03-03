import cv2 as cv
import numpy as np

cv2_luts = [lut for lut in dir(cv) if lut.startswith("COLORMAP_")]
print(f"opencv lut colormap number: {len(cv2_luts)}")
print(f"opencv luts colormap: {cv2_luts}")

print("cv.COLORMAP_COOL", type("cv.COLORMAP_COOL", ))
print(eval("cv.COLORMAP_COOL"), type(eval("cv.COLORMAP_COOL")))

cv.samples.addSamplesDataSearchPath('../mydata')
file = cv.samples.findFile('messi5.jpg')
if not file:
    raise FileNotFoundError('file not found')

image = cv.imread(file, cv.IMREAD_COLOR)
all_lut_imgs = [(lut, cv.applyColorMap(image, eval("cv." + lut))) for lut in cv2_luts]

add_text_imgs = [cv.putText(lut_img[1], lut_img[0], (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) for lut_img
                 in all_lut_imgs]

col1 = np.vstack(tuple(add_text_imgs[0:11]))
col2 = np.vstack(tuple(add_text_imgs[11:22]))
result = np.hstack((col1, col2))
# cv.imwrite("lut_result.jpg", result)
cv.imshow("result", result)
cv.waitKey()
