# coding: utf-8

import numpy as np

canvas = np.zeros((1080, 1920), dtype=np.uint8)
msg = 'src:{} = dup:{} is {}'

# assignment
print('\n', 'assignment')
img = canvas
print(msg.format(id(canvas), id(img), id(img) == id(canvas)))
img.shape = 2160, 960
print(msg.format(canvas.shape , img.shape, canvas.shape == img.shape))
canvas.shape = 1080, 1920

# view - shadow copy
print('\n', 'view')
view_ = canvas.view()
print(msg.format(id(canvas), id(view_),id(view_) == id(canvas)))

view_.shape = 2160, 960
print(msg.format(canvas.shape , view_.shape, canvas.shape == view_.shape))
canvas.shape = 1080, 1920

view_[0][0] = 123
print(msg.format(view_[0][0], canvas[0][0], view_[0][0] == canvas[0][0]))
view_[0][0] = 0


# index - shadow copy
print('\n', 'index')
roi = canvas[0:200, 0:300]
print(msg.format(id(canvas), id(roi), id(roi) == id(canvas)))

roi[0][0] = 123
print(msg.format(roi[0][0], canvas[0][0], roi[0][0] == canvas[0][0]))
roi[0][0] = 0

# copy - deep copy
print('\n', 'copy')
copy_ = canvas.copy()
print(msg.format(id(canvas), id(copy_), id(copy_) == id(canvas)))
copy_.shape = 2160, 960
print(msg.format(canvas.shape , copy_.shape, canvas.shape == copy_.shape))

copy_[0][0] = 123
print(msg.format(copy_[0][0], canvas[0][0], copy_[0][0] == canvas[0][0]))
copy_[0][0] = 0
