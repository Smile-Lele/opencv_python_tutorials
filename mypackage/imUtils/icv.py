import itertools
import math
from concurrent import futures
from functools import partial

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from mypackage.strUtils import str_utils

"""
Basic Image Utilities
"""


def isColor(img):
    return img.ndim == 3 and img.shape[-1] == 3


def imread_ex(filename, flags):
    return cv.imdecode(np.fromfile(filename, dtype=np.uint8), flags)


def imwrite_ex(file, img):
    ext = str_utils.split_dir(file)[-1]
    cv.imencode(ext, img)[1].tofile(file)


def img2Mat(img, mshape):
    """
    The function will convert image to matrix, each region of which is
    from the average of grayscale in the whole region.
    :param img: image should be a single channel matrix
    :param mshape: it is a tuple (row, col)
    :return: matrix(row, col)
    """

    cvtBGR2Gray(img)

    mrows, mcols = mshape
    ceils = [np.array_split(row, mcols, axis=1) for row in np.array_split(img, mrows, axis=0)]
    means = [np.mean(ceil) for row_img in ceils for ceil in row_img]
    mat = np.array(means).reshape(mshape)

    # print(f'src:({img.shape},{img.dtype}) -> mat:({mat.shape},{mat.dtype})')
    return mat


def mat2Mask(mat):
    if isColor(mat):
        mat = cvtBGR2Gray(mat)
    mask = np.ones_like(mat, np.float32) * 255
    min_ = mat.min() + 1 if mat.min() == 0 else mat.min()
    mask /= (mat / min_)
    return mask


def bitwise_mask(src, mask):
    """
    The method is to adjust the grayscale of each pixels in source image.
    :param mask:
    :param src:
    :return:
    """
    assert src.size == mask.size, f'{src.size=} != {mask.size=}'
    return cv.convertScaleAbs(src * (mask / 255))


def mat2GridImg(src: (np.uint8, np.float32), dsize):
    return cv.resize(src, tuple(reversed(dsize)), interpolation=cv.INTER_AREA)


def resize_ex(src: (np.uint8, np.float32), dsize):
    """
    This is a extension of resize
    :param src:
    :param dsize:
    :return:
    """
    # print(f'resize: \nsrc{src.shape} -> dst:{dsize}')
    inter_type = [cv.INTER_CUBIC, cv.INTER_AREA][src.size > dsize[0] * dsize[1]]
    dst = cv.resize(src, tuple(reversed(dsize)), interpolation=inter_type)
    return dst


def scaleAbs_ex(src, maxVal):
    return cv.convertScaleAbs(src, alpha=maxVal / src.max())


def remap_ex(img, mapx, mapy):
    row, col = img.shape[:2]
    mapx = mapx.reshape(row, col).astype(cv.CV_32FC1)
    mapy = mapy.reshape(row, col).astype(cv.CV_32FC1)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    return dst


def cvtBGR2Gray(img):
    assert img is not None, 'img is None'
    if isColor(img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def cvtGray2BGR(img):
    assert img is not None, 'img is None'
    if not isColor(img):
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    return img


def otsuThreshold(img, min_thre=0, max_thre=255, offset=0, inv=False, visibility=False):
    img = cv.GaussianBlur(img, (3, 3), 0)

    min_thre += offset
    max_thre -= offset
    img[img < min_thre] = min_thre
    img[img > max_thre] = max_thre

    thre, thre_img = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # print(f'otsu thre:{thre}')

    if inv:
        thre_img = cv.bitwise_not(thre_img)

    if visibility:
        imshow_ex(thre_img)
    return thre, thre_img


def concatFilter(imgs):
    sum_ = 0
    for img in imgs:
        sum_ += img
    return cv.convertScaleAbs(sum_ / len(imgs))


def imgs2Mats_helper(img, mshape, index):
    return [index, img2Mat(img, mshape)]


def imgs2Mats(imgs, mshape):
    to_do_list = list()
    with futures.ThreadPoolExecutor(max_workers=len(imgs)) as executor:
        for idx, img in enumerate(imgs):
            future = executor.submit(imgs2Mats_helper, img, mshape, idx)
            to_do_list.append(future)
        done_iter = futures.as_completed(to_do_list)

        res = [future.result() for future in done_iter]
        sort_res = sorted(res, key=lambda r: r[0])
        mats = [res[1] for res in sort_res]

    return mats


def maxHistGray(img):
    hist = cv.calcHist([img], [0], mask=None, histSize=[256], ranges=[0, 256])
    hist_list = hist.squeeze().tolist()
    return hist_list.index(max(hist_list))


def imEvaluate(mat):
    range_ = np.ptp(mat)
    mean, stddev = cv.meanStdDev(mat)
    print(f'evaluate: μ:{mean[0, 0]:.2f} | σ:{stddev[0, 0]:.2f} | range:{range_}')


def imClarity(img):
    beta = 0.5
    clarity_lap = cv.Laplacian(img, cv.CV_64F).var()
    clarity_sobel = cv.Sobel(img, cv.CV_64F, 1, 1).var()
    return beta * clarity_lap + (1 - beta) * clarity_sobel


def linearInterWithIndex(dsize, r_idx, c_idx, vals):
    canvas = np.zeros(dsize, np.float32)
    for r in range(len(r_idx)):
        head_step = 0
        for c in range(len(c_idx) - 1):
            canvas[r_idx[r], c_idx[c]:c_idx[c + 1] + 1], step = np.linspace(vals[r, c], vals[r, c + 1],
                                                                            num=1 + c_idx[c + 1] - c_idx[c],
                                                                            retstep=True)
            if c == 0:
                head_step = step

            if c == len(c_idx) - 2:
                tail_step = step
                canvas[r_idx[r], :c_idx[0]] = np.array(
                    list(reversed([vals[r, 0] - (n + 1) * head_step for n in range(c_idx[0])])))
                canvas[r_idx[r], c_idx[c + 1] + 1:] = np.array(
                    [vals[r, c + 1] + (n + 1) * tail_step for n in range(dsize[1] - c_idx[c + 1] - 1)])

    for c in range(dsize[1]):
        head_step = 0
        for r in range(len(r_idx) - 1):
            canvas[r_idx[r]:r_idx[r + 1] + 1, c], step = np.linspace(canvas[r_idx[r], c], canvas[r_idx[r + 1], c],
                                                                     num=1 + r_idx[r + 1] - r_idx[r],
                                                                     retstep=True)
            if r == 0:
                head_step = step
            if r == len(r_idx) - 2:
                tail_step = step
                canvas[:r_idx[0], c] = np.array(
                    list(reversed([canvas[r_idx[0], c] - (n + 1) * head_step for n in range(r_idx[0])])))
                canvas[r_idx[r + 1] + 1:, c] = np.array(
                    [canvas[r_idx[r + 1], c] + (n + 1) * tail_step for n in range(dsize[0] - r_idx[r + 1] - 1)])
    return canvas


"""
Image Visualization
"""


def implot_ex(imdict):
    """
    This is an adaptive drawing module
    :param imdict:
    :return:
    """
    if not isinstance(imdict, dict):
        raise TypeError('param must be type dict()')
    assert imdict, 'img dict is empty'

    # adaptively adjust the row and col of canvas
    len_ = len(imdict)
    row = (len_ - 1) // (2 + (len_ - 1) // 4) + 1
    col = math.ceil(len_ / row)

    for i, (title, data) in enumerate(imdict.items(), 1):
        plt.subplot(row, col, i)

        if isColor(data):
            data = cv.cvtColor(data, cv.COLOR_BGR2RGB)

        cmap = 'jet' if 'jet' in title else 'gray'
        plt.imshow(data, cmap)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
    plt.show()


def imshow_ex(img, win_size=(1920 * 4 / 5, 1080 * 4 / 5)):
    """
    The method is to make image to fit window size in the win while displaying
    :param img:
    :param win_size:
    :return:
    """
    row, col = img.shape[:2]
    win_c, win_r = win_size
    row_ratio = win_r / row
    col_ratio = win_c / col
    scale_ratio = row_ratio if row_ratio <= col_ratio else col_ratio
    viz_img = img
    if scale_ratio != 1:
        INTERPOLATION = cv.INTER_AREA if scale_ratio < 1 else cv.INTER_CUBIC
        img = cv.resize(img, None, fx=scale_ratio, fy=scale_ratio, interpolation=INTERPOLATION)
        viz_img = img
    cv.imshow('viz_img', viz_img)
    cv.waitKey()
    cv.destroyWindow('viz_img')


"""
Image Filter Algorithm:
"""


def fitLineAngle(pnts):
    line = cv.fitLine(pnts, cv.DIST_L2, 0, 0.01, 0.01)
    k = line[0] / line[1]
    sin_k = - k / np.sqrt(1 + k * k)
    angle = np.rad2deg(np.arcsin(sin_k))
    # angle = np.rad2deg(np.arctan(k))
    return angle


def fitLineAngle_ex(pnts):
    pnts_pairs = [pairs for pairs in itertools.combinations(pnts, 2)]
    pnts_pairs = np.asarray(pnts_pairs)
    vector = pnts_pairs[:, 0] - pnts_pairs[:, 1]
    mags, angles = cv.cartToPolar(vector[:, 0], vector[:, 1], angleInDegrees=True)
    angle = np.mean(angles) - 270  # rotate image coordinates to world coordinates
    return angle


def fit2dMatrixAngleHelper(pnts, func):
    row, col = pnts.shape[:2]
    angle_col = np.mean([func(pnts[:, c]) for c in range(col)])
    angle_row = np.mean([func(pnts[r, :]) for r in range(row)]) + 90
    angle = (angle_col + angle_row) / 2
    return angle


def fit2dMatrixAngle(pnts, algorithmType='polar'):
    return fit2dMatrixAngleHelper(pnts, fitLineAngle_ex) if algorithmType == 'polar' \
        else fit2dMatrixAngleHelper(pnts, fitLineAngle)


def convertPloyMatrix(x, degrees):
    X = np.asarray([np.power(x, c) for c in range(degrees, -1, -1)]).T
    return X


def polyfit(pnts, degrees=1):
    # X(n, m).T * K(n, 1) = Y(m, 1)
    m = len(pnts)
    n = degrees + 1
    assert m > n, f'{m=} should be greater than {n=}'

    # prepare dataset
    pnts = np.asarray(pnts).reshape(-1, 2)
    x = pnts[:, 0]
    y = pnts[:, 1]
    X = convertPloyMatrix(x, degrees)
    Y = np.asarray(y).reshape(-1, 1)

    # Least Square
    # K = np.linalg.inv(X.T @ X) @ X.T @ Y
    _, K = cv.solve(X, Y, flags=cv.DECOMP_QR)
    return K


def polyfunc(X, K):
    return X @ K


def solveHelper(coeffs, func) -> list:
    return func(coeffs)[1].squeeze().tolist()


def solve1d_ex(coeffs: list, y):
    assert len(coeffs) != 0, f'Error {len(coeffs)=}'
    coeffs = np.float32(coeffs)
    coeffs[-1] -= y
    return solveHelper(coeffs, cv.solveCubic) if len(coeffs) == 4 else solveHelper(coeffs[::-1], cv.solvePoly)


def filterRoots(roots: list, lo, hi):
    assert len(roots) != 0, f'Error: {len(roots)=}'
    if isinstance(roots[0], list) and len(roots[0]) == 2:
        return [r[0] for r in roots if abs(r[1]) < 1e-8 and lo < r[0] < hi]
    return list(filter(lambda r: lo < r < hi, roots))


def denoise(image):
    image = cv.fastNlMeansDenoising(image, 5, 7, 21)
    image = cv.edgePreservingFilter(image, sigma_s=25, sigma_r=0.3, flags=cv.RECURS_FILTER)
    return image


def remove_anomaly_gaussian(imgs):
    """
    This function is to remove less anomaly images
    :param imgs:
    :return: get images in larger possibility distribution
    """
    im_gray_mean = [np.mean(img) for img in imgs]
    min_, max_ = np.argmin(im_gray_mean), np.argmax(im_gray_mean)
    imgs.pop(min_)
    imgs.pop(max_)

    im_gray_mean = [np.mean(img) for img in imgs]
    im_gray_mean = np.asarray(im_gray_mean)
    mean, std = im_gray_mean.mean(), im_gray_mean.std()
    cond = np.abs(im_gray_mean - mean) < std
    valid_index = np.where(cond)
    # print(valid_index[0].tolist())
    valid_imgs = [imgs[i] for i in valid_index[0].tolist()]
    print(f'total: {len(imgs)}, valid: {len(valid_imgs)}')

    return valid_imgs


def remove_anomaly_kmeans(imgs):
    data = [np.mean(img) for img in imgs]

    data = np.asarray(data, dtype=np.float32).reshape(-1, 1)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-3)
    flags = cv.KMEANS_RANDOM_CENTERS

    # 应用K均值
    compactness, labels, centers = cv.kmeans(data, 5, None, criteria, 100, flags)
    # print(centers)

    class_index = np.argsort(centers, axis=0)
    med_class_index = class_index[1:4].ravel()

    valid_region = np.zeros_like(labels, bool)
    for i in med_class_index:
        valid_region |= (labels == i)

    valid_index = np.where(valid_region)
    valid_imgs = [imgs[i] for i in valid_index[0].tolist()]
    print(f'total: {len(imgs)}, valid: {len(valid_imgs)}')
    return valid_imgs


def remove_anomaly_ransac(imgs, sigma):
    data = [np.mean(img) for img in imgs]

    d_min, d_max = np.min(data), np.max(data)
    pivot = np.linspace(d_min, d_max + 1, num=np.int0(np.ptp(data)))
    max_pnts_num = -1
    retpivot = 0
    for p in pivot:
        region_pnts_num = len(data) - (np.sum(data > p + sigma) + np.sum(data < p - sigma))
        if region_pnts_num >= max_pnts_num:
            max_pnts_num = region_pnts_num
            retpivot = p
        if max_pnts_num == -1:
            print(p)

    min_index = np.argwhere(np.asarray(data) >= retpivot - sigma).ravel()
    max_index = np.argwhere(np.asarray(data) <= retpivot + sigma).ravel()
    union_index = set(min_index) & set(max_index)

    valid_imgs = [imgs[i] for i in union_index]
    print(f'total: {len(imgs)}, valid: {len(valid_imgs)}')

    return valid_imgs


"""
Image Template
"""


def create_canvas(dsize):
    """
    The function creates canvas with dsize, grayscale per pixel is 0
    :param dsize: img(row, col)
    :return: empty or black image
    """
    canvas = np.zeros(dsize, np.uint8)
    info = 'canvas: \nshape:{} \ndtype:{}\n'
    info = info.format(canvas.shape, canvas.dtype)
    print(info)
    return canvas


def create_whiteboard(dsize):
    """
    The function creates white board with dsize, grayscale per pixel is 255
    :param dsize: image(row, col)
    :return: white image
    """
    canvas = np.zeros(dsize, np.uint8)
    canvas.fill(255)
    info = 'whiteboard: \nshape:{} \ndtype:{}\n'
    info = info.format(canvas.shape, canvas.dtype)
    print(info)
    return canvas


def create_chessboard(mshape, pixels):
    """
    The function is to create a chessboard.
    :param mshape: (row, col)
    :param pixels: represents the pixel per grid
    :return: chessboard image
    """
    rows, cols = mshape
    height = (2 * rows - 1) * pixels
    width = (2 * cols - 1) * pixels
    img = np.zeros((width, height), np.uint8)
    img.fill(255)
    for r in range(rows):
        for c in range(cols):
            cv.rectangle(img, (2 * r * pixels, 2 * c * pixels),
                         ((2 * r + 1) * pixels, (2 * c + 1) * pixels), 0, -1)

            cv.rectangle(img, ((2 * r + 1) * pixels, (2 * c + 1) * pixels),
                         ((2 * r + 2) * pixels, (2 * c + 2) * pixels), 0, -1)
    info = 'chessboard: \nmshape:{} \npixels:{}\n'
    info = info.format(mshape, pixels)
    print(info)
    return img


def draw_mesh(image, mshape):
    """
    This is function to draw mesh (row, col) on the image
    Note: copy is shadow copy in order to block
    :param image: any
    :param mshape: it is a tuple(row, col)
    :return: meshed image
    """
    img = image.copy()
    rows, cols = mshape
    im_r, im_c = img.shape[:2]

    r_grid = im_r // rows
    c_grid = im_c // cols

    draw_line = partial(cv.line, img, color=0, thickness=3, lineType=cv.LINE_8)

    [draw_line((c * c_grid, 0), (c * c_grid, im_r)) for c in range(1, cols)]
    [draw_line((0, r * r_grid), (im_c, r * r_grid)) for r in range(1, rows)]

    info = 'mesh: \nimshape:{} \nmshape:{}\n'
    info = info.format(image.shape, mshape)
    print(info)
    return img


def draw_border(dsize, borderthickness=10):
    canvas = create_whiteboard(dsize)
    canvas[borderthickness:-borderthickness, borderthickness:-borderthickness] = 0
    return canvas


def draw_cross(dsize, linelength=50, linethickness=1):
    canvas = create_canvas(dsize)
    row, col = dsize
    cx, cy = col // 2, row // 2
    cv.line(canvas, (cx - linelength // 2, cy), (cx + linelength // 2, cy), 255, linethickness)
    cv.line(canvas, (cx, cy - linelength // 2), (cx, cy + linelength // 2), 255, linethickness)
    return canvas


def draw_gradient(dsize, gap: int):
    grad_v_mat, grad_h_mat = np.mgrid[64:256:gap, 64:256:gap].astype(np.uint8)
    grad_h = mat2GridImg(grad_h_mat, dsize)
    grad_v = mat2GridImg(grad_v_mat, dsize)
    info = 'gradient: \nshape:{} \ndtype:{}\n'
    info = info.format(grad_h.shape, grad_h.dtype)
    print(info)
    return grad_h, grad_v


def draw_calibboard(dsize, mshape):
    r, c = mshape
    row, col = dsize
    r_gap = row // (r - 1)
    c_gap = col // (c - 1)
    y, x = np.mgrid[0:row + 1:r_gap, 0:col + 1:c_gap].astype(np.float32)
    x[:, 0] = x[:, 0] + 6
    x[:, -1] = x[:, -1] - 6

    y[0, :] = y[0, :] + 6
    y[-1, :] = y[-1, :] - 6

    pnts = np.stack([x, y], axis=2).reshape(-1, 2)

    return pnts


def gen_pixel_coords(dsize):
    row, col = dsize
    height = np.arange(0, row)
    width = np.arange(0, col)
    x, y = np.meshgrid(width, height)
    coords = np.stack([x, y], axis=2)

    return coords


def imDecorate(src, pendant, position: cv.CV_32F, scale=1, angle=0, transparency=1):
    """
        This method is to decorate source image using given image.
    :param src: image needed to decorate
    :param pendant: single element used to decorate src
    :param position: the place where the center of pendant should be placed
    :param scale: scale pendant
    :param angle: rotate angle, positive is clockwise-counter
    :param transparency: adjust transparency
    :return: a new image that is decorated by pendant
    """
    assert src.ndim == pendant.ndim, f'src.dim{src.ndim} != pendant.dim{pendant.ndim}'

    # get center of src
    row, col = src.shape[:2]
    center = np.divide((col, row), 2).astype(np.float32)  # center(x, y)

    print(f'Position:{position}', end='')

    # scale
    if 0 < scale != 1:
        print(f' | Scale:{scale}', end='')
        interpolation = cv.INTER_CUBIC if scale > 1 else cv.INTER_AREA
        pendant = cv.resize(pendant, None, fx=scale, fy=scale, interpolation=interpolation)

    # get center of pendant
    row_p, col_p = pendant.shape[:2]
    center_p = np.divide((col_p, row_p), 2).astype(np.float32)  # center(x, y)

    # rotate, first step is to move pendant to the center of src, in order to avoid
    # lost pixels on the edges while rotating
    if angle != 0:
        print(f' | Angle:{angle}', end='')
        tx, ty = center - center_p
        tran_mat = np.matrix([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        pendant = cv.warpAffine(pendant, tran_mat, (col, row), cv.BORDER_TRANSPARENT)

        rot_mat = cv.getRotationMatrix2D(center, angle, scale=1)
        pendant = cv.warpAffine(pendant, rot_mat, (col, row), cv.BORDER_TRANSPARENT)

    # translate
    tx, ty = position - center
    tran_mat = np.matrix([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    pendant = cv.warpAffine(pendant, tran_mat, (col, row), cv.BORDER_TRANSPARENT)

    # concat
    print(f' | Transparency:{transparency}')
    target = cv.addWeighted(src, 1, pendant, transparency, 0)
    return target
