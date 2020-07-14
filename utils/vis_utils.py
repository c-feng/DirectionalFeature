import numpy as np
import colorsys
import random
import cv2

def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors

def ncolors(num):
    rgb_colors = []
    if num < 1:
        return np.array(rgb_colors)
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        rgb_colors.append([_r, _g, _b])
        # r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        # rgb_colors.append([r, g, b])

    return np.array(rgb_colors)

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # random.shuffle(colors)
    return colors

def mask2png(mask, file_name=None, suffix="png"):
    """ mask: (w, h)
        img_rgb: (w, h, rgb)
    """
    nums = np.unique(mask)[1:]
    if len(nums) < 1:
        colors = np.array([[0,0,0]])
    else:
        # colors = ncolors(len(nums))
        colors = (np.array(random_colors(len(nums))) * 255).astype(int)
        colors = np.insert(colors, 0, [0,0,0], 0)
    
    # 保证mask中的值为1-N连续
    mask_ordered = np.zeros_like(mask)
    for cnt, l in enumerate(nums, 1):
        mask_ordered[mask==l] = cnt

    im_rgb = colors[mask_ordered.astype(int)]
    if file_name is not None:
        cv2.imwrite(file_name+"."+suffix, im_rgb[:, :, ::-1])
    return im_rgb

def apply_mask(image, mask, color, alpha=0.5, scale=1):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * scale,
                                  image[:, :, c])
    return image

def img_mask_png(image, mask, file_name=None, alpha=0.5, suffix="png"):
    """ mask: (h, w)
        image: (h, w, rgb)
    """
    nums = np.unique(mask)[1:]
    if len(nums) < 1:
        colors = np.array([[0,0,0]])
    else:
        colors = ncolors(len(nums))
        colors = np.insert(colors, 0, [0,0,0], 0)
    
    # 保证mask中的值为1-N连续
    mask_ordered = np.zeros_like(mask)
    for cnt, l in enumerate(nums, 1):
        mask_ordered[mask==l] = cnt
    
    # mask_rgb = colors[mask_ordered.astype(int)]
    mix_im = image.copy()
    for i in np.unique(mask_ordered)[1:]:
        mix_im = apply_mask(mix_im, mask_ordered==i, colors[int(i)], alpha=alpha, scale=255)

    if file_name is not None:
        cv2.imwrite(file_name+"."+suffix, mix_im[:, :, ::-1])
    return mix_im

def _find_contour(mask):
    # _, contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 顶点
    _, contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cont = np.zeros_like(mask)
    for contour in contours:
        cont[contour[:,:,1], contour[:,:,0]] = 1
    return cont

def masks_to_contours(masks):
    # 包含多个区域
    nums = np.unique(masks)[1:]
    cont_mask = np.zeros_like(masks)
    for i in nums:
        cont_mask += _find_contour(masks==i)
    return (cont_mask>0).astype(int)
    
def batchToColorImg(batch, minv=None, maxv=None, scale=255.):
    """ batch: (N, H, W, C)
    """
    if batch.ndim == 3:
        N, H, W = batch.shape
    elif batch.ndim == 4:
        N, H, W, _ = batch.shape
    colorImg = np.zeros(shape=(N, H, W, 3))
    for i in range(N):
        if minv is None:
            a = (batch[i] - batch[i].min()) / (batch[i].max() - batch[i].min()) * 255
        else:
            a = (batch[i] - minv) / (maxv - minv) * scale
        a = cv2.applyColorMap(a.astype(np.uint8), cv2.COLORMAP_JET)
        colorImg[i, ...] = a[..., ::-1] / 255.
    return colorImg

if __name__ == "__main__":
    a = np.zeros((100, 100))
    a[0:5, 3:8] = 1
    a[75:85, 85:95] = 2

    # colors = ncolors(2)[::-1]
    colors = np.array(random_colors(2)) * 255
    colors = np.insert(colors, 0, [0, 0, 0], 0)

    b = colors[a.astype(int)].astype(np.uint8)
    import cv2, skimage
    # skimage.io.imsave("test_io.png", b)
    # # cv2.imwrite("test.jpg", b[:, :, ::-1])
    # print()

    # mask2png(a, "test")

    ################################################
    # img_mask_png(b, a, "test")

    #############################################
    # cont_mask = find_contours(a)
    # print()
    # # skimage.io.imsave("test_cont.png", cont_mask) 
    # b[cont_mask>0, :] = [255, 255, 255]
    # skimage.io.imsave("test_cont.png", b)

    gt0 = skimage.io.imread("gt0.png", as_gray=False)
    print()
    gt0[gt0==54] = 0
    # cont_mask = find_contours(gt0==237)  # Array([ 18,  54,  73, 237], dtype=uint8)
    # cont_mask += find_contours(gt0==18)
    # cont_mask += find_contours(gt0==73)
    cont_mask = masks_to_contours(gt0)
    
    colors = np.array(random_colors(1)) * 255
    colors = np.insert(colors, 0, [0, 0, 0], 0)
    cont_mask = colors[cont_mask.astype(int)].astype(np.uint8)

    skimage.io.imsave("test_cont.png", cont_mask)
