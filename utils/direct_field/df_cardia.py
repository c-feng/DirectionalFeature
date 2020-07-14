import numpy as np
from scipy import ndimage
import math
import cv2
from PIL import Image

def direct_field(a, norm=True):
    """ a: np.ndarray, (h, w)
    """
    if a.ndim == 3:
        a = np.squeeze(a)

    h, w = a.shape

    a_Image = Image.fromarray(a)
    a = a_Image.resize((w, h), Image.NEAREST)
    a = np.array(a)
    
    accumulation = np.zeros((2, h, w), dtype=np.float32)
    for i in np.unique(a)[1:]:
        # b, ind = ndimage.distance_transform_edt(a==i, return_indices=True)
        # c = np.indices((h, w))
        # diff = c - ind
        # dr = np.sqrt(np.sum(diff ** 2, axis=0))

        img = (a == i).astype(np.uint8)
        dst, labels = cv2.distanceTransformWithLabels(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)
        index = np.copy(labels)
        index[img > 0] = 0
        place = np.argwhere(index > 0)
        nearCord = place[labels-1,:]
        x = nearCord[:, :, 0]
        y = nearCord[:, :, 1]
        nearPixel = np.zeros((2, h, w))
        nearPixel[0,:,:] = x
        nearPixel[1,:,:] = y
        grid = np.indices(img.shape)
        grid = grid.astype(float)
        diff = grid - nearPixel
        if norm:
            dr = np.sqrt(np.sum(diff**2, axis = 0))
        else:
            dr = np.ones_like(img)

        # direction = np.zeros((2, h, w), dtype=np.float32)
        # direction[0, b>0] = np.divide(diff[0, b>0], dr[b>0])
        # direction[1, b>0] = np.divide(diff[1, b>0], dr[b>0])

        direction = np.zeros((2, h, w), dtype=np.float32)
        direction[0, img>0] = np.divide(diff[0, img>0], dr[img>0])
        direction[1, img>0] = np.divide(diff[1, img>0], dr[img>0])

        accumulation[:, img>0] = 0
        accumulation = accumulation + direction
    
    # mag, angle = cv2.cartToPolar(accumulation[0, ...], accumulation[1, ...])
    # for l in np.unique(a)[1:]:
    #     mag_i = mag[a==l].astype(float)
    #     t = 1 / mag_i * mag_i.max()
    #     mag[a==l] = t
    # x, y = cv2.polarToCart(mag, angle)
    # accumulation = np.stack([x, y], axis=0)

    return accumulation


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # gt_p = "/home/ffbian/chencheng/XieheCardiac/npydata/dianfen/16100000/gts/16100000_CINE_segmented_SAX_b3.npy"
    # gt = np.load(gt_p)[..., 9]  # uint8
    # print(gt.shape)

    # a_Image = Image.fromarray(gt)
    # a = a_Image.resize((224, 224), Image.NEAREST)
    # a = np.array(a)  # uint8
    # print(a.shape, np.unique(a))

    # # plt.imshow(a)
    # # plt.show()

    # ############################################################
    # direction = direct_field(gt)
    
    # theta = np.arctan2(direction[1,...], direction[0,...])
    # degree = theta * 180 / math.pi
    # degree = (degree + 180) / 360

    # plt.imshow(degree)
    # plt.show()

    ########################################################
    import json, time, pdb, h5py
    data_list = "/home/ffbian/chencheng/XieheCardiac/2DUNet/UNet/libs/datasets/train_new.json"
    data_list = "/root/chengfeng/Cardiac/source_code/libs/datasets/jsonLists/acdcList/Dense_TestList.json"
    with open(data_list, 'r') as f:
        data_infos = json.load(f)
    
    mag_stat = []
    st = time.time()
    for i, di in enumerate(data_infos):
        # img_p, times_idx = di
        # gt_p = img_p.replace("/imgs/", "/gts/")
        # gt = np.load(gt_p)[..., times_idx]
        
        img = h5py.File(di,'r')['image']
        gt = h5py.File(di,'r')['label']
        gt = np.array(gt).astype(np.float32)

        print(gt.shape)
        direction = direct_field(gt, False)
        # theta = np.arctan2(direction[1,...], direction[0,...])
        mag, angle = cv2.cartToPolar(direction[0, ...], direction[1, ...])
        # degree = theta * 180 / math.pi
        # degree = (degree + 180) / 360
        degree = angle / (2 * math.pi) * 255
        # degree = (theta - theta.min()) / (theta.max() - theta.min()) * 255
        # mag = np.sqrt(np.sum(direction ** 2, axis=0, keepdims=False))
        

        # 归一化
        # for l in np.unique(gt)[1:]:
        #     mag_i = mag[gt==l].astype(float)
        #     # mag[gt==l] = 1. - mag[gt==l] / np.max(mag[gt==l])
        #     t = (mag_i - mag_i.min()) / (mag_i.max() - mag_i.min())
        #     mag[gt==l] = np.exp(-10*t)
        #     print(mag_i.max(), mag_i.min())

        # for l in np.unique(gt)[1:]:
        #     mag_i = mag[gt==l].astype(float)
        #     t = 1 / (mag_i) * mag_i.max()
        #     # t = np.exp(-0.8*mag_i) * mag_i.max()
        #     # t = 1 / np.sqrt(mag_i+1) * mag_i.max()
        #     mag[gt==l] = t
        #     # print(mag_i.max(), mag_i.min())

        # mag[mag>0] = 2 * np.exp(-0.8*(mag[mag>0]-1))
        # mag[mag>0] = 2 * np.exp(0.8*(mag[mag>0]-1))


        mag_stat.append(mag.max())
        # pdb.set_trace()

        # plt.imshow(degree)
        # plt.show()

        ######################
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(degree)
        axs[1].imshow(gt)
        axs[2].imshow(mag)
        plt.show()

        ######################
        if i % 100 == 0:
            print("\r\r{}/{}  {:.4}s".format(i+1, len(data_infos), time.time()-st))
    print()

    print("total time: ", time.time()-st)
    print("Average time: ", (time.time()-st) / len(data_infos))
    # total time:  865.811030626297
    # Average time:  0.012969593759126428

    plt.hist(mag_stat)
    plt.show()
    print(mag_stat)