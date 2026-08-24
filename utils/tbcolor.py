import torch
from imageio import imwrite

dBZ2COLOR = torch.tensor([[255, 255, 255],
                          [0, 204, 255],
                          [0, 102, 255],
                          [0, 51, 204],
                          [0, 255, 102],
                          [51, 204, 102],
                          [0, 153, 0],
                          [255, 255, 102],
                          [255, 204, 51],
                          [255, 153, 0],
                          [255, 102, 102],
                          [255, 51, 51],
                          [204, 0, 0],
                          [255, 0, 255],
                          [205, 0, 205],
                          [128, 0, 128], ], dtype=torch.uint8)


def condtion_plot_image(x, vmin=0, vmax=70):
    """
    根据反射率的大小画伪色彩
    :param x:
    :param thresh:
    :return:
    """
    all_condtion = []
    num = 17
    thresh = torch.linspace(vmin, vmax, num)
    for idx in range(1, num):
        all_condtion.append((x > thresh[idx - 1]) & (x <= thresh[idx]))
    return all_condtion


def img2color_image(dBZ, image_path=None):
    """
    将雷达的灰度图转化为伪彩图
    :param pixel: 二维的图像，[nheight, nwidth]， 范围为[0,70] dBZ
    :param image_path: 图像的保存路径
    :return:
    """
    nheight, nwidth = dBZ.shape
    img = torch.full([nheight, nwidth, 3], 255, dtype=torch.uint8)
    for idx, icond in enumerate(condtion_plot_image(dBZ)):
        img[icond, :] = dBZ2COLOR[idx]
    imwrite(image_path, img)


def get_rgb_img(dBZ, vmin=-0.2, vmax=5):
    """
        将雷达的灰度图转化为伪彩图
        :param pixel: 二维的图像，[nheight, nwidth]， 范围为[0,70] dBZ
        :param image_path: 图像的保存路径
        :return:
    """
    nheight, nwidth = dBZ.shape
    img = torch.full([nheight, nwidth, 3], 255, dtype=torch.uint8)
    for idx, icond in enumerate(condtion_plot_image(dBZ, vmin, vmax)):
        img[icond, :] = dBZ2COLOR[idx]
    return img


def get_rgb_img_4d(dBZ, vmin=-0.2, vmax=5):
    """
        将雷达的灰度图转化为伪彩图
        :param pixel: 二维的图像，[nheight, nwidth]， 范围为[0,70] dBZ
        :param image_path: 图像的保存路径
        :return:
    """
    nbatch, ntime, nheight, nwidth = dBZ.shape
    dBZ = torch.clip(dBZ, vmin, vmax)
    img = torch.full([nbatch, ntime, nheight, nwidth, 3], 255, dtype=torch.uint8)
    for i in range(nbatch):
        for j in range(ntime):
            img[i, j, ...] = get_rgb_img(dBZ[i, j, ...], vmin, vmax)
    return img.permute(0, 1, 4, 2, 3)


if __name__ == "__main__":
    x = torch.rand(2, 20, 480, 480, device="cuda:0") * 8
    y = get_rgb_img_4d(x)
    print(y)
