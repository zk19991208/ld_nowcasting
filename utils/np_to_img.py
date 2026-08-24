import numpy as np
from PIL import Image
file =r'D:\Code\pl_GAN\work\save\PwafsRadarPrec\unet_js\case\0000\unet.npy'

da = np.load(file)
print(da.max(), da.min())
for i, img in enumerate(da):
    img = img[0] * 255
    img = Image.fromarray(img.astype('uint8'))
    filename = "d:/code/pl_GAN/work/save/PwafsRadarPrec/unet_js/case/0000/unet_%d.png" %i
    img.save(filename)
    print("save to", filename)


file =r'D:\Code\pl_GAN\work\save\PwafsRadarPrec\unet_js\case\0000\truth.npy'
da = np.load(file)
print(da.max(), da.min())
for i, img in enumerate(da):
    img = img[0] * 255
    img = Image.fromarray(img.astype('uint8'))
    filename = "d:/code/pl_GAN/work/save/PwafsRadarPrec/unet_js/case/0000/truth_%d.png" %i
    img.save(filename)
    print("save to", filename)