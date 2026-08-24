import numpy as np
from PIL import Image
from matplotlib import cm, colors


def array2img(array, cmap, vmin=0, vmax=70, rgb=False):
    """
    :params array(np.array):
    :params cmap(str): colormap name
    :params vmin(int): vmin for normalize
    :params vmax(int): vmax for normalize
    :params rgb(bool): if convert image to rgb format

    Ref:
      - https://stackoverflow.com/questions/10965417/how-to-convert-a-numpy-array-to-pil-image-applying-matplotlib-colormap
    """
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    if isinstance(cmap, str):
        cms = cm.get_cmap(cmap)
    elif isinstance(cmap, list) or isinstance(cmap, np.ndarray):
        cms = colors.LinearSegmentedColormap.from_list('cmap', cmap)
    else:
        raise ValueError(f'Unknown type {type(cmap)}.')

    if rgb:
        return Image.fromarray(np.uint8(cms(norm(array)) * 255)).convert('RGB')
    else:
        return Image.fromarray(np.uint8(cms(norm(array)) * 255))


def array2video(array, cmap=None, vmin=0, vmax=70, rgb=False):
    """
    :params array(np.array):
    :params cmap(str): colormap name
    :params vmin(int): vmin for normalize
    :params vmax(int): vmax for normalize
    :params rgb(bool): if convert image to rgb format
    """
    if cmap is None:
        cmap = 'NWSRef'

    bs, frames, channels, height, width = array.shape

    if channels != 1:
        raise ValueError('The number of channels should be 1!')

    if rgb:
        channel = 3
    else:
        channel = 4

    imgs = np.full([bs, frames, height, width, channel], 255, dtype=np.uint8)
    for i in range(bs):
        for j in range(frames):
            imgs[i, j] = np.array(array2img(array[i, j, 0], cmap, vmin=vmin, vmax=vmax, rgb=rgb))

    return imgs.transpose([0, 1, 4, 2, 3])


def togrey(array, size=None, save=None):
    """Convert numpy array to greyscale
    :param array(np.array):
    :param size(tuple):
    :param save(str):
    """
    img = Image.fromarray(np.uint8(array), 'L')

    if size is not None:
        img = img.resize(size, Image.ANTIALIAS)

    if save is not None:
        img2.save(f'{save}')
