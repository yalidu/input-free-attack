from PIL import Image
import matplotlib
import numpy as np
from tensorflow.python.client import device_lib
import tensorflow as tf

matplotlib.use('Agg')

IMAGENET_PATH = ' '
NUM_LABELS = 1000
SIZE = 299

image_size = SIZE
orig_class = 111  # without backgound class its 111


def one_hot(index, total):
    arr = np.zeros((total))
    arr[index] = 1.0
    return arr


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def show(img, name="output.png"):
    fig = np.around((img) * 255)
    fig = fig.astype(np.uint8).squeeze()
    pic = Image.fromarray(fig)
    # pic.resize((512,512), resample=PIL.Image.BICUBIC)
    pic.save(name)
    remap = "  .*#" + "#" * 100
    img = (img.flatten() + .5) * 3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i * 28:i * 28 + 28]]))
