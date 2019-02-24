"""Module to read data sets"""
import os
import gzip
import shutil
from urllib.request import urlretrieve
from urllib.parse import urlparse
import struct

import numpy as np

MNIST_TRAIN_IMAGES_URL = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
MNIST_TRAIN_LABELS_URL = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
MNIST_TEST_IMAGES_URL = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
MNIST_TEST_LABELS_URL = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

def download_data(url, filename='', force_download=False):
    """Download and cache the data from url into filename

    Parameters
    ~~~~~~~~~~
    url : string
        web location of the data
    filename : string (optional)
        location to save the data
    force_download : bool
        if True, force the redownload of the data

    Returns
    ~~~~~~~
    filename : string
        returns location of saved data
    """
    # If filename is not passed, get filename from URL
    if filename == '':
        filename = os.path.basename(urlparse(url).path)

    if force_download or not os.path.exists(filename):
        urlretrieve(url, filename)

    return filename

def get_mnist_data(url):
    """Grab NIST data or labels from url

    Parameters
    ~~~~~~~~~~
    url : string
        web location of the data

    Returns
    ~~~~~~~
    data :
        [num_images x row x col] numpy array of images from dataset if data is image set
        [num_labels] numpy array of labels from dataset if data is label set
    """
    filename = download_data(url)
    basename = os.path.splitext(filename)[0]
    with gzip.open(filename, 'rb') as s_file, \
            open(basename, 'wb') as d_file:
        shutil.copyfileobj(s_file, d_file, 65536)

    with open(basename, 'rb') as fdata:
        magic = struct.unpack(">I", fdata.read(4))[0]
        # Image Data Set
        if magic == 2051:
            num, rows, cols = struct.unpack(">III", fdata.read(12))
            data = np.fromfile(fdata, dtype=np.uint8).reshape(num, rows, cols)
        # Label Data Set
        elif magic == 2049:
            num = struct.unpack(">I", fdata.read(4))[0]
            data = np.fromfile(fdata, dtype=np.uint8)
        else:
            raise Exception('URL return neither image or label dataset')
        return data
