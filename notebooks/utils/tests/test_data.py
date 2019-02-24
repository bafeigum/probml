from utils import read_data
import numpy as np

def test_mnist_images():
    train_images = read_data.get_mnist_data(read_data.MNIST_TRAIN_IMAGES_URL)
    assert train_images.shape == (60000, 28, 28)
    test_images = read_data.get_mnist_data(read_data.MNIST_TEST_IMAGES_URL)
    assert test_images.shape == (10000, 28, 28)

def test_mnist_labels():
    train_labels = read_data.get_mnist_data(read_data.MNIST_TRAIN_LABELS_URL)
    assert train_labels.shape == (60000, )
    test_labels = read_data.get_mnist_data(read_data.MNIST_TEST_LABELS_URL)
    assert test_labels.shape == (10000, )
