from PIL import Image
import glob
import numpy as np
from matplotlib import pyplot as plt
import random

IMAGE_SIZE = 224
NUM_LABELS = 200

def load_filenames (data_dir):
    """
    Load filenames of each class from a directory
    :param data_dir: path to the directory where files are
    :return: [[class0_file0, class0_file1, ...][class1_file0, ...]...[classN_file0, ...]]
    """
    filenames = []
    for i, class_path in enumerate(glob.glob(data_dir + '/*')):
        print(i, class_path)
        curr_images = []
        for file_path in glob.glob(class_path + '/*'):
            curr_images.append(file_path)

        filenames.append(curr_images)

    return filenames

def train_test_split(filenames, train_percentage, val_percentage):
    """
    Split the dataset into train, val, and test.
    :param filenames: a list of lists, where each list contains filepaths of a class
    :param train_percentage: the percentage of training set
    :param val_percentage: the percentage of validation set out of the training set
    :return: train, val, and test set of the same type as the input
    """
    train_filenames = []
    valid_filenames = []
    test_filenames = []
    for class_files in filenames:
        val_start_idx = int(len(class_files) * train_percentage * (1 - val_percentage))
        test_start_idx = int(len(class_files) * train_percentage)

        class_files.sort()
        random.seed(230)
        random.shuffle(class_files)

        train_filenames.append(class_files[:val_start_idx])
        valid_filenames.append(class_files[val_start_idx:test_start_idx])
        test_filenames.append(class_files[test_start_idx:])

    return(train_filenames, valid_filenames, test_filenames)

def one_hot_encode(x, num_labels):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    return np.eye(num_labels)[x]

def img2arr(filename, img_size):
    img = Image.open(filename).convert('L')
    img = img.resize((img_size, img_size))
    img = np.array(img)

    return img


def get_min_len(filenames):
    print(min([len(label_files) for label_files in filenames]))


def get_batch(filenames, batch_size):
    """
    Randomly select 'batch_size' of images per class from training set(flatten), encode labels, and make images into np array
    :param batch_size:
    :param filenames:
    :return: batch_x(np array) and batch_y(np array of onehot encoded labels)
    """
    # randomly select 'batch_size' of data from each class
    batch_names = []
    batch_labels = []
    for label in range(len(filenames)):
        # print(label)
        randomly_selected = random.sample(filenames[label], batch_size)
        for filename in randomly_selected:
            batch_names.append(filename)
            batch_labels.append(label)

    # encode labels
    batch_encoded_labels = one_hot_encode(batch_labels, NUM_LABELS)

    # read in image files
    batch_images = []
    for filename in batch_names:
        # print(filename)
        # open, read, and resize image files
        img = img2arr(filename, IMAGE_SIZE)
        # print(img.shape)
        batch_images.append(img)

    # reshape 'batch_images'
    batch_images = np.array(batch_images)
    # print("batch_images shape = ", batch_images.shape)
    batch_images = batch_images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

    return(batch_images, batch_encoded_labels)

def read_data(filenames):
    """
    Read all the images in filenames(flatten), encode labels, and make images into np array
    :param filenames:
    :return: images(np array) and labels(np array of onehot encoded labels)
    """
    names = []
    labels = []
    for label in range(len(filenames)):
        for filename in filenames[label]:
            names.append(filename)
            labels.append(label)

    encoded_labels = one_hot_encode(labels, NUM_LABELS)

    # read in image files
    images = []
    for filename in names:
        # print(filename)
        # open, read, and resize image files
        img = img2arr(filename, IMAGE_SIZE)
        # print(img.shape)
        images.append(img)

    # reshape 'batch_images'
    images = np.array(images)
    # print("batch_images shape = ", batch_images.shape)
    images = images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

    return(images, encoded_labels)

#
#
# img = Image.open(train[0][0])
# img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
# img = np.array(img)
# plt.imshow(img)