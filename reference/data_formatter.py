import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm


if not os.path.exists('cifar-100-python/dataset'):
    os.makedirs('cifar-100-python/dataset/train')
    os.makedirs('cifar-100-python/dataset/test')

    for i in range(100):
        os.makedirs(f'cifar-100-python/dataset/train/{str(i).zfill(2)}')
        os.makedirs(f'cifar-100-python/dataset/test/{str(i).zfill(2)}')


def load_cifar100_batch(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict[b'data'], dict[b'fine_labels']


with open('meta', 'rb') as f:
    meta_data = pickle.load(f, encoding='bytes')
fine_label_names = meta_data[b'fine_label_names']


train_data, train_labels = load_cifar100_batch('train')
test_data, test_labels = load_cifar100_batch('test')


def save_images(data, labels, folder):
    for i, (img_data, label) in enumerate(tqdm(zip(data, labels), total=len(data), desc=f"整理 {folder} 中...")):
        img = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
        img_folder = os.path.join(
            'cifar-100-python/dataset', folder, str(label).zfill(2))
        img_path = os.path.join(img_folder, f'{i}.png')
        plt.imsave(img_path, img)


save_images(train_data, train_labels, 'train')
save_images(test_data, test_labels, 'test')
