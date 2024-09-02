import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm


def create_directories(base_path):
    if not os.path.exists(base_path):
        os.makedirs(os.path.join(base_path, 'train'))
        os.makedirs(os.path.join(base_path, 'test'))

        for i in range(100):
            os.makedirs(os.path.join(base_path, 'train', str(i).zfill(2)))
            os.makedirs(os.path.join(base_path, 'test', str(i).zfill(2)))


def load_cifar100_batch(file):
    with open(file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data']
    fine_labels = batch[b'fine_labels']
    filenames = batch[b'filenames']
    return data, fine_labels, filenames


def save_images(data, labels, filenames, folder, base_path):
    for img_data, label, filename in tqdm(zip(data, labels, filenames), total=len(data), desc=f"整理 {folder} 中..."):
        img = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
        img_folder = os.path.join(base_path, folder, str(label).zfill(2))

        if isinstance(filename, bytes):
            filename = filename.decode('utf-8')

        img_path = os.path.join(img_folder, filename)
        plt.imsave(img_path, img)


def main():
    base_path = 'cifar-100-python/dataset'
    create_directories(base_path)

    with open('meta', 'rb') as f:
        meta_data = pickle.load(f, encoding='bytes')
    fine_label_names = [label.decode('utf-8')
                        for label in meta_data[b'fine_label_names']]

    train_data, train_labels, train_filenames = load_cifar100_batch('train')

    test_data, test_labels, test_filenames = load_cifar100_batch('test')

    save_images(train_data, train_labels, train_filenames, 'train', base_path)
    save_images(test_data, test_labels, test_filenames, 'test', base_path)


if __name__ == "__main__":
    main()
