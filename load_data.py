import pickle
import os
import cv2

cifar_train_path = '.\\cifar-100-python\\train'
cifar_test_path = '.\\cifar-100-python\\test'
cifar_meta_path = '.\\cifar-100-python\\meta'

cifar_dataset_base_path = 'cifar-100-python\\dataset'
cifar_train_dataset = os.path.join(cifar_dataset_base_path, 'train')
cifar_test_dataset = os.path.join(cifar_dataset_base_path, 'test')

if not os.path.exists(cifar_dataset_base_path):
    os.makedirs(cifar_train_dataset)
    os.makedirs(cifar_test_dataset)

    for i in range(100):
        os.makedirs(f'{cifar_train_dataset}\\{str(i).zfill(2)}')
        os.makedirs(f'{cifar_test_dataset}\\{str(i).zfill(2)}')

def load_cifar_data(cifar_path):
    with open(cifar_path, 'rb') as file:
        '''
            data[b'fine_labels'] -> 數字標籤
            data[b'filenames'] -> png檔名
        '''
        data = pickle.load(file, encoding='bytes')
    return data[b'filenames'], data[b'fine_labels'], data[b'data']


with open(cifar_meta_path, 'rb') as file:
    meta_data = pickle.load(file, encoding='bytes')
keywords = meta_data[b'fine_label_names']

train_filenames, train_fine_labels, train_images = load_cifar_data(cifar_train_path)
test_filenames, test_fine_labels, test_images = load_cifar_data(cifar_test_path)

def main(filenames, fine_labels, images, save_fold):
    for (label, filename, image) in zip(fine_labels, filenames, images):
        label =str(label).zfill(2)
        print(label, filename.decode('utf-8'), image)
        save_path = os.path.join(os.path.join(save_fold, label), filename.decode('utf-8'))
        image = image.reshape(3, 32, 32).transpose(1, 2, 0)
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path, image)
    

if __name__ == '__main__':
    main(train_filenames, train_fine_labels, train_images, cifar_train_dataset)
    # main(test_filenames, test_fine_labels, test_images, cifar_test_dataset)