from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import pandas as pd
import os
from PIL import Image
import torchvision
import re
import numpy as np

class CostomData(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        transform = torchvision.transforms.ToTensor()
        data = transform(Image.open(self.data[idx]))
        label = self.labels[idx]
        return data, label

def search_png_file():
    cifar_100_python_path = '.\\cifar-100-python\\'
    image_list = []
    label_list = []
    for root, dir, file in os.walk(cifar_100_python_path):
        for file_name in file:
            if ".png" in file_name:
                image_list.append(os.path.join(root, file_name))
                label_list.append(root.split('\\')[-1])

    return image_list, label_list

def filter_keyword():
    file_list, _ = search_png_file()
    pattern = re.compile(r'^.+_\d{6}\.png$') 
    matching_file = [file for file in file_list if not pattern.match(file)]
    for file in matching_file:
        print(file)
    print(len(matching_file))

def k_fold_training_and_save_csv():
    '''
        KFold演算法，將資料做切分，n_split=5即表示4筆train、1筆test，並做5次分割
    '''
    save_path = '.\\cifar-100-python\\k_fold_csv'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_list, label_list = search_png_file()
    file_list = np.array(file_list)
    label_list = np.array(label_list)

    kf = KFold(n_splits=5, shuffle=True)
    fold=1
    for train_index, test_index in kf.split(file_list):
        data_train, data_test = file_list[train_index], file_list[test_index]
        label_train, label_test = label_list[train_index], label_list[test_index]
        
        ## 儲存資料
        train_dataFrame = pd.DataFrame({'Filename':data_train, 'Label':label_train})
        test_dataFrame = pd.DataFrame({'Filename':data_test, 'Label':label_test})

        train_dataFrame.to_csv(f'{save_path}\\train_fold_{fold}.csv', index=False)
        test_dataFrame.to_csv(f'{save_path}\\test_fold_{fold}.csv', index=False)
        fold+=1


def LoadData():
    images, labels = search_png_file()
    dataset = CostomData(images, labels)
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    for idx, (images, labels) in enumerate(dataloader):
        print(idx, images, labels)

if __name__ == '__main__':
    # search_png_file()
    LoadData()