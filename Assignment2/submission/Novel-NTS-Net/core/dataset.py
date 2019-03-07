import numpy as np
import scipy.misc
import os
from PIL import Image
from torchvision import transforms
from config import INPUT_SIZE


class CUB():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        if self.is_train:
            self.train_img = [scipy.misc.imread(os.path.join(self.root, 'images', train_file)) for train_file in
                              train_file_list[:data_len]]
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        if not self.is_train:
            self.test_img = [scipy.misc.imread(os.path.join(self.root, 'images', test_file)) for test_file in
                             test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)
class Predict_data():
    def __init__(self, image_file,label=0, data_len=None):
        self.image_file = image_file
        # label_txt_file = open(os.path.join(self.root, 'test_image_class_labels.txt'))
        # train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_txt_file = [line.rstrip('\n') for line in open(image_file)]
        img_name_list = []
        img_class_list = [int(line.rstrip('\n')) for line in open('coarse_output.txt')]
        label_list = []
        for line in img_txt_file:
            img_name_list.append(line)
            label_list.append(label)

        # train_test_list = []
        # for line in train_val_file:
            # train_test_list.append(int(line[:-1].split(' ')[-1]))
        # test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        test_file_list = [[i, j] for i, j in zip(img_name_list, img_class_list)]

        self.test_img = [[scipy.misc.imread(os.path.join('test_images', test_file[0])), test_file[1]] for test_file in test_file_list]
        self.test_label = [x for x in (label_list)][:data_len]

    def __getitem__(self, index):
        img, target = self.test_img[index][0], self.test_label[index]
        imclass = self.test_img[index][1]
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        img = transforms.Resize((600, 600), Image.BILINEAR)(img)
        img = transforms.CenterCrop(INPUT_SIZE)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        #print("img", type(img))
        # print(imclass)
        # print(target.shape)
        return img, imclass, target

    def __len__(self):
        return len(self.test_label)

class Predict():
    def __init__(self, image, label=0, data_len=None):
        self.image = image
        # img_txt_file = open(os.path.join(self.root, 'test_images.txt'))
        # label_txt_file = open(os.path.join(self.root, 'test_image_class_labels.txt'))
        # train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        img_name_list.append(image)
        # for line in img_txt_file:
            # img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        label_list.append(label)
        # for line in label_txt_file:
            # label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        # train_test_list = []
        # for line in train_val_file:
            # train_test_list.append(int(line[:-1].split(' ')[-1]))
        # test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        test_file_list = img_name_list

        self.test_img = test_file_list
        self.test_img = [scipy.misc.imread(test_file) for test_file in test_file_list]
        self.test_label = label_list
        # self.test_label = [x for x in (label_list)][:data_len]

    def __getitem__(self, index):
        img, target = self.test_img[index], self.test_label[index]
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        img = transforms.Resize((600, 600), Image.BILINEAR)(img)
        img = transforms.CenterCrop(INPUT_SIZE)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        return len(self.test_label)


if __name__ == '__main__':
    dataset = CUB(root='./CUB_200_2011')
    print(len(dataset.train_img))
    print(len(dataset.train_label))
    for data in dataset:
        print(data[0].size(), data[1])
    dataset = CUB(root='./CUB_200_2011', is_train=False)
    print(len(dataset.test_img))
    print(len(dataset.test_label))
    for data in dataset:
        print(data[0].size(), data[1])