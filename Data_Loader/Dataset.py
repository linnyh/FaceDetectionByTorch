
import torch
import torchvision.transforms as transforms
import torch.utils.data as Data
from PIL import Image
import numpy as np
import os


# 加载单张图片
def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Can not open {0}".format(path))


# 重载数据集类
class myDataset(Data.DataLoader):
    def __init__(self, img_dir, img_txt, transform=None, loader=default_loader):
        # 由于数据集很大，采用小批量加载到内存中
        img_list = [] # 存储图像矩阵
        img_labels = []  # 存储标签矩阵

        fp = open(img_txt, 'r') # 只读模式加载txt文件，文件中每行的内容为图片文件名以及标签
        for line in fp.readlines(): # 每次读取文件中的一行(即一个文件名)
            if len(line.split()) != 41:
                continue
            img_list.append(line.split()[0]) # 文件名添加到list中
            img_label_single = []            # 单张图片的标签
            for value in line.split()[1:]:
                if value == '-1':
                    img_label_single.append(0)
                if value == '1':
                    img_label_single.append(1)
            img_labels.append(img_label_single)
        self.imgs = [os.path.join(img_dir, file) for file in img_list] # 得到图片的相对路径
        self.labels = img_labels
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = torch.from_numpy(np.array(self.labels[index], dtype=np.int64))
        img = self.loader(img_path) # 将图片转为3通道
        if self.transform is not None:
            try:
                img = self.transform(img) # 数据处理
            except:
                print('Cannot transform image: {}'.format(img_path))
        return img, label


transform = transforms.Compose([
    transforms.Resize(40),              # 图像缩小
    transforms.CenterCrop(32),          # 中心剪裁
    transforms.RandomHorizontalFlip(),  # 依概率p水平翻转
    transforms.ToTensor(),              # 转tensor 并归一化
    transforms.Normalize(mean=[0.5, 0.5, 0.5], # 标准化
                         std=[0.5, 0.5, 0.5])
])


