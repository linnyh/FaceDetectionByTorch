# -*- coding: utf-8 -*-
"""
Created on Sun Jun  11:54:36 2020
@author: Liam
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as Data

import Net.net_model as net
import Data_Loader.Dataset as data

img_root = './CelebA/Img/img_align_celeba'
train_txt = './train5000.txt'
test_txt = './test1000.txt'
batch_size = 2


module = net.face_attr()
# print(module)

optimizer = optim.Adam(module.parameters(), lr=0.001, weight_decay=1e-8)

def train(epoch):

    # 载入训练数
    train_dataset = data.myDataset(img_dir=img_root, img_txt=train_txt, transform=data.transform)
    train_dataloader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss_list = []
    for i in range(40): # 40 个神经网络需要40个损失函数单元
        loss_func = nn.CrossEntropyLoss()
        loss_list.append(loss_func)
    # loss_func = nn.CrossEntropyLoss()
    for Epoch in range(epoch):                                              # 开始训练
        all_correct_num = 0     #
        for ii, (img, label) in enumerate(train_dataloader):
            img = Variable(img)
            label = Variable(label)                                         # 将Tensor用Variable包装下，Variable支持Tensor的几乎一切操作
            #    optimizer.zero_grad()
            output = module(img)                                            # 前馈计算
            optimizer.zero_grad()
            for i in range(40):                                             # 训练 40 个网络
                loss = loss_list[i](output[i], label[:, i])
                loss.backward()                                             # 反向传播
                _, predict = torch.max(output[i], 1)                        # 按列取最大值
                correct_num = sum(predict == label[:, i])                   # 累加预测正确的样本（以一个batch为单位）
                all_correct_num += correct_num.data.item()                  # 单轮（Epoch）预测的正确样本数
            optimizer.step()                                                # 优化器
        Accuracy = all_correct_num * 1.0 / (len(train_dataset) * 40.0)      # 计算本轮（Epoch）正确率
        print('Epoch ={0},all_correct_num={1},Accuracy={2}'.format(Epoch, all_correct_num, Accuracy))
        torch.save(module, './models/face_attr40.pkl')                      # 保存整个模型


if __name__ == '__main__':
    Epoch = input("Enter the number of generations you want to train: ")
    train(10)