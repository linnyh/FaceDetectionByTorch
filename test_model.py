import torch
import cv2
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as Data
import Data_Loader.Dataset as data

img_root = './CelebA/Img/img_align_celeba'
train_txt = './train1000.txt'
test_txt = './test1000.txt'
batch_size = 2

transform = transforms.Compose([
    transforms.Resize(40),              # 图像缩小
    transforms.CenterCrop(32),          # 中心剪裁
    transforms.RandomHorizontalFlip(),  # 依概率p水平翻转
    transforms.ToTensor(),              # 转tensor 并归一化
    transforms.Normalize(mean=[0.5, 0.5, 0.5], # 标准化
                         std=[0.5, 0.5, 0.5])
])

def test():
    test_dataset = data.myDataset(img_dir=img_root,img_txt = test_txt,transform= data.transform)
    test_dataloader = Data.DataLoader(test_dataset,batch_size = batch_size,shuffle=True)
    module = torch.load('./models/face_attr40.pkl')
    #module.load_state_dict(torch.load("./models/face_attr40.pkl"))
    module.eval()
    all_correct_num = 0
    for ii,(img,label) in enumerate(test_dataloader):
        img = Variable(img)
        label = Variable(label)
        output = module(img)
        for i in range(40):
            _,predict = torch.max(output[i],1)
            correct_num = sum(predict==label[:,i])
            all_correct_num += correct_num.data.item()
    Accuracy =  all_correct_num *1.0/(len(test_dataset)*40.0)
    print('all_correct_num={0},Accuracy={1}'.format(all_correct_num,Accuracy))

if __name__ == "__main__":
    test()