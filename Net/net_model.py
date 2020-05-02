import torch.nn as nn


# nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
def make_conv():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, 1, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, 1, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, 1, 1),
        nn.ReLU(),
        # nn.Dropout(0.5),
        nn.MaxPool2d(2)
    )


def make_fc():
    return nn.Sequential(
        nn.Linear(64 * 4 * 4, 128),
        nn.ReLU(),
        # nn.Dropout(0.5),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(32,2)
    )


# 脸部特征识别网络
class face_attr(nn.Module):
    def __init__(self):
        super(face_attr, self).__init__()
        # attr0
        self.attr0_layer1 = make_conv()
        self.attr0_layer2 = make_fc()
        # attr1
        self.attr1_layer1 = make_conv()
        self.attr1_layer2 = make_fc()
        # attr2
        self.attr2_layer1 = make_conv()
        self.attr2_layer2 = make_fc()
        # attr3
        self.attr3_layer1 = make_conv()
        self.attr3_layer2 = make_fc()
        # attr4
        self.attr4_layer1 = make_conv()
        self.attr4_layer2 = make_fc()
        # attr5
        self.attr5_layer1 = make_conv()
        self.attr5_layer2 = make_fc()
        # attr6
        self.attr6_layer1 = make_conv()
        self.attr6_layer2 = make_fc()
        # attr7
        self.attr7_layer1 = make_conv()
        self.attr7_layer2 = make_fc()
        # attr8
        self.attr8_layer1 = make_conv()
        self.attr8_layer2 = make_fc()
        # attr9
        self.attr9_layer1 = make_conv()
        self.attr9_layer2 = make_fc()
        # attr10
        self.attr10_layer1 = make_conv()
        self.attr10_layer2 = make_fc()
        # attr11
        self.attr11_layer1 = make_conv()
        self.attr11_layer2 = make_fc()
        # attr12
        self.attr12_layer1 = make_conv()
        self.attr12_layer2 = make_fc()
        # attr13
        self.attr13_layer1 = make_conv()
        self.attr13_layer2 = make_fc()
        # attr14
        self.attr14_layer1 = make_conv()
        self.attr14_layer2 = make_fc()
        # attr15
        self.attr15_layer1 = make_conv()
        self.attr15_layer2 = make_fc()
        # attr16
        self.attr16_layer1 = make_conv()
        self.attr16_layer2 = make_fc()
        # attr17
        self.attr17_layer1 = make_conv()
        self.attr17_layer2 = make_fc()
        # attr18
        self.attr18_layer1 = make_conv()
        self.attr18_layer2 = make_fc()
        # attr19
        self.attr19_layer1 = make_conv()
        self.attr19_layer2 = make_fc()
        # attr20
        self.attr20_layer1 = make_conv()
        self.attr20_layer2 = make_fc()
        # attr21
        self.attr21_layer1 = make_conv()
        self.attr21_layer2 = make_fc()
        # attr22
        self.attr22_layer1 = make_conv()
        self.attr22_layer2 = make_fc()
        # attr23
        self.attr23_layer1 = make_conv()
        self.attr23_layer2 = make_fc()
        # attr24
        self.attr24_layer1 = make_conv()
        self.attr24_layer2 = make_fc()
        # attr25
        self.attr25_layer1 = make_conv()
        self.attr25_layer2 = make_fc()
        # attr26
        self.attr26_layer1 = make_conv()
        self.attr26_layer2 = make_fc()
        # attr27
        self.attr27_layer1 = make_conv()
        self.attr27_layer2 = make_fc()
        # attr28
        self.attr28_layer1 = make_conv()
        self.attr28_layer2 = make_fc()
        # attr29
        self.attr29_layer1 = make_conv()
        self.attr29_layer2 = make_fc()
        # attr30
        self.attr30_layer1 = make_conv()
        self.attr30_layer2 = make_fc()
        # attr31
        self.attr31_layer1 = make_conv()
        self.attr31_layer2 = make_fc()
        # attr32
        self.attr32_layer1 = make_conv()
        self.attr32_layer2 = make_fc()
        # attr33
        self.attr33_layer1 = make_conv()
        self.attr33_layer2 = make_fc()
        # attr34
        self.attr34_layer1 = make_conv()
        self.attr34_layer2 = make_fc()
        # attr35
        self.attr35_layer1 = make_conv()
        self.attr35_layer2 = make_fc()
        # attr36
        self.attr36_layer1 = make_conv()
        self.attr36_layer2 = make_fc()
        # attr37
        self.attr37_layer1 = make_conv()
        self.attr37_layer2 = make_fc()
        # attr38
        self.attr38_layer1 = make_conv()
        self.attr38_layer2 = make_fc()
        # attr39
        self.attr39_layer1 = make_conv()
        self.attr39_layer2 = make_fc()

    def forward(self, x):
        out_list = []
        # out0
        out0 = self.attr0_layer1(x)
        out0 = out0.view(out0.size(0), -1)
        out0 = self.attr0_layer2(out0)
        out_list.append(out0)
        # out1
        out1 = self.attr1_layer1(x)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.attr1_layer2(out1)
        out_list.append(out1)
        # out2
        out2 = self.attr2_layer1(x)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.attr2_layer2(out2)
        out_list.append(out2)
        # out3
        out3 = self.attr3_layer1(x)
        out3 = out3.view(out3.size(0), -1)
        out3 = self.attr3_layer2(out3)
        out_list.append(out3)
        # out4
        out4 = self.attr4_layer1(x)
        out4 = out4.view(out4.size(0), -1)
        out4 = self.attr4_layer2(out4)
        out_list.append(out4)
        # out5
        out5 = self.attr5_layer1(x)
        out5 = out5.view(out5.size(0), -1)
        out5 = self.attr5_layer2(out5)
        out_list.append(out5)
        # out6
        out6 = self.attr6_layer1(x)
        out6 = out6.view(out6.size(0), -1)
        out6 = self.attr6_layer2(out6)
        out_list.append(out6)
        # out7
        out7 = self.attr7_layer1(x)
        out7 = out7.view(out7.size(0), -1)
        out7 = self.attr7_layer2(out7)
        out_list.append(out7)
        # out8
        out8 = self.attr8_layer1(x)
        out8 = out8.view(out8.size(0), -1)
        out8 = self.attr8_layer2(out8)
        out_list.append(out8)
        # out9
        out9 = self.attr9_layer1(x)
        out9 = out9.view(out9.size(0), -1)
        out9 = self.attr9_layer2(out9)
        out_list.append(out9)
        # out10
        out10 = self.attr10_layer1(x)
        out10 = out10.view(out10.size(0), -1)
        out10 = self.attr10_layer2(out10)
        out_list.append(out10)
        # out11
        out11 = self.attr11_layer1(x)
        out11 = out11.view(out11.size(0), -1)
        out11 = self.attr11_layer2(out11)
        out_list.append(out11)
        # out12
        out12 = self.attr12_layer1(x)
        out12 = out12.view(out12.size(0), -1)
        out12 = self.attr12_layer2(out12)
        out_list.append(out12)
        # out13
        out13 = self.attr13_layer1(x)
        out13 = out13.view(out13.size(0), -1)
        out13 = self.attr13_layer2(out13)
        out_list.append(out13)
        # out14
        out14 = self.attr14_layer1(x)
        out14 = out14.view(out14.size(0), -1)
        out14 = self.attr14_layer2(out14)
        out_list.append(out14)
        # out15
        out15 = self.attr15_layer1(x)
        out15 = out15.view(out15.size(0), -1)
        out15 = self.attr15_layer2(out15)
        out_list.append(out15)
        # out16
        out16 = self.attr16_layer1(x)
        out16 = out16.view(out16.size(0), -1)
        out16 = self.attr16_layer2(out16)
        out_list.append(out16)
        # out17
        out17 = self.attr17_layer1(x)
        out17 = out17.view(out17.size(0), -1)
        out17 = self.attr17_layer2(out17)
        out_list.append(out17)
        # out18
        out18 = self.attr18_layer1(x)
        out18 = out18.view(out18.size(0), -1)
        out18 = self.attr18_layer2(out18)
        out_list.append(out18)
        # out19
        out19 = self.attr19_layer1(x)
        out19 = out19.view(out19.size(0), -1)
        out19 = self.attr19_layer2(out19)
        out_list.append(out19)
        # out20
        out20 = self.attr20_layer1(x)
        out20 = out20.view(out20.size(0), -1)
        out20 = self.attr20_layer2(out20)
        out_list.append(out20)
        # out21
        out21 = self.attr21_layer1(x)
        out21 = out21.view(out21.size(0), -1)
        out21 = self.attr21_layer2(out21)
        out_list.append(out21)
        # out22
        out22 = self.attr22_layer1(x)
        out22 = out22.view(out22.size(0), -1)
        out22 = self.attr22_layer2(out22)
        out_list.append(out22)
        # out23
        out23 = self.attr23_layer1(x)
        out23 = out23.view(out23.size(0), -1)
        out23 = self.attr23_layer2(out23)
        out_list.append(out23)
        # out24
        out24 = self.attr24_layer1(x)
        out24 = out24.view(out24.size(0), -1)
        out24 = self.attr24_layer2(out24)
        out_list.append(out24)
        # out25
        out25 = self.attr25_layer1(x)
        out25 = out25.view(out25.size(0), -1)
        out25 = self.attr25_layer2(out25)
        out_list.append(out25)
        # out26
        out26 = self.attr26_layer1(x)
        out26 = out26.view(out26.size(0), -1)
        out26 = self.attr26_layer2(out26)
        out_list.append(out26)
        # out27
        out27 = self.attr27_layer1(x)
        out27 = out27.view(out27.size(0), -1)
        out27 = self.attr27_layer2(out27)
        out_list.append(out27)
        # out28
        out28 = self.attr28_layer1(x)
        out28 = out28.view(out28.size(0), -1)
        out28 = self.attr28_layer2(out28)
        out_list.append(out28)
        # out29
        out29 = self.attr29_layer1(x)
        out29 = out29.view(out29.size(0), -1)
        out29 = self.attr29_layer2(out29)
        out_list.append(out29)
        # out30
        out30 = self.attr30_layer1(x)
        out30 = out30.view(out30.size(0), -1)
        out30 = self.attr30_layer2(out30)
        out_list.append(out30)
        # out31
        out31 = self.attr31_layer1(x)
        out31 = out31.view(out31.size(0), -1)
        out31 = self.attr31_layer2(out31)
        out_list.append(out31)
        # out32
        out32 = self.attr32_layer1(x)
        out32 = out32.view(out32.size(0), -1)
        out32 = self.attr32_layer2(out32)
        out_list.append(out32)
        # out33
        out33 = self.attr33_layer1(x)
        out33 = out33.view(out33.size(0), -1)
        out33 = self.attr33_layer2(out33)
        out_list.append(out33)
        # out34
        out34 = self.attr34_layer1(x)
        out34 = out34.view(out34.size(0), -1)
        out34 = self.attr34_layer2(out34)
        out_list.append(out34)
        # out35
        out35 = self.attr35_layer1(x)
        out35 = out35.view(out35.size(0), -1)
        out35 = self.attr35_layer2(out35)
        out_list.append(out35)
        # out36
        out36 = self.attr36_layer1(x)
        out36 = out36.view(out36.size(0), -1)
        out36 = self.attr36_layer2(out36)
        out_list.append(out36)
        # out37
        out37 = self.attr37_layer1(x)
        out37 = out37.view(out37.size(0), -1)
        out37 = self.attr37_layer2(out37)
        out_list.append(out37)
        # out38
        out38 = self.attr38_layer1(x)
        out38 = out38.view(out38.size(0), -1)
        out38 = self.attr38_layer2(out38)
        out_list.append(out38)
        # out39
        out39 = self.attr39_layer1(x)
        out39 = out39.view(out39.size(0), -1)
        out39 = self.attr39_layer2(out39)
        out_list.append(out39)

        return out_list


# 五官位置识别网络
class features_pos(nn.Module):
    def __init__(self):
        super(features_pos,self).__init__()

