import torch
import torch.nn as nn


######################### 1. Functions for model definition #########################
def block1(in_channels, out_channels, stride=1):  # group1,2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


def block2(in_channels, out_channels, stride=1):  # group 3,4,5
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


def block3(input_size):
    return nn.Sequential(
        nn.Linear(input_size, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
    )


def predConv(input_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.Conv2d(input_channels, 33, kernel_size=3, stride=1, padding=1),
    )


def DeConv(k_scale, s_scale, p_scale):
    return nn.Sequential(
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(
            33, 33, kernel_size=k_scale, stride=s_scale, padding=p_scale
        ),
    )


def shift_image(image, width):
    shifted_image = torch.roll(image, shifts=width, dims=2)
    if width > 0:
        shifted_image[:, :, :width, :] = 0
    else:
        shifted_image[:, :, width:, :] = 0
    return shifted_image


def initialize_deconv_weights(S, K):
    C = (2 * S - 1 - (S % 2)) / (2 * S)

    weights = torch.zeros((K, K))
    for i in range(K):
        for j in range(K):
            weights[i, j] = (1 - abs(i / S - C)) * (1 - abs(j / S - C))

    return weights


def cal_one_channel(left_shift, softmax, channel, width=384, height=160):
    right_shift = torch.zeros(1, 3, width, height).cuda()
    for i in range(left_shift.shape[2]):
        for j in range(left_shift.shape[3]):
            # print(i,j)
            right_shift[0, :, i, j] = left_shift[0, :, i, j] * softmax[0, channel, i, j]

    return right_shift


############################### 2. Model definition ###############################
# define a torch module
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # define the layers
        # group 1-2
        self.group1 = block1(3, 64)
        self.group2 = block1(64, 128)

        # group 3-5
        self.group3 = block2(128, 256)
        self.group4 = block2(256, 512)
        self.group5 = block2(512, 512)

        # group6-7
        self.group6 = block3(input_size=30720)
        self.group7 = block3(input_size=512)

        # output
        self.fc8 = nn.Linear(512, 33 * 12 * 5)

        # batchnorm +pred layers
        self.pred4 = predConv(512)
        self.pred3 = predConv(256)
        self.pred2 = predConv(128)
        self.pred1 = predConv(64)

        # DeConv layers
        s = 1
        self.deConv1 = DeConv(k_scale=s, s_scale=s, p_scale=0)
        self.deConv1.weights = initialize_deconv_weights(s, s)
        s *= 2  # s=2
        self.deConv2 = DeConv(k_scale=2 * s, s_scale=s, p_scale=int(s / 2))
        self.deConv2.weights = initialize_deconv_weights(s, 2 * s)
        s *= 2  # s=4
        self.deConv3 = DeConv(k_scale=2 * s, s_scale=s, p_scale=int(s / 2))
        self.deConv3.weights = initialize_deconv_weights(s, 2 * s)
        s *= 2  # s=8
        self.deConv4 = DeConv(k_scale=2 * s, s_scale=s, p_scale=int(s / 2))
        self.deConv4.weights = initialize_deconv_weights(s, 2 * s)
        s *= 2  # s=16
        self.deConv5 = DeConv(k_scale=2 * s, s_scale=s, p_scale=int(s / 2))
        self.deConv5.weights = initialize_deconv_weights(s, 2 * s)

        s = 2
        self.up = DeConv(k_scale=2 * s, s_scale=s, p_scale=int(s / 2))
        self.up.weights = initialize_deconv_weights(s, 2 * s)

        self.upAct = nn.ReLU(inplace=True)
        self.upConv = nn.Conv2d(33, 33, kernel_size=3, stride=1, padding=1)
        self.SoftMax = nn.Softmax(dim=1)

    def forward(self, left_frame):
        # define the forward pass
        left_frame = left_frame.cuda()
        pool1 = self.group1(left_frame)
        pool2 = self.group2(pool1)
        pool3 = self.group3(pool2)
        pool4 = self.group4(pool3)
        pool5 = self.group5(pool4)

        flatten = torch.flatten(pool5)

        drop6 = self.group6(flatten)
        drop7 = self.group7(drop6)

        fc8 = self.fc8(drop7)
        # print(fc8.shape)
        pred5 = torch.reshape(fc8, (-1, 33, 12, 5))

        pred4 = self.pred4(pool4)
        pred3 = self.pred3(pool3)
        pred2 = self.pred2(pool2)
        pred1 = self.pred1(pool1)

        pred1 = self.deConv1(pred1)
        pred2 = self.deConv2(pred2)
        pred3 = self.deConv3(pred3)
        pred4 = self.deConv4(pred4)
        pred5 = self.deConv5(pred5)
        # print(pred1.shape)
        # print(pred2.shape)
        # element-wise sum
        feat = torch.add(pred1, pred2)
        # print(feat.shape)
        feat = torch.add(feat, pred3)
        # print(feat.shape)
        feat = torch.add(feat, pred4)
        # print(feat.shape)
        # print(pred5.shape)
        feat = torch.add(feat, pred5)

        up = self.up(feat)
        up = self.upAct(up)
        up = self.upConv(up)
        # print(up.shape)

        softmax = self.SoftMax(up)
        # print(softmax.shape)

        # #implement selection layer
        result = torch.zeros(1, 3, 384, 160).cuda()
        # for channel in tqdm(range(softmax.shape[1]),total = softmax.shape[1]):
        for channel in range(softmax.shape[1]):
            left_shift = shift_image(left_frame, channel - 16)
            # right_shift = cal_one_channel(left_shift,softmax, channel)
            right_shift = left_shift * softmax[:, channel, :, :]

            result = torch.add(result, right_shift)
            # print("Channel : ",channel)

        # return res,softmax
        return result, softmax


## 1980x1080 Model##############################################
class Net_1080(nn.Module):
    def __init__(self):
        super(Net_1080, self).__init__()
        # define the layers
        # group 1-2
        self.group1 = block1(3, 64)
        self.group2 = block1(64, 128)

        # group 3-5
        self.group3 = block2(128, 256)
        self.group4 = block2(256, 512)
        self.group5 = block2(512, 512)

        # group6-7
        self.group6 = block3(input_size=56320)
        # self.group7 = block3(input_size=512)

        # output
        self.fc8 = nn.Linear(512, 33 * 12 * 5)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=3)
        # batchnorm +pred layers
        self.pred4 = predConv(512)
        self.pred3 = predConv(256)
        self.pred2 = predConv(128)
        self.pred1 = predConv(64)

        self.zero_pad1 = nn.ZeroPad2d((0, 4, 0, 0, 0, 0, 0, 0))
        self.zero_pad2 = nn.ZeroPad2d((0, 460, 0, 288, 0, 0, 0, 0))

        # DeConv layers
        s = 1
        self.deConv1 = DeConv(k_scale=s, s_scale=s, p_scale=0)
        self.deConv1.weights = initialize_deconv_weights(s, s)
        s *= 2
        self.deConv2 = DeConv(k_scale=2 * s, s_scale=s, p_scale=int(s / 2))
        self.deConv2.weights = initialize_deconv_weights(s, 2 * s)
        s *= 2
        self.deConv3 = DeConv(k_scale=2 * s, s_scale=s, p_scale=int(s / 2))
        self.deConv3.weights = initialize_deconv_weights(s, 2 * s)
        s *= 2
        self.deConv4 = DeConv(k_scale=2 * s, s_scale=s, p_scale=int(s / 2))
        self.deConv4.weights = initialize_deconv_weights(s, 2 * s)
        s *= 2
        self.deConv5 = DeConv(k_scale=2 * s, s_scale=s, p_scale=int(s / 2))
        self.deConv5.weights = initialize_deconv_weights(s, 2 * s)

        s = 2
        self.up = DeConv(k_scale=2 * s, s_scale=s, p_scale=int(s / 2))
        self.up.weights = initialize_deconv_weights(s, 2 * s)

        self.upAct = nn.ReLU(inplace=True)
        self.upConv = nn.Conv2d(33, 33, kernel_size=3, stride=1, padding=1)
        self.SoftMax = nn.Softmax(dim=1)

    def forward(self, left_frame):
        # define the forward pass
        left_frame = left_frame.cuda()
        pool1 = self.group1(left_frame)
        pool2 = self.group2(pool1)
        pool3 = self.group3(pool2)
        pool4 = self.group4(pool3)
        pool5 = self.group5(pool4)
        pool5 = self.maxPool(pool5)
        # print("pool4 shape : ",pool5.shape)
        flatten = torch.flatten(pool5)

        drop6 = self.group6(flatten)
        # print("drop6 shape : ",drop6.shape)
        # drop7 = self.group7(drop6)
        # print("drop7 shape : ",drop7.shape)

        fc8 = self.fc8(drop6)
        # print("fc8 shape : ",fc8.shape)
        pred5 = torch.reshape(fc8, (-1, 33, 12, 5))

        pred4 = self.pred4(pool4)
        pred3 = self.pred3(pool3)
        pred2 = self.pred2(pool2)
        pred1 = self.pred1(pool1)

        pred1 = self.deConv1(pred1)
        pred2 = self.deConv2(pred2)
        pred3 = self.deConv3(pred3)
        pred4 = self.deConv4(pred4)
        pred4 = self.zero_pad1(pred4)
        # print(pred4.shape)
        pred5 = self.deConv5(pred5)
        # print(pred5.shape)
        pred5 = self.zero_pad2(pred5)
        # print(pred5.shape)
        # print(pred2.shape)
        # element-wise sum
        feat = torch.add(pred1, pred2)
        # print(feat.shape)
        feat = torch.add(feat, pred3)
        # print(feat.shape)
        feat = torch.add(feat, pred4)
        # print(feat.shape)
        # print(pred5.shape)
        feat = torch.add(feat, pred5)

        up = self.up(feat)
        up = self.upAct(up)
        up = self.upConv(up)
        # print(up.shape)

        softmax = self.SoftMax(up)
        # print(softmax.shape)
        # print(softmax.shape[1].dtype)

        # #implement selection layer
        result = torch.zeros(1, 3, 960, 1080).cuda()
        # for channel in tqdm(range(softmax.shape[1]),total = softmax.shape[1]):
        for channel in range(33):
            width = channel - 16
            left_shift = torch.roll(left_frame, shifts=width, dims=2)
            if width > 0:
                left_shift[:, :, :width, :] = 0
            else:
                left_shift[:, :, width:, :] = 0

            # left_shift = shift_image(left_frame, channel - 16)
            # right_shift = cal_one_channel(left_shift,softmax, channel)
            right_shift = left_shift * softmax[:, channel, :, :]

            result = torch.add(result, right_shift)
            # print("Channel : ",channel)

        # return res,softmax
        return result, softmax


############################### 3. Model Saving/Loading ###############################
