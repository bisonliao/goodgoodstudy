# -*- coding: UTF-8 -*-

from torchvision import models
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import numpy
import torch.nn as nn
import torch.utils.data.dataset as dataset
import torch.utils.data.dataloader as dataloader
import os


batchsz=16
device ='cuda:0'
compute=True
#lr = 0.0001 # for ep 0-31
lr =0.00005
epochnm=100
start_ep=45
pretrained='./MyYOLOv1_45.tar'
classes = 2

def show_img(img:torch.Tensor):
    img_toshow = img  # type:torch.Tensor
    img_toshow = img_toshow.numpy()  # type:numpy.ndarray
    img_toshow = numpy.uint8(img_toshow * 255)
    img_toshow = img_toshow.transpose(1, 2, 0)
    plt.imshow(img_toshow)
    plt.show()

class YOLOv1(nn.Module):
    '''
    input size: 3X224X224
    denseNet feature output size: 1024X7X7
    detector output size: O X 7 X 7, O=(BX5+C)
    '''
    def __init__(self, num_classes=2, pretrained_backbone_weights=''):
        super(YOLOv1, self).__init__()
        self.CELL = 7

        self.C = num_classes
        self.device = device
        self.B = 2 # bbox num per cell

        if len(pretrained_backbone_weights) > 0:
            self.backbone = models.densenet121(num_classes=10)
            mdict, tdict = torch.load(pretrained_backbone_weights)
            self.backbone.load_state_dict(mdict)
        else:
            self.backbone = models.densenet121(pretrained=True)

        self.detector = nn.Sequential(
            # version#1
            #nn.Conv2d(1024, (5 * self.B + self.C), 1),
            #nn.Sigmoid()

            #version#2
            nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.CELL*self.CELL*(5 * self.B + self.C)),
            nn.Sigmoid()

        )



    def forward(self, x):
        if x.shape[1] != 3 or x.shape[2] !=224 or x.shape[3] != 224:
            raise Exception('input size invalid!')
        features = self.backbone.features(x)
        out = self.detector(features)
        newshape=(out.shape[0], -1, self.CELL, self.CELL)
        return out.reshape(newshape)

class myDataset(dataset.Dataset):
    def __init__(self, imagePathList, isTrain=True, transform=None, num_classes=2):
        super(myDataset, self).__init__()
        self.isTrain = isTrain
        self.CELL = 7
        self.num_classes=num_classes
        fh = open(imagePathList, "r")
        self.images = fh.readlines()
        self.transform = transform
        fh.close()

    def label2tensor(self, labelPath):
        fh = open(labelPath, "r")
        labelList = fh.readlines()
        fh.close()


        featureSz = 5 + self.num_classes
        retTensor = torch.zeros((featureSz, self.CELL, self.CELL))
        for line in labelList:
            line = line.replace("\r", "").replace("\n", "")
            c, x, y, w, h = [float(n) for n in line.split(" ")] #五个字段：类 中心点x 中心点y 宽度w 高度h， 后面四个都是相对于图片宽高的比，取值0-1
            if c >= self.num_classes:
                raise Exception("invalid class index:%d"%(c))
            # find the cell where bbox center locates, cell[x_index, y_index] is what we want
            x_index = 0
            y_index = 0
            for i in range(self.CELL-1, -1, -1):
                if x > (i/self.CELL):
                    x_index = i
                    x = (x - x_index/self.CELL) / (1/self.CELL)# x is the bias to cell topleft, [0,1]
                    assert(x>=0 and x<=1)
                    break
            for i in range(self.CELL-1, -1, -1):
                if y > (i/self.CELL):
                    y_index = i
                    y = (y - y_index/self.CELL) / (1/self.CELL) # y is the bias to cell topleft, [0,1]
                    assert y>=0 and y<=1
                    break
            if retTensor[0][x_index][y_index] == 1: # there is already a label
                continue;

            #标注就按这样的方式来组织，和图片像素的NCHW组织没有关系
            retTensor[0][x_index][y_index] = 1.0 # confidence
            retTensor[1][x_index][y_index] = x
            retTensor[2][x_index][y_index] = y
            retTensor[3][x_index][y_index] = w
            retTensor[4][x_index][y_index] = h
            retTensor[int(5+c)][x_index][y_index] = 1 # classification label

        return retTensor


    def __getitem__(self, index):
        imagePath = self.images[index]
        imagePath = imagePath.replace("\r", "").replace("\n", "")

        img = Image.open(imagePath)  # use pillow to open a file
        img = img.resize((224, 224))  # resize the file to 256x256
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        img = numpy.asarray(img)#type:numpy.ndarray
        img = img.transpose(2, 0, 1) # we have to change the dimensions from width x height x channel (WHC) to channel x width x height (CWH)
        img = torch.from_numpy(img).type(torch.float32)/255.0  #type:torch.Tensor

        #对应的label文件路径稍有不同，是文本文件，每行一个bbox标注，以此用空格隔开的有五个字段：类 中心点x 中心点y 宽度w 高度h， 后面四个都是相对于图片宽高的比，取值0-1
        labelPath = imagePath.replace("images", "labels").replace(".jpg", ".txt")
        label = self.label2tensor(labelPath)

        return img, label


    def __len__(self):
        return len(self.images)

def calc_iou(a:torch.Tensor, b:torch.Tensor, x_cellindex, y_cellindex, cellnr=7):
    _, ox1, oy1, w1, h1 = a.detach().cpu().numpy()
    _, ox2, oy2, w2, h2 = b.detach().cpu().numpy()

    #相对位置转绝对位置
    x1 = x_cellindex/cellnr + ox1/cellnr
    x2 = x_cellindex / cellnr + ox2 / cellnr
    y1 = y_cellindex/cellnr+oy1/cellnr
    y2 = y_cellindex / cellnr + oy2 / cellnr

    #交集区域的上下左右边界
    left1, right1, top1, bottom1 = (
        x1-w1/2,
        x1+w1/2,
        y1-h1/2,
        y1+h1/2
    )

    left2, right2, top2, bottom2 = (
        x2 - w2 / 2,
        x2 + w2 / 2,
        y2 - h2 / 2,
        y2 + h2 / 2
    )
    #assert left1>=0 and left1 <= 1 and right1 >=0 and right1<=1 and top1>=0 and top1<=1 and bottom1>=0 and bottom1<=1
    assert left2 >= -0.01 and left2 <= 1.01 and right2 >= -0.01 and right2 <= 1.01 and top2 >= -0.01 and top2 <= 1.01 and bottom2 >= -0.01 and bottom2 <= 1.01

    inter_left = max(left1, left2)
    inter_right = min(right1, right2)
    inter_top = min(top1, top2)
    inter_bottom = max(bottom1, bottom2)

    #交集的面积
    if inter_left>inter_right or inter_bottom < inter_top:
        intersection = 0
    else:
        intersection = (inter_right-inter_left)*(inter_bottom-inter_top)
    #并集的面积
    union = w1*h1+w2*h2-intersection

    return intersection/union


def MyLoss(predict:torch.Tensor, target:torch.Tensor, cellnr=7, B=2, C=classes):
    loss = 0
    for batch_index in range(target.shape[0]):
        for x_cellindex in range(cellnr):
            for y_cellindex in range(cellnr):
                if target[batch_index][0][x_cellindex][y_cellindex] == 1: # there is an object
                    response_box_index = 0
                    max_iou = 0
                    for box_index in range(B):
                        start = 0+5*box_index
                        end = 5+5*box_index
                        iou = calc_iou(predict[batch_index, start:end,  x_cellindex, y_cellindex],
                                         target[batch_index, 0:5,  x_cellindex, y_cellindex],
                                        x_cellindex, y_cellindex )
                        if iou > max_iou:
                            max_iou = iou
                            response_box_index = box_index

                    loss_x = predict[batch_index, response_box_index*5+1,  x_cellindex, y_cellindex]-target[batch_index, 1,  x_cellindex, y_cellindex]
                    loss_x = loss_x * loss_x

                    loss_y = predict[batch_index, response_box_index * 5 + 2, x_cellindex, y_cellindex] - target[batch_index, 2, x_cellindex, y_cellindex]
                    loss_y = loss_y * loss_y

                    loss_w = torch.sqrt(predict[batch_index, response_box_index*5+3,  x_cellindex, y_cellindex]) - torch.sqrt(target[batch_index, 3,  x_cellindex, y_cellindex])
                    loss_w = loss_w * loss_w

                    loss_h = torch.sqrt(
                        predict[batch_index, response_box_index * 5 + 4, x_cellindex, y_cellindex]) - \
                             torch.sqrt(target[batch_index, 4, x_cellindex, y_cellindex])
                    loss_h = loss_h * loss_h

                    loss_conf = predict[batch_index, response_box_index * 5 + 0, x_cellindex, y_cellindex] - target[
                        batch_index, 0, x_cellindex, y_cellindex]
                    loss_conf = loss_conf * loss_conf

                    loss_class = torch.dist(predict[batch_index, 5*B:5*B+C, x_cellindex, y_cellindex], target[
                        batch_index, 5:5+C, x_cellindex, y_cellindex])

                    loss += loss_x
                    loss += loss_y
                    loss += loss_w
                    loss += loss_h
                    loss += loss_conf
                    loss += loss_class

                    #没有命中的bbox，是confidence的负样本
                    for box_index in range(B):
                        if box_index == response_box_index:
                            continue
                        loss_conf = predict[batch_index, 0+box_index*5, x_cellindex, y_cellindex] - target[
                            batch_index, 0, x_cellindex, y_cellindex]
                        loss_conf = loss_conf * loss_conf
                        loss += loss_conf
                else:
                    # 没有命中的cell，是confidence的负样本
                    for box_index in range(B):
                        loss_conf = predict[batch_index, 0+box_index*5, x_cellindex, y_cellindex] - target[
                            batch_index, 0, x_cellindex, y_cellindex]
                        loss_conf = loss_conf * loss_conf
                        loss += loss_conf



    return loss




def train():
    set1 = myDataset("E:\\DeepLearning\\PyTorch-YOLOv3-master\\data\\bison\\train.txt", num_classes=classes)
    train_data = dataloader.DataLoader(set1, batchsz, False)# type:dataloader.DataLoader
    #model = YOLOv1(num_classes=classes, pretrained_backbone_weights="E:\\DeepLearning\\mxnet\\pyproject\\untitled\\denseNet_cifa_10_saved.torch.load").to(device)#type:YOLOv1
    model = YOLOv1(num_classes=classes).to(device)#type:YOLOv1
    trainer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)  # type:torch.optim
    if len(pretrained) > 0:  # load from a pretrained file
        mdict, tdict = torch.load(pretrained)
        model.load_state_dict(mdict)
        # trainer.load_state_dict(tdict)

    print("start training...")
    minbatch = 0
    loss_sum = 0
    for e in range(start_ep, epochnm):
        model.train()
        for imgs, labels in train_data:
            imgs = imgs.to(device)
            labels = labels.to(device)
            trainer.zero_grad()
            y = model(imgs)

            L = MyLoss(y, labels)


            L.backward()
            trainer.step()

            loss_sum += L.to("cpu").data.numpy()
            minbatch += 1
            if (minbatch % 20) == 0:
                print(e, "loss:", loss_sum)
                loss_sum = 0
        torch.save((model.state_dict(), trainer.state_dict()), "./MyYOLOv1_%d.tar" % (e))

def detect(pretrained_model:str, samples_path:str, cellnr=7, B=2, C=classes):
    model = YOLOv1(num_classes=classes)  # type:YOLOv1
    # load from a pretrained file
    mdict, tdict = torch.load(pretrained)
    model.load_state_dict(mdict)
    model.eval()
    for file in os.listdir(samples_path):
        filename = os.path.join(samples_path, file)
        if not os.path.isfile(filename) or filename.find(".jpg")<0:
            continue

        fh = Image.open(filename)  # use pillow to open a file
        fh = fh.resize((224, 224))  # resize the file to 256x256
        fh = fh.convert('RGB')
        img = numpy.asarray(fh)  # type:numpy.ndarray
        fh.close()
        img = img.transpose(2, 0, 1)  # we have to change the dimensions from width x height x channel (WHC) to channel x width x height (CWH)
        img = torch.from_numpy(img).type(torch.float32) / 255.0  # type:torch.Tensor

        input = torch.zeros((1, 3, 224,224))
        input[0] = img
        out = model(input) #type:torch.Tensor

        fh = Image.open(filename) #type:Image.Image
        draw = ImageDraw.Draw(fh)
        for x_cellindex in range(cellnr):
            for y_cellindex in range(cellnr):
                response_box_index = 0
                max_conf = 0
                for box_index in range(B):
                    conf = out[0, 0 + 5 * box_index, x_cellindex, y_cellindex].cpu().item()
                    if conf > max_conf:
                        max_conf = conf
                        response_box_index = box_index
                conf = out[0, 0 + 5 * response_box_index, x_cellindex, y_cellindex].cpu().item()
                if conf < 0.6:
                    continue

                conf, ox, oy, w, h = out[0, 5 * response_box_index:5+5 * response_box_index, x_cellindex, y_cellindex].detach().cpu().numpy()


                # 中心点相对位置转绝对位置
                x = x_cellindex / cellnr + ox / cellnr
                y = y_cellindex / cellnr + oy / cellnr

                classification = out[0,5*B:5*B+C, x_cellindex, y_cellindex].detach().cpu()

                cls_conf, cls_idx = torch.max( classification, 0 )
                cls_conf = cls_conf.item()
                cls_idx = cls_idx.item()

                # rect上下左右边界
                left, right, top, bottom = (
                    x - w / 2,
                    x + w / 2,
                    y - h / 2,
                    y + h / 2
                )
                left = int(left* fh.width)
                right = int(right* fh.width)
                top = int(top*fh.height)
                bottom = int(bottom * fh.height)
                draw.rectangle(((left,top),(right,bottom)))
                draw.text((left,top), "%d"%(cls_idx) )
        fh.save(filename.replace(".jpg", ".png"))


#train()
detect(pretrained, "E:\\DeepLearning\\PyTorch-YOLOv3-master\\data\\samples")