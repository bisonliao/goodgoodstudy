# -*- coding: UTF-8 -*-

import json
import os
import shutil
import stat


only_one = True #只识别检测火车

json_data=open('E:\\DeepLearning\\data\\image_data\\coco\\annotations\\instances_train2014.json').read()
jsonobj = json.loads(json_data)

#卧槽，COCO数据集里的category的id居然不是连续的，导致labels文件里的categoryid超出80，pytorch yolo crash了，因为比classes names实际数量（80）大
cate_list=[]
for cate in jsonobj["categories"][:]:
    cate_list.append( (cate["id"], cate["name"]) )
cate_list = sorted(cate_list)

print(cate_list)

continual_idx = 0
category_dict={}
for cate in cate_list[:]:
    #print(cate)
    id, name = cate
    category_dict[id]=(continual_idx, name)
    print(id, continual_idx, name)
    continual_idx+=1




height_dict={}
width_dict={}
filename_dict={}
no_train_files={}
content_dict={}

for image in jsonobj["images"][:]:
    filename_dict[ image["id"] ] = image["file_name"]
    height_dict[ image["id"] ] = image["height"]
    width_dict[ image["id"] ] = image["width"]
    content_dict[ image["file_name"] ] = ""
    if only_one:
        no_train_files[ image["file_name"] ] = image["file_name"]



'''
The dataloader expects that the annotation file corresponding to the image data/custom/images/train.jpg has 
the path data/custom/labels/train.txt. Each row in the annotation file should define one bounding box, using 
the syntax label_idx x_center y_center width height. The coordinates should be scaled [0, 1], and the 
label_idx should be zero-indexed and correspond to the row number of the class name in data/custom/classes.names.
'''



cnt = 0
print("total annotation:", len(jsonobj["annotations"]))
for anno in jsonobj["annotations"][:]:
    [x,y,w,h]=anno["bbox"]
    image_id = anno["image_id"]

    continual_idx, _ = category_dict[anno["category_id"]]
    if continual_idx >= 80 or continual_idx < 0:
        print("invalid category continual id!!!")
        exit(-1)
    filename = filename_dict[image_id]#type:str

    if only_one:
        if continual_idx != 6 : #不是火车, 只识别一种物体看看
            continue

    if continual_idx == 6 and no_train_files.__contains__(filename):
        no_train_files.__delitem__(filename)


    center_x = (x + w / 2)
    center_x = center_x / width_dict[image_id]
    center_y =  y + h / 2
    center_y = center_y / height_dict[image_id]
    w = w / width_dict[image_id]
    h = h / height_dict[image_id]
    if only_one:
        content = content_dict[filename] + "%d %.4f %.4f %.4f %.4f\n" % (0, center_x, center_y, w, h)
    else:
        content = content_dict[ filename ] + "%d %.4f %.4f %.4f %.4f\n" % (continual_idx, center_x, center_y, w, h)
    content_dict[filename] = content
    cnt+=1
    if (cnt%10000) == 0:
        print("%d"%(cnt))



ft = open('E:\\DeepLearning\\data\\image_data\\coco\\train.txt', "w")
fv = open('E:\\DeepLearning\\data\\image_data\\coco\\valid.txt', "w")
cnt = 0
for id in filename_dict.keys():
    filename = filename_dict[id]
    if len(content_dict[ filename ]) < 3 : # no bbox in the image, jump
        continue
    cnt += 1

    if (cnt % 19) == 0:
        fv.write("E:/DeepLearning/data/image_data/coco/images/%s\n"%(filename))
    else:
        ft.write("E:/DeepLearning/data/image_data/coco/images/%s\n" % (filename))
cnt = 0
for k in no_train_files.keys():
    cnt +=1
    filename = k #type:str
    if (cnt%19)==0:
        fv.write("E:/DeepLearning/data/image_data/coco/images/%s\n" % (filename))
    else:
        ft.write("E:/DeepLearning/data/image_data/coco/images/%s\n" % (filename))
    if cnt > 1000:
        break

ft.close()
fv.close()


cnt=0
print("file number:", len(content_dict.keys()) )
for f in content_dict.keys():
    cnt += 1
    content = content_dict[f]
    if len(content) < 3:
        continue
    f = f.replace("jpg", "txt")
    path = 'E:\\DeepLearning\\data\\image_data\\coco\\labels\\'+f
    fo = open(path, "w")
    fo.write(content)
    fo.close()

    if (cnt % 10000) == 0:
        print("%d" % (cnt))


for k in no_train_files.keys():
    path = 'E:\\DeepLearning\\data\\image_data\\coco\\labels\\'+k.replace(".jpg", ".txt")
    fo = open(path, "w")
    fo.close()



