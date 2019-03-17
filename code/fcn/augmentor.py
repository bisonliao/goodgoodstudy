import Augmentor
import glob
import os
import random
import shutil

img_type = 'jpg'

def start(train_path,groud_truth_path):
    train_img = glob.glob(train_path+'/*.'+img_type)
    masks = glob.glob(groud_truth_path+'/*.'+img_type)

    train_tmp_path = train_path+'\\tmp\\train'
    mask_tmp_path = groud_truth_path+'\\tmp\\mask'

    if len(train_img) != len(masks):
        print ("trains can't match masks",  len(train_img) , " ",  len(masks) )
        return 0
    for i in range(len(train_img)):
        train_img_tmp_path = train_tmp_path + '\\'+str(i)
        if not os.path.lexists(train_img_tmp_path):
            os.makedirs(train_img_tmp_path)
        shutil.copy(train_path+'/'+str(i)+'.'+img_type, train_img_tmp_path+'/'+str(i)+'.'+img_type)

        mask_img_tmp_path =mask_tmp_path +'/'+str(i)
        if not os.path.lexists(mask_img_tmp_path):
            os.makedirs(mask_img_tmp_path)
        shutil.copy(groud_truth_path+'/'+str(i)+'.'+img_type,  mask_img_tmp_path+'/'+str(i)+'.'+img_type)
        print ("%s folder has been created!"%str(i))
    return i+1


def doAugment(train_path,groud_truth_path, num):
    sum = 0
    train_tmp_path = train_path+'\\tmp\\train'
    mask_tmp_path = groud_truth_path+'\\tmp\\mask'
    for i in range(num):
        p = Augmentor.Pipeline(train_tmp_path+'/'+str(i))
        p.ground_truth(mask_tmp_path+'/'+str(i))
        p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
        p.flip_left_right(probability=0.5)
        p.zoom_random(probability=0.6, percentage_area=0.99)
        p.flip_top_bottom(probability=0.6)
        p.random_distortion(probability=0.8,grid_width=10,grid_height=10, magnitude=20)
        count = random.randint(40, 60)
        print("\nNo.%s data is being augmented and %s data will be created"%(i,count))
        sum = sum + count
        p.sample(count)
        print("Done")
    print("%s pairs of data has been created totally"%sum)


a = start("E:\\DeepLearning\\data\\test\\image", "E:\\DeepLearning\\data\\test\\groundtruth")
doAugment("E:\\DeepLearning\\data\\test\\image", "E:\\DeepLearning\\data\\test\\groundtruth", a)
