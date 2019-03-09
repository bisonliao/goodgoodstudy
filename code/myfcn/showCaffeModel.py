import caffe
import struct
import numpy as np
import sys
import os
 
np.set_printoptions(threshold='nan')
 
MODEL_FILE = sys.argv[2]
PRETRAIN_FILE = sys.argv[1]
 
params_txt = 'params.txt'
pf_txt = open(params_txt, 'w')
 
net = caffe.Net(MODEL_FILE, PRETRAIN_FILE, caffe.TEST)
 
for param_name in net.params.keys():
    weight = net.params[param_name][0].data
                     
    pf_dat = open('./weight/'+param_name+'.txt', 'w')
    len_w=len(weight.shape)
    if (len_w==4):##conv layer
        byte1=struct.pack('i',weight.shape[3])
        byte3=struct.pack('i',weight.shape[1])
        byte4=struct.pack('i',weight.shape[0])
    elif(len_w==2):##fc layer
        byte1=struct.pack('i',weight.shape[1])
        byte2=struct.pack('i',weight.shape[0])
         
    pf_txt.write(param_name)
    pf_txt.write('\n')
    pf_txt.write('shape:(')
    pf_dat.write('shape:(')
    for dim in weight.shape[:]:
        pf_txt.write(str(dim))
        pf_dat.write(str(dim))
        pf_txt.write(" ")
        pf_dat.write(" ")
    pf_txt.write(')\n')
    pf_dat.write(')\n')
 
    pf_dat.write('\n' + param_name + '_weight:\n\n')
    
    weight.shape = (-1, 1)
        
    for w in weight:
        pf_dat.write('%f, ' % w)
 
    if len(net.params[param_name]) < 2:
        pf_dat.write("\n\nlayer %s has NO bias!!"%param_name)
        pf_dat.close
        continue
    
    bias =  net.params[param_name][1].data
    pf_dat.write('\n\n' + param_name + '_bias:\n\n')
    bias.shape = (-1, 1)
    for b in bias:
        pf_dat.write('%f, ' % b)
    pf_dat.close
 
pf_txt.close
 
