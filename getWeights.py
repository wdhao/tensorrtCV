# -*- coding: utf-8 -*-
"""
Created on Sat May 29 19:12:58 2021

@author: Administrator
"""
import xml.etree.ElementTree as ET
import os
from PIL import Image
import numpy as np
import torch

from PIL import Image
import torch
import torchvision
import struct




path = "H:/myGitHub/tensorrtF/model/yolov5/"
Path = os.path.join(path, "wts")
if not os.path.isdir(Path):
    os.makedirs(Path)
def getweights(model_path):
    state_dict = torch.load(model_path,map_location= lambda storage,loc :storage)
    print(state_dict )
    keys = [v for key,v in enumerate(state_dict)]
    print(keys)
    with open(os.path.join(Path,"network.txt"),'w') as fw:
        for key in keys:
            print("~~~~~~~~~~~ ",key)
            ts = state_dict[key]
            shape = ts.shape
            size = shape
            allsize = 1
            fw.write(key + " ")
            for idx in range(len(size)):
                allsize *= size[idx]
                fw.write(str(size[idx])+ " ")
            fw.write('\n')
            ts = ts.reshape(allsize)
            with open(Path + '/'+ key + '.wgt','wb') as f:
                a = struct.pack('i',allsize)
                f.write(a)
                for i in range(allsize):
                    a = struct.pack('f',ts[i])#.hex()
                    f.write(a)
                            
                            
                    
if __name__ == '__main__':

    model = torch.load(path+'yolov5s.pt')['model'].float() 
    torch.save(model.state_dict(),path+'yolov5s.pth')
    getweights(path + "yolov5s.pth")
    #model = torchvision.models.resnet50()
    #model.eval()
    #torch.save(model.state_dict(),r"H:\myGitHub\tensorrtF\model\resnet50\res50.pth")
    #a = torch.randn(1,3,256,256).type(torch.float32)
    #torch.onnx.export(model, a,r"H:\myGitHub\tensorrtF\model\resnet50\res50.onnx",training=2 )