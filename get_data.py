#coding: utf-8#

from PIL import Image
import os
import numpy as np

filepath='哈士奇'
filelists=os.walk(filepath)
filenames=list(filelists)[0][2]
ha_images=None
for filename in filenames:
    try:
        print(filepath+'/'+filename)
        img=Image.open(filepath+'/'+filename)
        img=img.convert('L')
        img=img.resize((128,128))
        img_array=np.asarray(img,dtype=np.float32).reshape((1,128,128))
        try:
            ha_images=np.concatenate((ha_images,img_array))
        except:
            ha_images=img_array
    except:
        pass


filepath='泰迪'
filelists=os.walk(filepath)
filenames=list(filelists)[0][2]
tai_images=None
for filename in filenames:
    try:
        print(filepath + '/' + filename)
        img=Image.open(filepath+'/'+filename)
        img=img.convert('L')
        img=img.resize((128,128))
        img_array=np.asarray(img,dtype=np.float32).reshape((1,128,128))
        try:
            tai_images=np.concatenate((tai_images,img_array))
        except:
            tai_images=img_array
    except:
        pass

images=np.concatenate((ha_images,tai_images))
labels=np.concatenate((np.ones(len(ha_images)),np.ones(len(tai_images))+1))
