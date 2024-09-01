#conda activate readyVE
#python pngMasks_to_npyLabels.py
from PIL import Image
import numpy as np
 
path ="/home/mxochicale/Desktop/nystagmus-tracking/ready/data/mobious/sample-frames/test640x400"
#imagename="1_1i_Ll_1"
#imagename="1_1i_Ll_2"
#imagename="1_1i_Lr_1"
#imagename="1_1i_Lr_2"
imagename="1_1i_Ls_1"

img = np.asarray( Image.open(path+"/masks/"+imagename+".png").convert("L") ) #(400, 640)
#img = np.asarray( Image.open(path+"/masks/"+imagename+".png").convert("RGB") ) #(400, 640, 3)
print(type(img))
print(img.shape)

np.save(path+"/labels/"+imagename, img)
