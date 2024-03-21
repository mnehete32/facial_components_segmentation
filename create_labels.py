import os
import cv2
import glob
import numpy as np
import argparse



parser = argparse.ArgumentParser(description='CelebAMask-HQ')

parser.add_argument("--img-dir",type = str,required=True,
                    help = "input root directory of Images")

parser.add_argument("--save-label-dir",type = str,required=True,
                    help = "input directory to save label")


args = parser.parse_args()

def make_folder(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path)) 
        
        
#list1
#label_list = ['skin', 'neck', 'hat', 'eye_g', 'hair', 'ear_r', 'neck_l', 'cloth', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'nose', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip']
#list2 
label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l','neck', 'cloth']

folder_base = args.img_dir              #'./CelebAMask-HQ/CelebAMask-HQ-mask-anno'
folder_save = args.save_label_dir       #'./CelebAMask-HQ/CelebAMaskHQ-mask'
img_num = 30000

make_folder(folder_save)

for k in range(img_num):
    folder_num = k // 2000
    im_base = np.zeros((512, 512))
    for idx, label in enumerate(label_list):
        filename = os.path.join(folder_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
        if (os.path.exists(filename)):
            im = cv2.imread(filename)
            im = im[:, :, 0]
            
            im_base[im != 0] = (idx + 1)

    filename_save = os.path.join(folder_save, str(k) + '.png')
#     print (filename_save)
    cv2.imwrite(filename_save, im_base)
    if k % 1000 == 0 and k >0 :
        print(k)