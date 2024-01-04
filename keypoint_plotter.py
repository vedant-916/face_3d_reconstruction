import numpy as np
import os
from api import PRN
from utils.cv_plot import plot_kpt
import cv2
import ast
import argparse
from skimage.io import imread, imsave
def main(args):
 os.environ['CUDA_VISIBLE_DEVICES'] = "0"
 prn = PRN(args.isDlib)
 collection = "G:/FACE_VIDEO/combFOLDER/COMBINED_PADKEYPTS"
 for k, filename in enumerate(os.listdir(collection)):
   image_path = "G:/FACE_VIDEO/COMBINED_PAD/"+ filename[:-4]+ ".png"
   image = imread(image_path)
   file1 = open("G:/FACE_VIDEO/combFOLDER/COMBINED_PADKEYPTS/"+ filename,"r")
   kpts = np.zeros([3, 68], dtype=float)
   # print(filename)
   for i in range(68):
       t = file1.readline()
       t = t.strip()
       ind1 = t.index(" ")
       num1 = t[0:ind1]
       # print(num1)
       t = t[ind1 + 1:]
       ind2 = t.index(" ")
       num2 = t[0:ind2]
       # print(num2)
       t = t[ind2 + 1:]
       num3 = t.strip()
       # print(num3)
       kpts[0][i] = float(num1)
       kpts[1][i] = float(num2)
       kpts[2][i] = float(num3)
   pos = prn.process(image, kpts)
   kpt = prn.get_landmarks(pos)
   [h, w, c] = image.shape
   if c>3:
       image = image[:,:,:3]
   image2_path = "G:/FACE_VIDEO/OVERLAPPED_3D_RESIZED/"+ filename[:-4]+ ".png"
   image2 = imread(image2_path)
   #image2 = image/255.
   #print(kpt)
   cv2.imwrite('G:/FACE_VIDEO/3D_OVERLAPPED_KEYPOINTS/'+ filename[:-4]+ ".png", plot_kpt(image2, kpt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')
    parser.add_argument('--isDlib', default=False, type=ast.literal_eval)
    main(parser.parse_args())