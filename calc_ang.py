import numpy as np
import os
import cv2
import argparse
import ast
from api import PRN
from utils.estimate_pose import estimate_pose
from skimage.io import imread, imsave
#import logging
#def angl_calc(image, P, kpt, color=(0, 255, 0), line_width=2,mode=""):
#    image = image.copy()
#
#    point_3d = []
#    rear_size = 90
#    rear_depth = 0
#    point_3d.append((-rear_size, -rear_size, rear_depth))
#    point_3d.append((-rear_size, rear_size, rear_depth))
#    point_3d.append((rear_size, rear_size, rear_depth))
#    point_3d.append((rear_size, -rear_size, rear_depth))
#    point_3d.append((-rear_size, -rear_size, rear_depth))
#
#    front_size = 105
#    front_depth = 110
#    point_3d.append((-front_size, -front_size, front_depth))
#    point_3d.append((-front_size, front_size, front_depth))
#    point_3d.append((front_size, front_size, front_depth))
#    point_3d.append((front_size, -front_size, front_depth))
#    point_3d.append((-front_size, -front_size, front_depth))
#    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)
#
#    # Map to 2d image points
#    point_3d_homo = np.hstack((point_3d, np.ones([point_3d.shape[0], 1])))  # n x 4
#    point_2d = point_3d_homo.dot(P.T)[:, :2]
#    point_2d[:, :2] = point_2d[:, :2] - np.mean(point_2d[:4, :2], 0) + np.mean(kpt[:27, :2], 0)
#    point_2d = np.int32(point_2d.reshape(-1, 2))
#    # print(point_2d)
#    # print(point_2d[8])
#    # print(point_2d[3])
#    # print(point_2d[3][0])
#    a = np.array([point_2d[8][0], point_2d[8][1], 50])
#    b = np.array([point_2d[3][0], point_2d[3][1], 0])
#    c = np.array([point_2d[3][0], 0, 0])
#
#    ba = a - b
#    bc = c - b
#
#    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#    angle = np.arccos(cosine_angle)
#    ########### y -- axis
#    a2 = np.array([point_2d[7][0], point_2d[7][1], 50])
#    b2 = np.array([point_2d[2][0], point_2d[7][1], 0])
#    c2 = np.array([point_2d[2][0] + 10, point_2d[7][1], 0])
#
#    ba2 = a2 - b2
#    bc2 = c2 - b2
#
#    cosine_angle2 = np.dot(ba2, bc2) / (np.linalg.norm(ba2) * np.linalg.norm(bc2))
#    angle2 = np.arccos(cosine_angle2)
#    #print(np.degrees(angle2))
#    #print(angle2)
#    ################################ z ------- axis
#    a3 = np.array([point_2d[3][0], point_2d[3][1], 0])
#    b3 = np.array([point_2d[0][0], point_2d[0][1], 0])
#    c3 = np.array([point_2d[0][0] + 10, point_2d[0][1], 0])
#
#    ba3 = a3 - b3
#    bc3 = c3 - b3
#
#    cosine_angle3 = np.dot(ba3, bc3) / (np.linalg.norm(ba3) * np.linalg.norm(bc3))
#    angle3 = np.arccos(cosine_angle3)
#    #print(np.degrees(angle3))
#    #print(angle3)
#
#    if mode == "x":
#        return angle
#    elif mode == "y":
#       return angle2
#    else:
#        return angle3
#
#    #print(np.degrees(angle))
#    #print(angle)
#
#def main(args):
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# prn = PRN(args.isDlib)
# collection = "G:/FACE_VIDEO/nigger/keypts_fr_depthnet"
# for k, filename in enumerate(os.listdir(collection)):
#     collection2 = "G:/FACE_VIDEO/nigger/keypts_fr_depthnet/" + filename
#     for l2, filename2 in enumerate(os.listdir(collection2)):
#      image = imread("G:/FACE_VIDEO/nigger/cropED/151/frame36.png")
#      [h, w, c] = image.shape
#      if c>3:
#         image = image[:,:,:3]
#      file1 = open("G:/FACE_VIDEO/nigger/keypts/151/frame36.png.txt","r")
#      kpts = np.zeros([3, 68], dtype=float)
#      # print(filename)
#      for i in range(68):
#          t = file1.readline()
#          t = t.strip()
#          ind1 = t.index(" ")
#          num1 = t[0:ind1]
#          # print(num1)
#          t = t[ind1 + 1:]
#          ind2 = t.index(" ")
#          num2 = t[0:ind2]
#          # print(num2)
#          t = t[ind2 + 1:]
#          num3 = t.strip()
#          # print(num3)
#          kpts[0][i] = float(num1)
#          kpts[1][i] = float(num2)
#          kpts[2][i] = float(num3)
#      pos = prn.process(image, kpts)
#      image = image / 255.
#      vertices = prn.get_vertices(pos)
#      camera_matrix, pose = estimate_pose(vertices)
#      kpt = prn.get_landmarks(pos)
#      ang = angl_calc(image, camera_matrix, kpt,mode='x')
#      ang = ang
#      #file3 = open("G:/FACE_VIDEO/nigger/mp_diff/" + filename + "/" + filename2[:-9]  + ".txt","w+")
#      #print(filename)
#      print(ang)
#      ang2 = angl_calc(image, camera_matrix, kpt, mode='y')
#      print(ang2)
#      ang3= angl_calc(image, camera_matrix, kpt,mode='z')
#      print(ang3)
#      diff_x = 1.4917341347767357-ang
#      diff_y = 1.4317003853128254 - ang2
#      #file3.write(str(diff_x) +"\n" + str(diff_y) + "\n" + str(ang3))
#      #file3.close()
#      #return ang,ang2,ang3
#
#if __name__ == '__main__':
#    parser = argparse.ArgumentParser(description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')
#    parser.add_argument('--isDlib', default=False, type=ast.literal_eval)
#    main(parser.parse_args())
