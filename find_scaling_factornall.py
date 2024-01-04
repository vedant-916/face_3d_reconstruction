import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast
from utils.cv_plot import plot_kpt, plot_vertices, plot_pose_box
import sys
from PRNet.utils.render import vis_of_vertices, render_texture
import cv2
from PRNet.api import PRN
np.set_printoptions(threshold=sys.maxsize)
from PRNet.utils.estimate_pose import estimate_pose2
from PRNet.utils.estimate_pose import P2sRt
from PRNet.utils.rotate_vertices import frontalize
from PRNet.utils.render_app import get_visibility, get_uv_mask, get_depth_image
from PRNet.utils.write import write_obj_with_colors, write_obj_with_texture
from scipy.io import loadmat,savemat
#prn = PRN(is_dlib=False)
#collection = "C:/Users/Games/Pictures/Video_Projects/2"
#for k, filename in enumerate(os.listdir(collection)):
#  image_path = "C:/Users/Games/Pictures/Video_Projects/2\\" + filename
#  image = imread(image_path)
#  [h, w, c] = image.shape
#  if c>3:
#      image = image[:,:,:3]
#
#  inter_index = image_path.index('\\')
#  file1 = open("C:/Users/Games/Pictures/Video_Projects/2_KEYS/" + image_path[inter_index + 1:-4] + ".txt", "r")
#  kpts = np.zeros([3, 68], dtype=float)
#
#  for i in range(68):
#      t = file1.readline()
#      t = t.strip()
#      # print(t)
#      ind1 = t.index(" ")
#      num1 = t[0:ind1]
#      # print(num1)
#      t = t[ind1 + 1:]
#      ind2 = t.index(" ")
#      num2 = t[0:ind2]
#      # print(num2)
#      t = t[ind2 + 1:]
#      num3 = t.strip()
#
#      kpts[0][i] = float(num1)
#      kpts[1][i] = float(num2)
#      kpts[2][i] = float(num3)
#
#  box = np.array([0, image.shape[1] - 1, 0, image.shape[0] - 1])  # cropped with bounding box
#  pos = prn.process(image, kpts)
#  IMG2 = imread("C:/Users/Games/PycharmProjects/PRNet/Data/uv-data/images/frame0.png")
#  IMG2 = IMG2 / 255.
#  vertices = prn.get_vertices(pos)
#
#  #vertices = vertices*1.3278158890
#  #camera_matrix, pose = estimate_pose(vertices)
#  #sca,R,tra = P2sRt(camera_matrix) # decompose affine matrix to s, R, t
#  #print(sca)
#  #cv2.imshow('desne',plot_vertices(IMG2, vertices))
#  #cv2.waitKey(0)
#  cv2.imwrite('C:/Users/Games/PycharmProjects/PRNet/Data/uv-data/images/New_folder2/' + filename, plot_vertices(IMG2, vertices))
#
#  image_path = "C:/Users/Games/Pictures/Video_Projects/1\\" + filename
#  image = imread(image_path)
#  [h, w, c] = image.shape
#  if c>3:
#      image = image[:,:,:3]
#
#  inter_index = image_path.index('\\')
#  file1 = open("C:/Users/Games/Pictures/Video_Projects/1_KEYS/" + image_path[inter_index + 1:-4] + ".txt", "r")
#  kpts = np.zeros([3, 68], dtype=float)
#
#  for i in range(68):
#      t = file1.readline()
#      t = t.strip()
#      # print(t)
#      ind1 = t.index(" ")
#      num1 = t[0:ind1]
#      # print(num1)
#      t = t[ind1 + 1:]
#      ind2 = t.index(" ")
#      num2 = t[0:ind2]
#      # print(num2)
#      t = t[ind2 + 1:]
#      num3 = t.strip()
#
#      kpts[0][i] = float(num1)
#      kpts[1][i] = float(num2)
#      kpts[2][i] = float(num3)
#
#  box = np.array([0, image.shape[1] - 1, 0, image.shape[0] - 1])  # cropped with bounding box
#  pos = prn.process(image, kpts)
#  IMG2 = imread("C:/Users/Games/PycharmProjects/PRNet/Data/uv-data/images/frame0.png")
#  IMG2 = IMG2 / 255.
#  vertices2 = prn.get_vertices(pos)
#  #cv2.imshow('desne2',plot_vertices(IMG2, vertices2))
#  #cv2.waitKey(0)
#  camera_matrix, pose = estimate_pose2(vertices,vertices2)
#  sca,R,tra = P2sRt(camera_matrix) # decompose affine matrix to s, R, t
#  print(tra)
#  #vertices2 = vertices2*sca
#  arr1 = []
#  arr2 =[]
#  file1 = open("C:/Users/Games/PycharmProjects/FaceMesh/sav_filt/N/" + filename[:-4]+ ".txt",'r')
#  file2 = open("C:/Users/Games/PycharmProjects/FaceMesh/sav_filt/N2/" + filename[:-4] + ".txt", 'r')
#  for u in range(468):
#      text1 = file1.readline()
#      text1 = text1.strip()
#      text1 = text1.split()
#      arr1.append([float(text1[0]), float(text1[1]),float(text1[2])])
#      text1 = file2.readline()
#      text1 = text1.strip()
#      text1 = text1.split()
#      arr2.append([float(text1[0]), float(text1[1]), float(text1[2])])
#
#  camera_matrix2, pose2 = estimate_pose2(arr2,arr1)
#  sca2, R2, tra2 = P2sRt(camera_matrix2)
#  #vertices2 = arr1
#  for i in range(len(vertices2)):
#    a = vertices2[i][0]*sca2
#    b = vertices2[i][1]*sca2
#    c = vertices2[i][2]*sca2
#    elem11 = R2[0][0]
#    elem12 = R2[0][1]
#    elem13 = R2[0][2]
#    elem21 = R2[1][0]
#    elem22 = R2[1][1]
#    elem23 = R2[1][2]
#    elem31 = R2[2][0]
#    elem32 = R2[2][1]
#    elem33 = R2[2][2]
#    newx = (elem11 * a) + (elem12 * b) + (elem13 * c)
#    newy = (elem21 * a) + (elem22 * b) + (elem23 * c)
#    newz = (elem31 * a) + (elem32 * b) + (elem33 * c)
#    vertices2[i][0] = newx
#    vertices2[i][1] = newy
#    vertices2[i][2] = newz
#
#  camera_matrix, pose = estimate_pose2(vertices, vertices2)
#  sca, R, tra = P2sRt(camera_matrix)  # decompose affine matrix to s, R, t
#  for i in range(len(vertices2)):
#      vertices2[i][0] = vertices2[i][0]  + tra[0]
#      vertices2[i][1] = vertices2[i][1] + tra[1]
#
#  cv2.imwrite('C:/Users/Games/PycharmProjects/PRNet/Data/uv-data/images/New_folder/' + filename,plot_vertices(IMG2, vertices2))
#import math
#file = open("C:/Users/Games/PycharmProjects/FaceMesh/sav_filt/ims/REF.txt",'w+')
#file1 = open("C:/Users/Games/PycharmProjects/FaceMesh/sav_filt/Z/frame71.txt",'r')
#mat1 = loadmat("C:/Users/Games/Pictures/Video_Projects/1_PROUT/1/frame71_mesh.mat")
#
#for i in range(468):
#    keys1 = file1.readline()
#    keys1 = keys1.strip()
#    keys1 = keys1.split()
#    keys1X = keys1[1]
#    keys1Y = keys1[2]
#    p1 = [float(keys1X), float(keys1Y)]
#    distance = 1000
#    for t in range(len(mat1['vertices'])):
#        p2 = [mat1['vertices'][t][0], mat1['vertices'][t][1]]
#        dist = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
#        if dist < distance:
#            indu = t
#            distance = dist
#    print(distance)
#    file.write(keys1[0] + " " + str(indu) + "\n")
#

#from matplotlib import pyplot as plt
#file = open("C:/Users/Games/PycharmProjects/FaceMesh/sav_filt/ims/REFUP2.txt",'w+')
##mat1 = loadmat("C:/Users/Games/Pictures/Video_Projects/1_PROUT/1/frame71_mesh.mat")
#file2 = open("C:/Users/Games/PycharmProjects/FaceMesh/sav_filt/Z2/frame71.txt", 'r')
#arr2x = []
#arr2y = []
#for i in range(468):
#   tex2 = file2.readline()
#   tex2 = tex2.strip()
#   tex2 = tex2.split()
#   if float(tex2[2]) <= 196:
#      file.write(str(i) + '\n')
#      arr2x.append(float(tex2[1]))
#      arr2y.append(float(tex2[2]))
#
#im = plt.imread("G:/FACE_VIDEO/FOLDER/Untitled.png")
#implot = plt.imshow(im)
#plt.scatter(arr2x, arr2y, c="blue", s=1)
#plt.axis('off')
#plt.savefig("G:/FACE_VIDEO/FOLDER/1IM.png", bbox_inches='tight', pad_inches=0, transparent=True)
#matN = loadmat("C:/Users/Games/Pictures/Video_Projects/2_MESH/2/frame71_mesh.mat")
#fileUP = open("C:/Users/Games/PycharmProjects/FaceMesh/sav_filt/ims/REFUP.txt",'r')
#arrUP =  []
#for pl in range(32155):
#    tux = fileUP.readline()
#    tux = tux.strip()
#    tux = int(tux)
#    arrUP.append(tux)
#
#
#
#matN = loadmat("C:/Users/Games/Pictures/Video_Projects/2_MESH/2/frame71_mesh.mat")
#no = matN['vertices']
#noe = []
#noe2 = []
#
#for plo in arrUP:
#    #print(plo)
#    noe.append([matN['vertices'][plo]])
#    noe2.append(no[plo])

#TEST
#from matplotlib import pyplot as plt
#file = open("C:/Users/Games/PycharmProjects/FaceMesh/sav_filt/ims/REF.txt",'r')
#mat1 = loadmat("C:/Users/Games/Pictures/Video_Projects/1_PROUT/1/frame71_mesh.mat")
#arrx = []
#arry = []
#for i in range(468):
#    tex = file.readline()
#    tex = tex.strip()
#    tex = tex.split()
#    ind = tex[1]
#    ind = int(ind)
#    print(ind)
#    arrx.append(mat1['vertices'][ind][0])
#    arry.append(mat1['vertices'][ind][1])
#im = plt.imread("G:/FACE_VIDEO/FOLDER/Untitled.png")
#implot = plt.imshow(im)
#plt.scatter(arrx, arry, c="blue", s=1)
#plt.axis('off')
#plt.savefig("C:/Users/Games/Pictures/Video_Projects/1_PROUT/PLOT.png", bbox_inches='tight', pad_inches=0, transparent=True)

#NEXT
#collection = "C:/Users/Games/Pictures/Video_Projects/2_MESH/2"
#for i, filename in enumerate(os.listdir(collection)):
# file1 = open("C:/Users/Games/PycharmProjects/FaceMesh/sav_filt/ims/REF.txt",'r')
# mat = loadmat("C:/Users/Games/Pictures/Video_Projects/2_MESH/2/" + filename)
# file2 = open("C:/Users/Games/PycharmProjects/FaceMesh/sav_filt/Z2/" + filename[:-9] + ".txt",'r')
# vertOR = mat['vertices']
# vertices = []
# for i in range(468):
#     tex = file1.readline()
#     tex = tex.strip()
#     tex = tex.split()
#     tex = tex[1]
#     tex = int(tex)
#     vertices.append([mat['vertices'][tex][0],mat['vertices'][tex][1],mat['vertices'][tex][2]])
#
# vertices2 = []
# for i in range(468):
#   texN = file2.readline()
#   texN = texN.strip()
#   texN = texN.split()
#   texX = texN[1]
#   texY = texN[2]
#   texZ = texN[3]
#   texX = float(texX)
#   texY = float(texY)
#   texZ = float(texZ)*2
#   vertices2.append([texX,texY,texZ])
# #print(vertices2[1])
# #print("PEEL")
# #print(vertices[1])
# camera_matrix, pose = estimate_pose2(vertices2,vertices)
# sca,R,tra = P2sRt(camera_matrix) # decompose affine matrix to s, R, t
#
# for i in range(len(mat['vertices'])):
#     a = mat['vertices'][i][0]
#     b = mat['vertices'][i][1]
#     c = mat['vertices'][i][2]
#     elem11 = R[0][0]
#     elem12 = R[0][1]
#     elem13 = R[0][2]
#     elem21 = R[1][0]
#     elem22 = R[1][1]
#     elem23 = R[1][2]
#     elem31 = R[2][0]
#     elem32 = R[2][1]
#     elem33 = R[2][2]
#     newx = (elem11 * a) + (elem12 * b) + (elem13 * c)
#     newy = (elem21 * a) + (elem22 * b) + (elem23 * c)
#     newz = (elem31 * a) + (elem32 * b) + (elem33 * c)
#     mat['vertices'][i][0] = newx
#     mat['vertices'][i][1] = newy
#     mat['vertices'][i][2] = newz
#
#
#
#
# vert3 = mat['vertices']
# #savemat("C:/Users/Games/PycharmProjects/FaceMesh/plots/FOLD/" + filename, mat)
# #IMG2 = imread("C:/Users/Games/PycharmProjects/PRNet/Data/uv-data/images/frame0.png")
# #IMG2 = IMG2 / 255.
# #cv2.imshow('desne',plot_vertices(IMG2, vert3))
# #cv2.waitKey(0)
# #
# #cv2.imshow('desne2',plot_vertices(IMG2, vertOR))
# #cv2.waitKey(0)
#
#
# file1 = open("C:/Users/Games/PycharmProjects/FaceMesh/sav_filt/ims/REF.txt",'r')
# mat = loadmat("C:/Users/Games/Pictures/Video_Projects/1_PROUT/1/" + filename)
# file2 = open("C:/Users/Games/PycharmProjects/FaceMesh/sav_filt/Z/" + filename[:-9] + ".txt",'r')
# vertOR = mat['vertices']
# vertices = []
# for i in range(468):
#     tex = file1.readline()
#     tex = tex.strip()
#     tex = tex.split()
#     tex = tex[1]
#     tex = int(tex)
#     vertices.append([mat['vertices'][tex][0],mat['vertices'][tex][1],mat['vertices'][tex][2]])
#
# vertices2 = []
# for i in range(468):
#   texN = file2.readline()
#   texN = texN.strip()
#   texN = texN.split()
#   texX = texN[1]
#   texY = texN[2]
#   texZ = texN[3]
#   texX = float(texX)
#   texY = float(texY)
#   texZ = float(texZ)*2
#   vertices2.append([texX,texY,texZ])
# #print(vertices2[1])
# #print("PEEL")
# #print(vertices[1])
# camera_matrix, pose = estimate_pose2(vertices2,vertices)
# sca,R,tra = P2sRt(camera_matrix) # decompose affine matrix to s, R, t
#
# for i in range(len(mat['vertices'])):
#     a = mat['vertices'][i][0]
#     b = mat['vertices'][i][1]
#     c = mat['vertices'][i][2]
#     elem11 = R[0][0]
#     elem12 = R[0][1]
#     elem13 = R[0][2]
#     elem21 = R[1][0]
#     elem22 = R[1][1]
#     elem23 = R[1][2]
#     elem31 = R[2][0]
#     elem32 = R[2][1]
#     elem33 = R[2][2]
#     newx = (elem11 * a) + (elem12 * b) + (elem13 * c)
#     newy = (elem21 * a) + (elem22 * b) + (elem23 * c)
#     newz = (elem31 * a) + (elem32 * b) + (elem33 * c)
#     mat['vertices'][i][0] = newx
#     mat['vertices'][i][1] = newy
#     mat['vertices'][i][2] = newz
#
#
#
#
# vert5 = mat['vertices']
#
# camera_matrix, pose = estimate_pose2(vert3,vert5)
# sca,R,tra = P2sRt(camera_matrix)
# print(sca)
# for i in range(len(mat['vertices'])):
#     a = mat['vertices'][i][0]*sca
#     b = mat['vertices'][i][1]*sca
#     c = mat['vertices'][i][2]*sca
#     elem11 = R[0][0]
#     elem12 = R[0][1]
#     elem13 = R[0][2]
#     elem21 = R[1][0]
#     elem22 = R[1][1]
#     elem23 = R[1][2]
#     elem31 = R[2][0]
#     elem32 = R[2][1]
#     elem33 = R[2][2]
#     newx = (elem11 * a) + (elem12 * b) + (elem13 * c)
#     newy = (elem21 * a) + (elem22 * b) + (elem23 * c)
#     newz = (elem31 * a) + (elem32 * b) + (elem33 * c)
#     mat['vertices'][i][0] = newx
#     mat['vertices'][i][1] = newy
#     mat['vertices'][i][2] = newz+300
#
# no =   mat['vertices']
# matN = loadmat("C:/Users/Games/Pictures/Video_Projects/2_MESH/2/" + filename)
# camera_matrix, pose = estimate_pose2(matN['vertices'],no)
# sca,R,tra = P2sRt(camera_matrix)
#
# for i in range(len(mat['vertices'])):
#     mat['vertices'][i][0] = mat['vertices'][i][0] + tra[0]
#     mat['vertices'][i][1] = mat['vertices'][i][1]  + tra[1]
#
#
#
#
# #IMG2 = imread("C:/Users/Games/PycharmProjects/PRNet/Data/uv-data/images/frame0.png")
# #IMG2 = IMG2 / 255.
# #cv2.imshow('desne',plot_vertices(IMG2, vertOR))
# #cv2.waitKey(0)
# #cv2.imshow('desne2',plot_vertices(IMG2, no))
# #cv2.waitKey(0)
#
# savemat("C:/Users/Games/PycharmProjects/FaceMesh/plots/FOLD/" + filename,mat)

#matN = loadmat("C:/Users/Games/Pictures/Video_Projects/2_MESH/2/frame0_mesh.mat")
#matN2 = loadmat("C:/Users/Games/Pictures/Video_Projects/1_PROUT/1/frame0_mesh.mat")
#vert1= matN['vertices']
#vert2= matN2['vertices']
#camera_matrix, pose = estimate_pose2(vert1,vert2)
#sca,R,tra = P2sRt(camera_matrix)
#
#a = matN2['vertices'][:,0]
#b = matN2['vertices'][:,1]
#c = matN2['vertices'][:,2]
#elem11 = R[0][0]
#elem12 = R[0][1]
#elem13 = R[0][2]
#elem21 = R[1][0]
#elem22 = R[1][1]
#elem23 = R[1][2]
#elem31 = R[2][0]
#elem32 = R[2][1]
#elem33 = R[2][2]
#newx = (elem11 * a) + (elem12 * b) + (elem13 * c)
#newy = (elem21 * a) + (elem22 * b) + (elem23 * c)
#newz = (elem31 * a) + (elem32 * b) + (elem33 * c)
#matN2['vertices'][:,0] = newx
#matN2['vertices'][:,1] = newy
#matN2['vertices'][:,2] = newz
#savemat("C:/Users/Games/PycharmProjects/FaceMesh/plots/FOLD2/frame0_mesh.mat",matN2)
from matplotlib import pyplot as plt
arrx = []
arry = []
mat = loadmat("C:/Users/Games/Pictures/Video_Projects/2_MESH/2/frame0_mesh.mat")
file = open("C:/Users/Games/PycharmProjects/FaceMesh/sav_filt/ims/REF.txt",'r')
for i in range(468):
    tx = file.readline()
    tx = tx.strip()
    tx = tx.split()
    tx = tx[1]
    tx = int(tx)
    arrx.append(mat['vertices'][tx][0])
    arry.append(mat['vertices'][tx][1])

im = plt.imread("G:/FACE_VIDEO/FOLDER/TESAAAA2_CC.png")
implot = plt.imshow(im)
plt.scatter(arrx, arry, c="blue", s=1)
plt.axis('off')
plt.savefig("G:/FACE_VIDEO/FOLDER/1IM.png", bbox_inches='tight', pad_inches=0, transparent=True)