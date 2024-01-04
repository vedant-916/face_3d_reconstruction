import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
from PRNet.api_for_fixr_EDIT import PRN
from PIL import Image
import sys
import time
from matplotlib.tri import Triangulation, LinearTriInterpolator, CubicTriInterpolator
from scipy.interpolate import LinearNDInterpolator
from PRNet.utils.write import write_obj_with_colors, write_obj_with_texture
from skimage.io import imread, imsave
from scipy.io import savemat
import math
def area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                + x3 * (y1 - y2)) / 2.0)


def isInside(x1, y1, x2, y2, x3, y3, x, y):
    # Calculate area of triangle ABC
    A = area(x1, y1, x2, y2, x3, y3)

    # Calculate area of triangle PBC
    A1 = area(x, y, x2, y2, x3, y3)

    # Calculate area of triangle PAC
    A2 = area(x1, y1, x, y, x3, y3)

    # Calculate area of triangle PAB
    A3 = area(x1, y1, x2, y2, x, y)

    # Check if sum of A1, A2 and A3
    # is same as A
    sum = A1+A2+A3
    sum = round(sum,2)
    A = round(A,2)
    if (A == sum):
        return True
    else:
        return False

def interp_point(x, y, X, Y, Z):
 """
 x, y: scalar coordinates to interpolate at
 X, Y, Z: arrays of coordinates corresponding to function
 """
 X_OLD = list(X)
 Y_OLD = list(Y)
 Z_OLD = list(Z)
 X = X.ravel()
 Y = Y.ravel()
 Z = Z.ravel()

 # distances from x, y to all X, Y points
 dist = np.hypot(X - x, Y - y)
 # indices of the nearest points
 nearest3 = np.argpartition(dist, 2)[:3]
 # extract the coordinates
 points = np.stack((X[nearest3], Y[nearest3], Z[nearest3]))
 p1 = np.array([points[0][0], points[1][0], points[2][0]])
 p2 = np.array([points[0][1], points[1][1], points[2][1]])
 p3 = np.array([points[0][2], points[1][2], points[2][2]])

 if (isInside(p1[0], p1[1], p2[0], p2[1], p3[0],p3[1], x, y)):
  v1 = p3 - p1
  v2 = p2 - p1
  cp = np.cross(v1, v2)
  a, b, c = cp
  d = np.dot(cp, p3)
  XQ = x
  YQ = y
  ZQ = (d - a * XQ - b * YQ) / c
  return ZQ

 else:
  pT1 = [x, y]
  pT2 = p1
  distance1 = math.sqrt(((pT1[0] - pT2[0]) ** 2) + ((pT1[1] - pT2[1]) ** 2))

  pT1 = [x, y]
  pT2 = p2
  distance2 = math.sqrt(((pT1[0] - pT2[0]) ** 2) + ((pT1[1] - pT2[1]) ** 2))

  pT1 = [x, y]
  pT2 = p3
  distance3 = math.sqrt(((pT1[0] - pT2[0]) ** 2) + ((pT1[1] - pT2[1]) ** 2))

  minna = min(distance1,distance2,distance3)
  if minna==distance1:
     ite = X_OLD.index(p1[0])
     X_OLD.pop(ite)
     Y_OLD.pop(ite)
     Z_OLD.pop(ite)
     X_OLD = np.array(X_OLD)
     Y_OLD = np.array(Y_OLD)
     Z_OLD = np.array(Z_OLD)
     return interp_point(x,y,X_OLD,Y_OLD,Z_OLD)




  elif minna==distance2:
   ite = X_OLD.index(p2[0])
   X_OLD.pop(ite)
   Y_OLD.pop(ite)
   Z_OLD.pop(ite)
   X_OLD = np.array(X_OLD)
   Y_OLD = np.array(Y_OLD)
   Z_OLD = np.array(Z_OLD)
   return interp_point(x, y, X_OLD, Y_OLD, Z_OLD)


  elif minna== distance3:
   ite = X_OLD.index(p3[0])
   X_OLD.pop(ite)
   Y_OLD.pop(ite)
   Z_OLD.pop(ite)
   X_OLD = np.array(X_OLD)
   Y_OLD = np.array(Y_OLD)
   Z_OLD = np.array(Z_OLD)
   return interp_point(x, y, X_OLD, Y_OLD, Z_OLD)



prn = PRN(is_dlib=False)
collection = r"F:\PRESERVE_SPACE_PROJECTS\face_parsing\RES\STOR_MAT2"
for k, filename in enumerate(os.listdir(collection)):
    os.remove(r"F:\PRESERVE_SPACE_PROJECTS\face_parsing\RES\STOR_MAT2/" + filename)

collection = r"F:\PRESERVE_SPACE_PROJECTS\face_parsing\RES\STOR_MAT2P"
for k, filename in enumerate(os.listdir(collection)):
    os.remove(r"F:\PRESERVE_SPACE_PROJECTS\face_parsing\RES\STOR_MAT2P/" + filename)



collection = r"F:\PRESERVE_SPACE_PROJECTS\face_parsing\RES\New folder"
for k, filename in enumerate(os.listdir(collection)):
 mat1  = scipy.io.loadmat(r"F:\PRESERVE_SPACE_PROJECTS\3DDFA\OUT\EXPI2OUT\Newfolder2/" + filename[:-4] + "_mesh.mat")
 image2 = imread(r"F:\PRESERVE_SPACE_PROJECTS\face_parsing\RES\TAR/" +filename[:-4] +  "/TRAP15.png")
 image2 = image2 / 255
 arr = []
 arr2 = []
 ind = mat1['vertices'].astype(np.int32)
 file1 = open(r"F:\PRESERVE_SPACE_PROJECTS\face_parsing\RES/TAR/" +filename[:-4]+ "/ZIMMER.txt",'r')
 txt = file1.readline()
 txt = txt.strip()
 txt = float(txt)
 txt = int(txt)
 L_arr = mat1['vertices'][:,0]
 #print(txt)
 I = np.where(L_arr > txt)
 I = I[0]

 for jim in I:
    #print(str(ind[I[jim], 1]) + " " + str(ind[I[jim], 0]))
    if image2[ind[jim, 1], ind[jim, 0], 0] != 0 and image2[ind[jim, 1], ind[jim, 0], 1] != 0 and image2[ind[jim, 1], ind[jim, 0], 2] != 0:
       arr.append(jim)

 file1.close()

#print(arr)

collection = r"C:\Users\Games\Pictures\Video_Projects\1_PROUT\1_STRAIGHTENED"
for k, filename in enumerate(os.listdir(collection)):
    #stt = time.time()
    triObj = 0
    fz = 0
    file_ks = open(r"C:\Users\Games\Pictures\Video_Projects\1_PROUT\STRAIGHTENED_KPTS/" + filename[:-9] + ".txt", 'r')
    txtLAS = file_ks.readlines()
    txtLAS = txtLAS[300]
    txtLAS = txtLAS.strip()
    txtLAS = txtLAS.split()
    XLAS = txtLAS[1]
    YLAS = txtLAS[2]
    XLAS = float(XLAS)
    XLAS = int(XLAS)
    YLAS = float(YLAS)
    YLAS = int(YLAS)
    file_ks.close()

    inter_mat = scipy.io.loadmat(r"C:\Users\Games\Pictures\Video_Projects\1_PROUT\1_STRAIGHTENED/" + filename)
    arrx = []
    arry = []
    arrz = []
    arrx = inter_mat['vertices'][:,0]
    arry = inter_mat['vertices'][:, 1]
    arrz = inter_mat['vertices'][:, 2]
    #Xv = arrx
    #Yv = arry
    #
    #triObj = Triangulation(Xv,Yv)
    #Zv = arrz
    #fz = LinearTriInterpolator(triObj,Zv)
    #ZIM = fz(XLAS,YLAS)


    #for i in range(43867):
    #    x = inter_mat['vertices'][i][0]
    #    y = inter_mat['vertices'][i][1]
    #    z = inter_mat['vertices'][i][2]
    #    arrx.append(x)
    #    arry.append(y)
    #    arrz.append(z)
    arrx = np.array(arrx)
    arry = np.array(arry)
    arrz = np.array(arrz)
    ARRX1Y1 = np.zeros([43867, 2])
    ARRZ1 = np.zeros([43867, 1])
    ARRX1Y1[:, 0] = inter_mat['vertices'][:, 0]
    ARRX1Y1[:, 1] = inter_mat['vertices'][:, 1]
    ARRZ1[:, 0] = inter_mat['vertices'][:, 2]
    interp = LinearNDInterpolator(ARRX1Y1, ARRZ1)
    ZIM = interp(XLAS, YLAS)
    ZIM = ZIM[0]
    #eno = time.time()
    #print(ZIM)
    #print(eno - stt)
    elem11 = 0.5
    elem12 = 0
    elem13 = -0.86602540378
    elem21 = 0
    elem22 = 1
    elem23 = 0
    elem31 = 0.86602540378
    elem32 = 0
    elem33 = 0.5

    a = XLAS
    b = YLAS
    c = ZIM
    newx = (elem11 * a) + (elem12 * b) + (elem13 * c)
    newy = (elem21 * a) + (elem22 * b) + (elem23 * c)
    newz = (elem31 * a) + (elem32 * b) + (elem33 * c)
    newx = newx + 200

    elem11 = 0.5
    elem12 = 0
    elem13 = -0.86602540378
    elem21 = 0
    elem22 = 1
    elem23 = 0
    elem31 = 0.86602540378
    elem32 = 0
    elem33 = 0.5

    a, b, c = inter_mat['vertices'][:, 0], inter_mat['vertices'][:, 1], inter_mat['vertices'][:, 2]
    inter_mat['vertices'][:,0], inter_mat['vertices'][:,1], inter_mat['vertices'][:,2] = ((elem11 * a) + (elem12 * b) + (elem13 * c)), ((elem21 * a) + (elem22 * b) + (elem23 * c)), ((elem31 * a) + (elem32 * b) + (elem33 * c))
    inter_mat['vertices'][:,0] = inter_mat['vertices'][:,0] + 200
    inter_mat['vertices'][:, 1] = inter_mat['vertices'][:, 1]
    inter_mat['vertices'][:, 2] = inter_mat['vertices'][:, 2]

    #for i in range(43867):
    #    a = inter_mat['vertices'][i][0]
    #    b = inter_mat['vertices'][i][1]
    #    c = inter_mat['vertices'][i][2]
    #    inter_mat['vertices'][i][0] = (elem11 * a) + (elem12 * b) + (elem13 * c)
    #    inter_mat['vertices'][i][1]= (elem21 * a) + (elem22 * b) + (elem23 * c)
    #    inter_mat['vertices'][i][2] = (elem31 * a) + (elem32 * b) + (elem33 * c)
    #    inter_mat['vertices'][i][0] = inter_mat['vertices'][i][0] +200
    #    inter_mat['vertices'][i][1] = inter_mat['vertices'][i][1]
    #    inter_mat['vertices'][i][2] = inter_mat['vertices'][i][2]

    file_ks = open(r"C:\Users\Games\Pictures\Video_Projects\1_PROUT\STRAIGHTENED_KPTS/" + filename[:-9] + ".txt", 'r')
    txts = file_ks.readlines()
    txts = txts[5]
    txts = txts.strip()
    txts = txts.split()
    txts = txts[1]
    txts = float(txts)
    txts = int(txts)

    imm = Image.open(r"C:\Users\Games\Pictures\Video_Projects\2/" + filename[:-9] + ".png")
    wid, hei = imm.size
    mat = scipy.io.loadmat(r"C:\Users\Games\Pictures\Video_Projects\1_PROUT\1_STRAIGHTENED/" + filename)
    matP = scipy.io.loadmat(r"C:\Users\Games\Pictures\Video_Projects\1_PROUT\1_STRAIGHTENED/" + filename)
    for i in arr:
        if mat['vertices'][i][0] > txts and inter_mat['vertices'][i][0] > newx:
            mat['colors'][i] = image2[ind[i, 1], ind[i, 0], :]
            matP['colors'][i] = [0, 0, 1]
    matn = mat
    matn2 = matP

    elem11 = 0.5
    elem12 = 0
    elem13 = -0.86602540378
    elem21 = 0
    elem22 = 1
    elem23 = 0
    elem31 = 0.86602540378
    elem32 = 0
    elem33 = 0.5

    a, b, c =  matn['vertices'][:,0], matn['vertices'][:,1], matn['vertices'][:,2]
    newx, newy, newz = ((elem11 * a) + (elem12 * b) + (elem13 * c) + 600), ((elem21 * a) + (elem22 * b) + (elem23 * c)), ((elem31 * a) + (elem32 * b) + (elem33 * c))
    matn['vertices'][:,0] = newx
    matn['vertices'][:, 1] = newy
    matn['vertices'][:, 2] = newz

    matn2['vertices'][:,0] = newx
    matn2['vertices'][:, 1] = newy
    matn2['vertices'][:, 2] = newz




    #for i in range(43867):
    #    a = matn['vertices'][i][0]
    #    b = matn['vertices'][i][1]
    #    c = matn['vertices'][i][2]
    #    newx = (elem11 * a) + (elem12 * b) + (elem13 * c)
    #    newx = (elem11 * a) + (elem12 * b) + (elem13 * c)
    #    newy = (elem21 * a) + (elem22 * b) + (elem23 * c)
    #    newz = (elem31 * a) + (elem32 * b) + (elem33 * c)
    #    matn['vertices'][i][0]= newx +600
    #    matn['vertices'][i][1] = newy
    #    matn['vertices'][i][2] = newz
    #    matn2['vertices'][i][0] = newx +600
    #    matn2['vertices'][i][1] = newy
    #    matn2['vertices'][i][2] = newz

    #mat['vertices'][:,1] = hei - mat['vertices'][:,1]
    #matn['vertices'][:,1] = hei - matn['vertices'][:,1]
    #for i in range(len(mat['vertices'])):
    #     #mat['vertices'][i][1] =  hei- mat['vertices'][i][1]
    #     matn['vertices'][i][1] = hei - matn['vertices'][i][1]
    #matn['vertices'][:,1] = hei - matn['vertices'][:,1]
    savemat(r"F:\PRESERVE_SPACE_PROJECTS\face_parsing\RES\STOR_MAT2/" + filename, matn)
    savemat(r"F:\PRESERVE_SPACE_PROJECTS\face_parsing\RES\STOR_MAT2P/" + filename, matn2)

    #write_obj_with_colors(r"F:\PRESERVE_SPACE_PROJECTS\face_parsing\RES\STOR/" + filename[:-4] + ".obj",mat['vertices'], prn.triangles, mat['colors'])  # save 3d face(can open with meshlab)

