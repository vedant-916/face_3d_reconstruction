import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import cv2
from PRNet.api_for_fixr_EDIT import PRN
from PIL import Image
import sys
from PRNet.utils.write import write_obj_with_colors, write_obj_with_texture
from skimage.io import imread, imsave
from scipy.io import savemat
import math
#prn = PRN(is_dlib=False)
#mat = scipy.io.loadmat(r"C:\Users\Games\Videos\Captures/2.mat")
##mat['triangles'] = mat['triangles'] - 1
#
#for i in range(len(mat['vertices'])):
#
#    if mat['vertices'][i][0]<315 and mat['vertices'][i][2]<300:
#        print("P")
#        mat['colors'][i] = [0,0,0]
#
#write_obj_with_colors(r"C:\Users\Games\Videos\Captures/2N.obj",mat['vertices'], mat['triangles'],mat['colors'])  # save 3d face(can open with meshlab)
#
##savemat(os.path.join(r'C:\Users\Games\Videos\Captures/2_mesh.mat'),{'vertices': mat['vertices'], 'colors': colors, 'triangles': prn.triangles})




################################################################################################################################################################################



#mat = scipy.io.loadmat(r"C:\Users\Games\Pictures\Video_Projects\1_PROUT\1_STRAIGHTENED2/frame0_mesh.mat")
#elem11 = 0.5
#elem12 = 0
#elem13 = -0.86602540378
#elem21 = 0
#elem22 = 1
#elem23 = 0
#elem31 = 0.86602540378
#elem32 = 0
#elem33 = 0.5
#
#a ,b,c= mat['vertices'][:,0],mat['vertices'][:,1],mat['vertices'][:,2]
#newx,newy,newz = ((elem11 * a) + (elem12 * b) + (elem13 * c)),((elem21 * a) + (elem22 * b) + (elem23 * c)),((elem31 * a) + (elem32 * b) + (elem33 * c))
#mat['vertices'][:,0],mat['vertices'][:,1],mat['vertices'][:,2] = newx,newy,newz
#savemat(r"C:\Users\Games\Videos\Captures/ROT_L.mat",mat)
#mat['vertices'][:,1] = 1080-mat['vertices'][:,1]
#
#file = open(r"C:\Users\Games\Videos\Captures/LEAR_indices.txt",'w+')
#for i in range(len(mat['vertices'])):
#    if mat['vertices'][i][2]>111 and mat['vertices'][i][0]>544:
#        file.write(str(i) + "\n")
#        mat['colors'][i] = [0,0,0]
#write_obj_with_colors(r"C:\Users\Games\Videos\Captures/ROT_L.obj",mat['vertices'], mat['triangles'],mat['colors'])  # save 3d face(can open with meshlab)


#elem11 = 0.5
#elem12 = 0
#elem13 = 0.86602540378
#elem21 = 0
#elem22 = 1
#elem23 = 0
#elem31 = -0.86602540378
#elem32 = 0
#elem33 = 0.5
#
#a ,b,c= mat['vertices'][:,0],mat['vertices'][:,1],mat['vertices'][:,2]
#newx,newy,newz = ((elem11 * a) + (elem12 * b) + (elem13 * c)),((elem21 * a) + (elem22 * b) + (elem23 * c)),((elem31 * a) + (elem32 * b) + (elem33 * c))
#mat['vertices'][:,0],mat['vertices'][:,1],mat['vertices'][:,2] = newx,newy,newz
#mat['vertices'][:,0] = mat['vertices'][:,0]+700
#savemat(r"C:\Users\Games\Videos\Captures/ROT_R.mat",mat)
#mat['vertices'][:,1] = 1080-mat['vertices'][:,1]
#
#file = open(r"C:\Users\Games\Videos\Captures/REAR_indices.txt",'w+')
#for i in range(len(mat['vertices'])):
#    if mat['vertices'][i][2]>-331 and mat['vertices'][i][0]<415:
#        file.write(str(i) + "\n")
#        mat['colors'][i] = [0,0,0]
#write_obj_with_colors(r"C:\Users\Games\Videos\Captures/ROT_R.obj",mat['vertices'], mat['triangles'],mat['colors'])  # save 3d face(can open with meshlab)

#mat = scipy.io.loadmat(r"F:\PRESERVE_SPACE_PROJECTS\3DDFA\samples\EAR_COMB_ROT_MASK2/frame5.mat")
#mat['triangles'] = mat['triangles'] - 1
#file = open(r"C:\Users\Games\Videos\Captures/EAR_ALINGN_INDICES_RIGHT.txt",'r')
#for i in range(1082):
#    txt = file.readline()
#    txt = txt.strip()
#    txt = txt.split()
#    txt = int(float(txt[0]))
#    mat['colors'][txt] = [0, 0, 0]
#mat['vertices'][:,1] = 1080-mat['vertices'][:,1]
#write_obj_with_colors(r"C:\Users\Games\Videos\Captures/ROT_TESTR.obj",mat['vertices'], mat['triangles'],mat['colors'])  # save 3d face(can open with meshlab)

#collection = r"C:\Users\Games\Pictures\Video_Projects\1_PROUT\FINISH_MESH"
#file = open(r"C:\Users\Games\Videos\Captures/LEAR_indices.txt",'r')
#EAR_ARR = []
#for i in range(896):
#    txt = file.readline()
#    txt = txt.strip()
#    txt = int(float(txt))
#    EAR_ARR.append(txt)
#file = open(r"C:\Users\Games\Videos\Captures/REAR_indices.txt",'r')
#for i in range(894):
#    txt = file.readline()
#    txt = txt.strip()
#    txt = int(float(txt))
#    EAR_ARR.append(txt)
#for k, filename in enumerate(os.listdir(collection)):
#    mat = scipy.io.loadmat(r"C:\Users\Games\Pictures\Video_Projects\1_PROUT\FINISH_MESH/" + filename)
#    mat['colors'][EAR_ARR] = [0,0,0]
#    savemat(r"C:\Users\Games\Videos\Captures\MESH/" + filename,mat)


from math import cos, sin, atan2, asin
def matrix2angle(R):
    if R[2,0] !=1 or R[2,0] != -1:
        x = asin(R[2,0])
        y = atan2(R[2,1]/cos(x), R[2,2]/cos(x))
        z = atan2(R[1,0]/cos(x), R[0,0]/cos(x))
    else:# Gimbal lock
        z = 0 #can be anything
        if R[2,0] == -1:
            x = np.pi/2
            y = z + atan2(R[0,1], R[0,2])
        else:
            x = -np.pi/2
            y = -z + atan2(-R[0,1], -R[0,2])
    return x, y, z

def P2sRt(P):
    t2d = P[:2, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2.0
    r1 = R1/np.linalg.norm(R1)
    r2 = R2/np.linalg.norm(R2)
    r3 = np.cross(r1, r2)
    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t2d

def compute_similarity_transform(points_static, points_to_transform):
    #http://nghiaho.com/?page_id=671
    p0 = np.copy(points_static).T
    p1 = np.copy(points_to_transform).T
    t0 = -np.mean(p0, axis=1).reshape(3,1)
    t1 = -np.mean(p1, axis=1).reshape(3,1)
    t_final = t1 -t0
    p0c = p0+t0
    p1c = p1+t1
    covariance_matrix = p0c.dot(p1c.T)
    U,S,V = np.linalg.svd(covariance_matrix)
    R = U.dot(V)
    if np.linalg.det(R) < 0:
        R[:,2] *= -1
    rms_d0 = np.sqrt(np.mean(np.linalg.norm(p0c, axis=0)**2))
    rms_d1 = np.sqrt(np.mean(np.linalg.norm(p1c, axis=0)**2))
    s = (rms_d0/rms_d1)
    P = np.c_[s*np.eye(3).dot(R), t_final]
    return P

def estimate_pose2(vertices2,vertices1):
    P = compute_similarity_transform(vertices2, vertices1)
    _,R,_ = P2sRt(P) # decompose affine matrix to s, R, t
    pose = matrix2angle(R)
    return P, pose


collection = r"E:\TEST\PR\RESIZEDN"
for k, filename in enumerate(os.listdir(collection)):
  mats = scipy.io.loadmat(r"E:\TEST\PR\RESIZEDN/" + filename)
  matt = scipy.io.loadmat(r"E:\TEST\PR\2/" + filename)
  camera_matrix, pose = estimate_pose2(matt['vertices'], mats['vertices'])
  sca, R, tra = P2sRt(camera_matrix)
  mats['vertices'] = mats['vertices']*sca
  #mats['vertices'][:,0] = mats['vertices'][:,0] + tra[0]
  #mats['vertices'][:,1] = mats['vertices'][:,1] + tra[1]
  #print(tra)
  savemat(r"E:\TEST\PR\SHIFT_MAT_SIZ/" + filename,mats)
  img = cv2.imread(r"C:\Users\Games\Pictures\Video_Projects\1_PIX2PIXOUTP\Final52\test_latest\FOR_SCFE\RESIZEDN/" + filename[:-9] + ".png")
  img = cv2.resize(img, (int(1920*sca), int(1080*sca)), interpolation=cv2.INTER_LINEAR)
  #rows, cols, ch = img.shape
  #matrix = np.float32([[1, 0, math.ceil(tra[0])], [0, 1, math.ceil(tra[1])]])
  #imgB = cv2.warpAffine(img, matrix, (cols,rows), borderValue=(150, 20, 13))
  cv2.imwrite(r"E:\TEST\PR\SHIFTED_SIZ/" + filename[:-9] + ".png",img)


#ANS_ARR = []
#ARR_IND = []
#file = open(r"C:\Users\Games\Videos\Captures/EAR_ALINGN_INDICES_LEFT.txt",'r')
#for i in range(1082):
#    txt = file.readline()
#    txt = txt.strip()
#    txt = txt.split()
#    ANS_ARR.append(int(txt[1]))
#    ARR_IND.append(int(txt[0]))
#
#collection = r"C:\Users\Games\Videos\Captures\MESH"
#for k, filename in enumerate(os.listdir(collection)):
#  mat_src = scipy.io.loadmat(r"F:\PRESERVE_SPACE_PROJECTS\3DDFA\samples\PROUT\OUTR/frame0.mat")
#  mat_tar = scipy.io.loadmat(r"C:\Users\Games\Videos\Captures\MESH/" + filename)
#  src_arr = []
#  for i in ANS_ARR:
#      src_arr.append([mat_src['vertex'][i][0],mat_src['vertex'][i][1],mat_src['vertex'][i][2]])
#  src_arr = np.array(src_arr)
#  tar_arr = []
#  for i in ARR_IND:
#      tar_arr.append([mat_tar['vertices'][i][0],mat_tar['vertices'][i][1],mat_tar['vertices'][i][2]])
#  tar_arr = np.array(tar_arr)
#  y = tar_arr[:, 1]
#  x = tar_arr[:, 0]
#  fig, ax = plt.subplots()
#  ax.scatter(x, y)
#  for i in range(len(x)):
#      ax.annotate(i, (x[i], y[i]))
#  #plt.savefig(r"C:\Users\Games\Videos\MASK/" + filename[:-9] + ".png")
#  plt.close()
#  camera_matrix, pose = estimate_pose2(tar_arr, src_arr)
#  sca, R, tra = P2sRt(camera_matrix)
#  mat_src['vertex'] = mat_src['vertex']*sca
#  src_arr = src_arr*sca
#  camera_matrix, pose = estimate_pose2(tar_arr, src_arr)
#  sca, R, tra = P2sRt(camera_matrix)
#  elem11, elem12, elem13, elem21, elem22, elem23, elem31, elem32, elem33 = R[0][0], R[0][1], R[0][2], R[1][0], R[1][1], R[1][2], R[2][0], R[2][1],R[2][2]
#  a, b, c = mat_src['vertex'][:, 0], mat_src['vertex'][:, 1], mat_src['vertex'][:, 2]
#  newx, newy, newz = ((elem11 * a) + (elem12 * b) + (elem13 * c)), ((elem21 * a) + (elem22 * b) + (elem23 * c)), ((elem31 * a) + (elem32 * b) + (elem33 * c))
#  mat_src['vertex'][:, 0], mat_src['vertex'][:, 1], mat_src['vertex'][:, 2] = newx, newy, newz
#  a, b, c = src_arr[:, 0], src_arr[:, 1], src_arr[:, 2]
#  newx, newy, newz = ((elem11 * a) + (elem12 * b) + (elem13 * c)), ((elem21 * a) + (elem22 * b) + (elem23 * c)), ((elem31 * a) + (elem32 * b) + (elem33 * c))
#  src_arr[:, 0], src_arr[:, 1], src_arr[:, 2] = newx, newy, newz
#  camera_matrix, pose = estimate_pose2(tar_arr, src_arr)
#  sca, R, tra = P2sRt(camera_matrix)
#  mat_src['vertex'][:, 0] = mat_src['vertex'][:, 0]+tra[0]
#  mat_src['vertex'][:, 1] = mat_src['vertex'][:, 1] + tra[1]
#  savemat(r"C:\Users\Games\Videos\Captures\MESH_CONT/" + filename,mat_src)