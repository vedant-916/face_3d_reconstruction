import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import sys
from scipy.io import savemat

def Sort(sub_li):
    return (sorted(sub_li, key=lambda x: x[1]))

def slope(x1, y1, x2, y2):
    return (float)(y2 - y1) / (x2 - x1)

file = open("F:/PRESERVE_SPACE_PROJECTS/3DDFA/OUT/EXPI2/Newfolder3/frame30.txt",'r')
mat  = scipy.io.loadmat("F:/PRESERVE_SPACE_PROJECTS/3DDFA/OUT/EXPI2OUT/New_folder/frame30_mesh.mat")
lins = file.readlines()
leftMar = lins[127]
leftMar = leftMar.strip()
leftMar =leftMar.split()
leftMarx = leftMar[1]
leftMarx = leftMarx.strip()
leftMarx = float(leftMarx)
leftMary = leftMar[2]
leftMary = leftMary.strip()
leftMary = float(leftMary)
rightMar = lins[356]
rightMar = rightMar.strip()
rightMar =rightMar.split()
rightMarx = rightMar[1]
rightMarx = rightMarx.strip()
rightMarx = float(rightMarx)
rightMary = rightMar[2]
rightMary = rightMary.strip()
rightMary = float(rightMary)
key_meshSize = (rightMarx+5) - (leftMarx-5)
file.close()
x = []
y = []
file = open("F:/PRESERVE_SPACE_PROJECTS/3DDFA/OUT/EXPI2/Newfolder3/frame30.txt",'r')
ptref = file.readlines()
ptref = ptref[118]
ptref = ptref.strip()
ptref = ptref.split()
ptref = ptref[1]
ptref = ptref.strip()
ptref = float(ptref)
comb = []
for i in range(43867):
  if abs(mat['vertices'][i][1] -leftMary)<1 and mat['vertices'][i][0]<ptref:
    x.append(float(mat['vertices'][i][2]))
    comb.append([float(mat['vertices'][i][2]),float(mat['vertices'][i][0])])
    y.append(float(mat['vertices'][i][0]))
newx = []
newy = []
comb.sort()
for i in range(len(comb)):
    newx.append(comb[i][0])
    newy.append(comb[i][1])
minslope = 1000
minslopecor = 0
cap = 3
if len(newx)%cap==0:
    for i in range(0,len(newx)-(cap-1),cap):
        x1 = newx[i]
        y1 = newy[i]
        x2 = newx[i+(cap-1)]
        y2 = newy[i+(cap-1)]
        slpe = slope(x1, y1, x2, y2)
        slpe = abs(slpe)
        if slpe<minslope:
            minslope = slpe
            minslopecor = []
            for zil in range(cap):
                minslopecor.append([newx[i + zil], newy[i + zil]])
else:
    df = len(newx)%cap
    for i in range(0,len(newx)-df-(cap-1),cap):
        x1 = newx[i]
        y1 = newy[i]
        x2 = newx[i+(cap-1)]
        y2 = newy[i+(cap-1)]
        slpe = slope(x1, y1, x2, y2)
        slpe = abs(slpe)
        if slpe<minslope:
            minslope = slpe
            minslopecor = []
            for zil in range(cap):
                minslopecor.append([newx[i + zil], newy[i + zil]])
            #minslopecor = [[x1,y1],[newx[i+1],newy[i+1]],[x2,y2]]
#minslopecor = Sort(minslopecor)
#L1 = minslopecor[0][1]
minslopecor.sort()
L1 = minslopecor[int(len(minslopecor)/2)-1][1]
file.close()
x = []
y = []
file = open("F:/PRESERVE_SPACE_PROJECTS/3DDFA/OUT/EXPI2/Newfolder3/frame0.txt",'r')
ptref = file.readlines()
ptref = ptref[347]
ptref = ptref.strip()
ptref = ptref.split()
ptref = ptref[1]
ptref = ptref.strip()
ptref = float(ptref)
comb = []
for i in range(43867):
  if abs(mat['vertices'][i][1] -rightMary)<1 and mat['vertices'][i][0]>ptref:
    x.append(float(mat['vertices'][i][2]))
    comb.append([float(mat['vertices'][i][2]),float(mat['vertices'][i][0])])
    y.append(float(mat['vertices'][i][0]))
newx = []
newy = []
comb.sort(reverse=True)
for i in range(len(comb)):
    newx.append(comb[i][0])
    newy.append(comb[i][1])
minslope = 1000
minslopecor = 0
if len(newx)%cap==0:
    for i in range(0,len(newx)-(cap-1),cap):
        x1 = newx[i]
        y1 = newy[i]
        x2 = newx[i+(cap-1)]
        y2 = newy[i+(cap-1)]
        slpe = slope(x1, y1, x2, y2)
        slpe = abs(slpe)
        if slpe<minslope:
            minslope = slpe
            minslopecor = []
            for zil in range(cap):
                minslopecor.append([newx[i + zil], newy[i + zil]])
            #minslopecor = [[x1,y1],[newx[i+1],newy[i+1]],[x2,y2]]
else:
    df = len(newx)%cap
    for i in range(0,len(newx)-df-(cap-1),cap):
        x1 = newx[i]
        y1 = newy[i]
        x2 = newx[i+(cap-1)]
        y2 = newy[i+(cap-1)]
        slpe = slope(x1, y1, x2, y2)
        slpe = abs(slpe)
        if slpe<minslope:
            minslope = slpe
            minslopecor = []
            for zil in range(cap):
                minslopecor.append([newx[i + zil], newy[i + zil]])
            #minslopecor = [[x1,y1],[newx[i+1],newy[i+1]],[x2,y2]]

#minslopecor = Sort(minslopecor)
minslopecor.sort()
L2 =  minslopecor[int(len(minslopecor)/2)-1][1]
difr = L2-L1
scl_fact = key_meshSize/difr
x_tran = leftMarx-L1
print(scl_fact)
print(x_tran)
#NEW VERT  = VER*SCAL_FACT       VER+ XTRAN

