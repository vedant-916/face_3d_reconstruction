from PRNet.utils.render_app import get_visibility
import scipy.io
import numpy as np
import parser
import random
from scipy.io import savemat
import sys
np.set_printoptions(threshold=sys.maxsize)

path = "C:/Users/Games/Pictures/Video_Projects/TEST_OUT/1/frame0_mesh.mat"
mat  = scipy.io.loadmat(path)
vertices = mat['vertices']
triangles = mat['triangles']
vertices_vis = get_visibility(vertices, triangles, 768, 422)
#print(str(vertices_vis))
#print(len(vertices_vis))
#print(len(vertices))
for i in range(len(vertices)):
    tr_value = vertices_vis[i]
    if tr_value == 0:
        mat['colors'][i] = random.choice([[1,0,0],[1,0,0]])


 #print(mat['colors'])
savemat("C:/Users/Games/Pictures/Video_Projects/TEST_OUT/1/frame02_mesh.mat",mat)

