import numpy as np
import os
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp
from time import time
import cv2
from PRNet.utils.estimate_pose import estimate_pose2
from PRNet.utils.estimate_pose import P2sRt
from PIL import Image
from PRNet.predictor import PosPrediction
import math
class PRN:
    ''' Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network
    Args:
        is_dlib(bool, optional): If true, dlib is used for detecting faces.
        prefix(str, optional): If run at another folder, the absolute path is needed to load the data.
    '''
    def __init__(self, is_dlib = False, prefix = '.'):

        # resolution of input and output image size.
        self.resolution_inp = 256
        self.resolution_op = 256

        #---- load detectors
        if is_dlib:
            import dlib
            detector_path = os.path.join(prefix, 'Data/net-data/mmod_human_face_detector.dat')
            self.face_detector = dlib.cnn_face_detection_model_v1(
                    detector_path)

        #---- load PRN 
        self.pos_predictor = PosPrediction(self.resolution_inp, self.resolution_op)
        prn_path = os.path.join(prefix, 'Data/net-data/256_256_resfcn256_weight')
        if not os.path.isfile(prn_path + '.data-00000-of-00001'):
            print("please download PRN trained model first.")
            exit()
        self.pos_predictor.restore(prn_path)

        # uv file
        self.uv_kpt_ind = np.loadtxt(prefix + '/Data/uv-data/uv_kpt_ind.txt').astype(np.int32) # 2 x 68 get kpt
        self.face_ind = np.loadtxt(prefix + '/Data/uv-data/face_ind.txt').astype(np.int32) # get valid vertices in the pos map
        self.triangles = np.loadtxt(prefix + '/Data/uv-data/triangles.txt').astype(np.int32) # ntri x 3
        
        self.uv_coords = self.generate_uv_coords()        

    def generate_uv_coords(self):
        resolution = self.resolution_op
        uv_coords = np.meshgrid(range(resolution),range(resolution))
        uv_coords = np.transpose(np.array(uv_coords), [1,2,0])
        uv_coords = np.reshape(uv_coords, [resolution**2, -1]);
        uv_coords = uv_coords[self.face_ind, :]
        uv_coords = np.hstack((uv_coords[:,:2], np.zeros([uv_coords.shape[0], 1])))
        return uv_coords

    def dlib_detect(self, image):
        return self.face_detector(image, 1)

    def net_forward(self, image):
        ''' The core of out method: regress the position map of a given image.
        Args:
            image: (256,256,3) array. value range: 0~1
        Returns:
            pos: the 3D position map. (256, 256, 3) array.
        '''
        return self.pos_predictor.predict(image)

    def process(self, input, image_info = None):
        ''' process image with crop operation.
        Args:
            input: (h,w,3) array or str(image path). image value range:1~255. 
            image_info(optional): the bounding box information of faces. if None, will use dlib to detect face. 

        Returns:
            pos: the 3D position map. (256, 256, 3).
        '''
        if isinstance(input, str):
            try:
                image = imread(input)
            except IOError:
                print("error opening file: ", input)
                return None
        else:
            image = input

        if image.ndim < 3:
            image = np.tile(image[:,:,np.newaxis], [1,1,3])

        if image_info is not None:
            if np.max(image_info.shape) > 4: # key points to get bounding box
                kpt = image_info
                if kpt.shape[0] > 3:
                    kpt = kpt.T
                left = np.min(kpt[0, :]); right = np.max(kpt[0, :]); 
                top = np.min(kpt[1,:]); bottom = np.max(kpt[1,:])
                #print(left)
                #print(right)
                #print(top)
                #print(bottom)
            else:  # bounding box
                bbox = image_info
                left = bbox[0]; right = bbox[1]; top = bbox[2]; bottom = bbox[3]
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
            size = int(old_size*1.6)
        else:
            detected_faces = self.dlib_detect(image)
            if len(detected_faces) == 0:
                print('warning: no detected face')
                return None

            d = detected_faces[0].rect ## only use the first detected face (assume that each input image only contains one face)
            left = d.left(); right = d.right(); top = d.top(); bottom = d.bottom()
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size*0.14])
            size = int(old_size*1.58)

        # crop image
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        image = image/255.
        cropped_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        #img2  = imread("F:/PRESERVE_SPACE_PROJECTS/3DDFA/OUT/EXPI2/MASK3.png")
        #img2 = img2/255
        #crop2 =  warp(img2, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        # run our net
        #st = time()
        #imsave("F:/PRESERVE_SPACE_PROJECTS/3DDFA/OUT/EXPI2/im5N.png",crop2)
        #imsave("F:/PRESERVE_SPACE_PROJECTS/3DDFA/OUT/EXPI2/im5.png",cropped_image)
        cropped_pos = self.net_forward(cropped_image)
        #print(cropped_pos[0,0,:2])
        #im = Image.open("F:/PRESERVE_SPACE_PROJECTS/3DDFA/OUT/EXPI2/im5.png")
        #mapi = im.load()
        #print(cropped_pos[0,0,0])
        #for i in range(256):
        #    for j in range(256):
        #        r = cropped_pos[j,i,0]
        #        r = int(r)
        #        g = cropped_pos[j, i,1]
        #        g = int(g)
        #        b = cropped_pos[j, i,2]
        #        b = int(b)
        #        mapi[i,j] = (r,g,b)
        #im.save("F:/PRESERVE_SPACE_PROJECTS/3DDFA/OUT/EXPI2/POSF.png")
#
        #imsave("F:/PRESERVE_SPACE_PROJECTS/3DDFA/OUT/EXPI2/POS.png",cropped_pos)
        #print 'net time:', time() - st
        #cropped_pos = imread("F:/PRESERVE_SPACE_PROJECTS/3DDFA/OUT/EXPI2/POSF30N.png")
        # restore 
        cropped_vertices = np.reshape(cropped_pos, [-1, 3]).T
        z = cropped_vertices[2,:].copy()/tform.params[0,0]
        cropped_vertices[2,:] = 1
        vertices = np.dot(np.linalg.inv(tform.params), cropped_vertices)
        vertices = np.vstack((vertices[:2,:], z))
        pos = np.reshape(vertices.T, [self.resolution_op, self.resolution_op, 3])
        #imsave("F:/PRESERVE_SPACE_PROJECTS/3DDFA/OUT/EXPI2/POS2.png", pos)
        return pos
            
    def get_landmarks(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            kpt: 68 3D landmarks. shape = (68, 3).
        '''
        kpt = pos[self.uv_kpt_ind[1,:], self.uv_kpt_ind[0,:], :]
        print(kpt)
        return kpt


    def get_vertices(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
        '''
        all_vertices = np.reshape(pos, [self.resolution_op**2, -1])
        vertices = all_vertices[self.face_ind, :]

        return vertices

    def get_colors_from_texture(self, texture):
        '''
        Args:
            texture: the texture map. shape = (256, 256, 3).
        Returns:
            colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
        '''
        all_colors = np.reshape(texture, [self.resolution_op**2, -1])
        colors = all_colors[self.face_ind, :]

        return colors


    def get_colors(self, image, vertices):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
        '''
        [h, w, _] = image.shape
        vertices[:,0] = np.minimum(np.maximum(vertices[:,0], 0), w - 1)  # x
        vertices[:,1] = np.minimum(np.maximum(vertices[:,1], 0), h - 1)  # y
        #print(vertices[0])
        #print(str(vertices[0][0]) + " " + str(vertices[0][1]) + " " + str(vertices[0][2]))
        #vertices2 = vertices * 0.97
        #camera_matrix, pose = estimate_pose2(vertices,vertices2)
        #sca,R,tra = P2sRt(camera_matrix) # decompose affine matrix to s, R, t
        #print(tra)
        #for ti in range(43867):
        #    vertices[ti][0] = vertices[ti][0] -7
        #camera_matrix, pose = estimate_pose2(vertices, vertices2)
        #sca, R, tra = P2sRt(camera_matrix)  # decompose affine matrix to s, R, t
        #for ti in range(43867):
        #    vertices2[ti][0] =  vertices2[ti][0]+tra[0]-5
        #    vertices2[ti][1] = vertices2[ti][1] + tra[1]
        #vertices = vertices2

            #img = Image.open("F:/PRESERVE_SPACE_PROJECTS/3DDFA/OUT/EXPI/MASK2.png")
        #for ti in range(43867):
        #    xc = vertices[ti][0]
        #    xc = math.ceil(xc)
        #    yc = vertices[ti][1]
        #    yc = math.ceil(yc)
        #    arrx = []
        #    arry = []
        #    for i in range(736):
        #        pix = img.getpixel((i,yc))
        #        if pix!=(0,0,0):
        #          arrx.append(i)
        #    arrx.sort()
        #    if xc<arrx[0]:
        #        vertices[ti][0] = arrx[0]
        #    elif xc>arrx[len(arrx)-1]:
        #        vertices[ti][0] =arrx[len(arrx)-1]
#
        #    for j in range(979):
        #        pix = img.getpixel((xc, j))
        #        if pix!=(0,0,0):
        #          arry.append(j)
        #    arry.sort()
        #    if yc<arry[0]:
        #        vertices[ti][1] = arry[0]
        #    elif yc>arry[len(arry)-1]:
        #        vertices[ti][1] =arry[len(arry)-1]
        #fc = 0.75
        #fcN = 0.86
        #mnarr = []
        #mnarrN = []
        #for ti in range(43867):
        #    if vertices[ti][0] >= 500:
        #        if vertices[ti][1]>443:
        #            mnarrN.append([vertices[ti][0], ti])
        #        else:
        #         mnarr.append([vertices[ti][0],ti])
        #mnarr.sort()
        #mnarrN.sort()
        #print("PEEL")
        #print(len(mnarr))
        #shifmag = mnarr[0][0]-fc*mnarr[0][0]
        #shifmagN = mnarrN[0][0] - fcN * mnarrN[0][0]
        #print(shifmag)
        #for ti in range(43867):
        #    if vertices[ti][0] >= 500:
        #       if vertices[ti][1]>443:
        #           vertices[ti][0] = vertices[ti][0] * fcN
        #           vertices[ti][0] = vertices[ti][0] + shifmagN
        #       else:
        #        vertices[ti][0] = vertices[ti][0]*fc
        #        vertices[ti][0]  =  vertices[ti][0] + shifmag

        #for ti in range(43867):
        #    vertices[ti][0] =  vertices[ti][0]+16
        ind = vertices.astype(np.int32)



        image2 = imread(r"F:\PRESERVE_SPACE_PROJECTS\face_parsing\RES/frame1.jpg")
        image2 = image2/255
        #file = open("F:/PRESERVE_SPACE_PROJECTS/3DDFA/OUT/RESULT/INDIC.txt", 'w+')
        #arr_REP = []
        #for ti in range(43867):
        #    arr_REP.append([ind[ti][0], ind[ti][1]])
        #    file.write(str(ind[ti][0]) + str(",") + str(ind[ti][1]) + "," + str(ind[ti][2]) + '\n')
        #arr_REP = np.array(arr_REP)
        #for ti in range(43867):
        #    # print(arr_REP[ti])
        #    # print([ind[0][ti], ind[1][ti]])
        #    # print(arr_REP[ti,0])
        #    cou = np.where(abs(arr_REP[:, 0] - [ind[ti][0]])<2)
        #    cou2 = np.where(abs(arr_REP[:, 1] - [ind[ti][1]])<2)
#
        #    iner = np.intersect1d(cou, cou2)
        #    if len(iner) > 1:
        #        # print(len(iner))
        #        iner = list(iner)
        #        # print(iner)
        #        #print(ind.shape)
        #        #print(iner)
        #        #print(ind[iner,2])
#
        #        arrmax = ind[iner,2]
        #        # print(arrmax)
        #        # for ji in iner:
        #        # arrmax.append(ind[2][ji])
        #        res = []
        #        [res.append(x) for x in arrmax if x not in res]
        #        res.sort()
        #        big = res[len(res) - 1]
        #        if abs(big - ind[ti][2]) > 6:
        #            # if ind[0][ti] == 278 and ind[1][ti] == 107:
#
        #            ind[ti][0] = 0
        #            ind[ti][1] = 0

#EXPERIMENT########
        #file = open("F:/PRESERVE_SPACE_PROJECTS/3DDFA/OUT/RESULT/INDIC.txt", 'w+')
        #arr_REP = []
        #for ti in range(43867):
        #   arr_REP.append([ind[ti][0], ind[ti][1]])
        #   file.write(str(ind[ti][0]) + str(",") + str(ind[ti][1]) + "," + str(ind[ti][2]) + '\n')
        #arr_REP = np.array(arr_REP)
        #for ti in range(43867):
        #   # print(arr_REP[ti])
        #   # print([ind[0][ti], ind[1][ti]])
        #   # print(arr_REP[ti,0])
        #   cou = np.where(abs(arr_REP[:, 0] - [ind[ti][0]])<2)
        #   cou2 = np.where(abs(arr_REP[:, 1] - [ind[ti][1]])<2)
#
        #   iner = np.intersect1d(cou, cou2)
        #   if len(iner) > 1:
        #       # print(len(iner))
        #       iner = list(iner)
        #       # print(iner)
        #       #print(ind.shape)
        #       #print(iner)
        #       #print(ind[iner,2])
#
        #       arrmax = ind[iner,2]
        #       # print(arrmax)
        #       # for ji in iner:
        #       # arrmax.append(ind[2][ji])
        #       res = []
        #       [res.append(x) for x in arrmax if x not in res]
        #       res.sort()
        #       big = res[len(res) - 1]
        #       if abs(big - ind[ti][2]) > 5:
        #           # if ind[0][ti] == 278 and ind[1][ti] == 107:
#
        #           ind[ti][0] = 0
        #           ind[ti][1] = 0
        #dicMAX = {}
        #img = Image.open(r"F:\PRESERVE_SPACE_PROJECTS\face_parsing\RES\VIDS\EROSION2/frame120.png")
        #for j in range(284,499):
        #    max = -10000
        #    for i in range(379,626):
        #        pix = img.getpixel((i,j))
        #        if pix!=(0,0,0):
        #            if i>max:
        #                max = i
        #    dicMAX[j] = max
#
        #for i in range(len(ind)):
        #   if  ind[i][1] in dicMAX:
        #    if ind[i][0]> dicMAX[ind[i][1]]:
        #        ind[i][0] = dicMAX[ind[i][1]]
        #for i in range(len(ind)):
        #    ind[i][0] = int(ind[i][0] *0.965)
        colors = image[ind[:,1], ind[:,0], :] # n x 3
        #rpfile = open("F:/PRESERVE_SPACE_PROJECTS/3DDFA/OUT/EXPI2/rpfile.txt", 'w+')
        #rpArr = []
        #for i in range(len(ind)):
        #    if rpArr.count([ind[i][0], ind[i][1]])>1:
        #        rpfile.write(str(ind[i][0]) + " " + str(ind[i][1]) + "\n")
        #        colors[i][0] = 0
        #        colors[i][1] = 255
        #        colors[i][2] = 0
        #    else:
        #        rpArr.append([ind[i][0], ind[i][1]])
#
#
        #print(colors.shape)

        #for i in range(43867):
        #    x = vertices[i][0]
        #    y = vertices[i][1]
        #    cv2.circle(image2, (int(x), int(y)), radius=1, color=(225, 0, 100), thickness=1)
        #cv2.imwrite("F:/PRESERVE_SPACE_PROJECTS/3DDFA/OUT/EXPI2/New_folder/ROUND.png",image2)
        #print(str(ind[:,1]) +"\n")
        #print(str(ind[:,0]) )
        #cols = colors*255
        #cv2.imwrite("C:/Users/Games/Pictures/Video_Projects/2_OUT_OBJ/2/col.png",cols)
        #print(colors)
        return colors








