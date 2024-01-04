import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast
import sys
from PRNet.utils.render import vis_of_vertices, render_texture
from PRNet.api import PRN

np.set_printoptions(threshold=sys.maxsize)
from PRNet.utils.estimate_pose import estimate_pose
from PRNet.utils.rotate_vertices import frontalize
from PRNet.utils.render_app import get_visibility, get_uv_mask, get_depth_image
from PRNet.utils.write import write_obj_with_colors, write_obj_with_texture


def main(args):
    if args.isShow or args.isTexture:
        import cv2
        from utils.cv_plot import plot_kpt, plot_vertices, plot_pose_box

    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # GPU number, -1 for CPU
    prn = PRN(is_dlib=args.isDlib)

    # ------------- load data
    image_folder = args.inputDir
    save_folder = args.outputDir
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    types = ('*.jpg', '*.png')
    image_path_list = []
    for files in types:
        image_path_list.extend(glob(os.path.join(image_folder, files)))
    total_num = len(image_path_list)

    for i, image_path in enumerate(image_path_list):

        name = image_path.strip().split('/')[-1][:-4]

        # read image
        image = imread(image_path)
        [h, w, c] = image.shape
        if c > 3:
            image = image[:, :, :3]

        # the core: regress position map
        if args.isDlib:
            max_size = max(image.shape[0], image.shape[1])
            if max_size > 1000:
                image = rescale(image, 1000. / max_size)
                image = (image * 255).astype(np.uint8)
            pos = prn.process(image)  # use dlib to detect face
        else:
            # if image.shape[0] == image.shape[1]:
            #    image = resize(image, (256,256))
            #    pos = prn.net_forward(image/255.) # input image has been cropped to 256x256
            # else:
            # print("HERE")
            #                arr = [[19.0 ,95.0 ,-39.025936],
            # [20.0 ,109.0, -35.41957],
            # [24.0 ,121.0, -32.328506],
            # [26.0 ,132.0, -28.138275],
            # [31.0 ,142.0, -19.92508],
            # [39.0 ,151.0, -7.994181],
            # [47.0 ,154.0, 5.0580955],
            # [59.0 ,156.0, 16.357195],
            # [74.0 ,158.0, 20.56169],
            # [90.0 ,156.0, 13.527635],
            # [100.0, 153.0 ,0.09313274],
            # [106.0, 149.0 ,-14.3995495],
            # [113.0, 141.0 ,-27.414913],
            # [118.0, 129.0 ,-36.2947],
            # [120.0, 117.0 ,-40.945812],
            # [122.0, 105.0 ,-44.62332],
            # [122.0, 94.0, -48.588436],
            # [31.0 ,72.0 ,-1.9319059],
            # [37.0 ,63.0 ,3.3754504],
            # [44.0 ,60.0 ,7.2022867],
            # [52.0 ,60.0 ,9.754878],
            # [59.0 ,62.0 ,11.068278],
            # [85.0 ,62.0 ,9.271925],
            # [91.0 ,60.0 ,6.869508],
            # [98.0 ,58.0 ,3.0022101],
            # [106.0, 62.0, -2.2292027],
            # [111.0, 70.0, -8.789739],
            # [73.0 ,77.0 ,13.65015],
            # [73.0 ,83.0 ,20.38289],
            # [74.0 ,89.0 ,27.992535],
            # [74.0 ,95.0 ,30.636787],
            # [64.0 ,107.0, 20.719645],
            # [69.0 ,107.0, 22.886301],
            # [74.0 ,107.0, 23.89627],
            # [79.0 ,107.0, 22.18057],
            # [83.0 ,107.0, 19.492493],
            # [41.0 ,82.0 ,4.371751],
            # [46.0 ,78.0 ,8.347733],
            # [52.0 ,78.0 ,8.0301],
            # [58.0 ,82.0 ,6.446164],
            # [52.0 ,83.0 ,8.756745],
            # [46.0 ,83.0 ,8.050654],
            # [86.0 ,80.0 ,4.5573378],
            # [93.0 ,78.0 ,5.1179156],
            # [98.0 ,78.0 ,4.3574305],
            # [103.0, 82.0, -0.61175007],
            # [98.0 ,82.0 ,3.8944302],
            # [91.0 ,82.0 ,5.803936],
            # [56.0 ,127.0, 20.467278],
            # [61.0 ,121.0, 25.170424],
            # [69.0 ,117.0, 27.328955],
            # [74.0 ,117.0, 27.755314],
            # [78.0 ,117.0, 26.912157],
            # [86.0 ,121.0, 23.841146],
            # [93.0 ,127.0, 18.26969],
            # [86.0 ,129.0, 24.90198],
            # [81.0 ,131.0, 28.155813],
            # [74.0 ,131.0, 29.188078],
            # [69.0 ,131.0, 28.810225],
            # [63.0 ,129.0, 26.22091],
            # [58.0 ,127.0, 20.463808],
            # [69.0 ,122.0, 25.526966],
            # [74.0 ,122.0, 26.306004],
            # [79.0 ,122.0, 25.049593],
            # [91.0 ,127.0, 18.524767],
            # [79.0 ,124.0, 26.00097],
            # [74.0 ,124.0, 26.747955],
            # [69.0 ,124.0, 26.4695]]
            #                kpts = np.zeros([3, 68], dtype=float)
            #                for q in range(68):
            #                    kpts[0][q] = arr[q][0]
            #                    kpts[1][q] = arr[q][1]
            #                    kpts[2][q] = arr[q][2]

            # inter_index = image_path.index('Person')
            # print(inter_index)
            inter_index = image_path.index('\\')
            # inter_index2 = image_path.index('f')
            # inter_index2 = image_path.index('_')
            # print(inter_index)
            # print(inter_index2)
            # indO = inter_index+1
            # indE = inter_index2-1
            # print("G:/FACE_VIDEO/combFOLDER/COMBINED_PADKEYPTS/" + image_path[indO:indE] +"_" + image_path[inter_index2:-4] + ".txt")
            # print(image_path)
            # file1 = open("C:/Users/Games/Pictures/Video_Projects/KS3/frame0.png.txt" ,"r")
            #print("C:/Users/Games/Pictures/Video_Projects/1_KEYS/" + image_path[inter_index + 1:-4] + ".txt")
            file1 = open(r"F:\PRESERVE_SPACE_PROJECTS\face_parsing\RES\VIDS\vid37G\37K/" +  image_path[inter_index+1:-4] + ".txt" ,"r")
            # new_str = image_path[inter_index:]
            # print(new_str)
            # indU = new_str.index("_")
            # file1 = open("F:/ALL_PADDED_REFINED/ISHAN_COMBINED_KEYPOINTS/" + new_str[1:indU] + "/" + new_str[indU+1:-4] + ".png.txt","r")
            # print("H:/KARTIK_ANNU_ALL_FOLDERS/NEW_KEYPOINTS/" + image_path[indO:indE] + "/" + image_path[inter_index2:-4] + ".png.txt")
            # file1 = open("H:/VEDANT_KEYPOINTS/" + image_path[inter_index+1:] + ".txt")
            # file1 = open("G:/FACE_VIDEO/combFOLDER/COMBINED_PADKEYPTS/CINNAMON_frame56.txt","r")
            kpts = np.zeros([3, 68], dtype=float)

            for i in range(68):
                t = file1.readline()
                t = t.strip()
                # print(t)
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
            box = np.array([0, image.shape[1] - 1, 0, image.shape[0] - 1])  # cropped with bounding box
            pos = prn.process(image, kpts)

        image = image / 255.
        if pos is None:
            continue

        if args.is3d or args.isMat or args.isPose or args.isShow:
            # 3D vertices
            vertices = prn.get_vertices(pos)
            if args.isFront:
                save_vertices = frontalize(vertices)
            else:
                save_vertices = vertices.copy()
            save_vertices[:, 1] = h - 1 - save_vertices[:, 1]

        if args.isImage:
            imsave(os.path.join(save_folder, name + '.jpg'), image)

        if args.is3d:
            # corresponding colors
            colors = prn.get_colors(image, vertices)

            if args.isTexture:
                if args.texture_size != 256:
                    pos_interpolated = resize(pos, (args.texture_size, args.texture_size), preserve_range=True)
                else:
                    pos_interpolated = pos.copy()
                texture = cv2.remap(image, pos_interpolated[:, :, :2].astype(np.float32), None,
                                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
                if args.isMask:
                    vertices_vis = get_visibility(vertices, prn.triangles, h, w)
                    print(vertices)
                    uv_mask = get_uv_mask(vertices_vis, prn.triangles, prn.uv_coords, h, w, prn.resolution_op)
                    uv_mask = resize(uv_mask, (args.texture_size, args.texture_size), preserve_range=True)
                    texture = texture * uv_mask[:, :, np.newaxis]
                # print(colors)
                write_obj_with_texture(os.path.join(save_folder, name + '.obj'), save_vertices, prn.triangles, texture,
                                       prn.uv_coords / prn.resolution_op)  # save 3d face with texture(can open with meshlab)
            else:
                # save_vertices[:, 1] = h - 1 - save_vertices[:, 1]

                write_obj_with_colors(os.path.join(save_folder, name + '.obj'), save_vertices, prn.triangles,
                                      colors)  # save 3d face(can open with meshlab)

        if args.isDepth:
            depth_image = get_depth_image(vertices, prn.triangles, h, w, True)
            depth = get_depth_image(vertices, prn.triangles, h, w)
            imsave(os.path.join(save_folder, name + '_depth.jpg'), depth_image)
            sio.savemat(os.path.join(save_folder, name + '_depth.mat'), {'depth': depth})

        if args.isMat:
            colors = prn.get_colors(image, vertices)
            sio.savemat(os.path.join(save_folder, name + '_mesh.mat'),
                        {'vertices': vertices, 'colors': colors, 'triangles': prn.triangles})

        if args.isKpt or args.isShow:
            # get landmarks
            kpt = prn.get_landmarks(pos)
            np.savetxt(os.path.join(save_folder, name + '_kpt.txt'), kpt)

        if args.isPose or args.isShow:
            # estimate pose
            camera_matrix, pose = estimate_pose(vertices)
            np.savetxt(os.path.join(save_folder, name + '_pose.txt'), pose)
            np.savetxt(os.path.join(save_folder, name + '_camera_matrix.txt'), camera_matrix)

            np.savetxt(os.path.join(save_folder, name + '_pose.txt'), pose)

        if args.isShow:
            # ---------- Plot
            # image_pose = plot_pose_box(image, camera_matrix, kpt)

            cv2.imshow('sparse alignment', plot_kpt(image, kpt))
            cv2.imshow('dense alignment', plot_vertices(image, vertices))
            cv2.imwrite("C:/Users/Games/PycharmProjects/PRNet/OUTPUT/img2/write.png", plot_vertices(image, vertices))
            cv2.imshow('pose', plot_pose_box(image, camera_matrix, kpt, nam=image_path[-10:-4]))
            cv2.imwrite(
                "C:/Users/Games/PycharmProjects/DepthNets/depthnetpytorch/input_facewarper/rotation_example5/expected_result/post_alpha_removal/with_boxs/" + image_path[
                                                                                                                                                              -10:-4] + ".png",
                plot_pose_box(image, camera_matrix, kpt, nam=image_path[-10:-4]))
            cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

    parser.add_argument('-i', '--inputDir', default='C:/Users/Games/Pictures/Video_Projects/1/', type=str,
                        help='path to the input directory, where input images are stored.')
    parser.add_argument('-o', '--outputDir', default='C:/Users/Games/Pictures/Video_Projects/1_PROUT', type=str,
                        help='path to the output directory, where results(obj,txt files) will be stored.')
    parser.add_argument('--gpu', default='0', type=str,
                        help='set gpu id, -1 for CPU')
    parser.add_argument('--isDlib', default=False, type=ast.literal_eval,
                        help='whether to use dlib for detecting face, default is True, if False, the input image should be cropped in advance')
    parser.add_argument('--is3d', default=False, type=ast.literal_eval,
                        help='whether to output 3D face(.obj). default save colors.')
    parser.add_argument('--isMat', default=True, type=ast.literal_eval,
                        help='whether to save vertices,color,triangles as mat for matlab showing')
    parser.add_argument('--isKpt', default=False, type=ast.literal_eval,
                        help='whether to output key points(.txt)')
    parser.add_argument('--isPose', default=False, type=ast.literal_eval,
                        help='whether to output estimated pose(.txt)')
    parser.add_argument('--isShow', default=False, type=ast.literal_eval,
                        help='whether to show the results with opencv(need opencv)')
    parser.add_argument('--isImage', default=False, type=ast.literal_eval,
                        help='whether to save input image')
    # update in 2017/4/10
    parser.add_argument('--isFront', default=False, type=ast.literal_eval,
                        help='whether to frontalize vertices(mesh)')
    # update in 2017/4/25
    parser.add_argument('--isDepth', default=False, type=ast.literal_eval,
                        help='whether to output depth image')
    # update in 2017/4/27
    parser.add_argument('--isTexture', default=False, type=ast.literal_eval,
                        help='whether to save texture in obj file')
    parser.add_argument('--isMask', default=False, type=ast.literal_eval,
                        help='whether to set invisible pixels(due to self-occlusion) in texture as 0')
    # update in 2017/7/19
    parser.add_argument('--texture_size', default=256, type=int,
                        help='size of texture map, default is 256. need isTexture is True')
    main(parser.parse_args())
