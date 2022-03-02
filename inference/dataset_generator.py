import os
import glob
import time
import cv2
from cv2 import COLORMAP_JET
import imutils
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from hardware.camera import RealSenseCamera
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.dataset_processing.grasp import detect_grasps
from utils.visualisation.plot import plot_grasp, plot_results

from RAS_Com import RAS_Connect
import json

class DatasetGenerator:
    def __init__(self, saved_model_path, dataset_dir, material_dir):
        self.saved_model_path = saved_model_path
        self.dataset_dir = dataset_dir
        self.material_dir = material_dir
        self.save_path = os.path.join(dataset_dir, material_dir)

        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        
        self.i = 0

        already_saved_rgbs = glob.glob(os.path.join(self.save_path, "pcd????r.png"))
        if (len(already_saved_rgbs)) > 0:
            already_saved_rgbs.sort()
            self.i = int(os.path.split(already_saved_rgbs[-1])[-1].split("pcd")[-1].split("r.png")[0])
        print("Last: ", self.i)


        self.width = 640
        self.height = 480
        self.output_size = 300
        self.delta = [(self.width - self.output_size) // 2, (self.height - self.output_size) // 2]

        self.grip_height = 0.5

        self.camera = RealSenseCamera(width=self.width,
                                      height=self.height,
                                      fps=30)

        self.saved_model_path = saved_model_path
        self.model = None
        self.device = None

        self.cam_data = CameraData(width=self.width,
                                   height=self.height,
                                   output_size=self.output_size,
                                   output_width=self.output_size,
                                   output_height=self.output_size,
                                   include_depth=True,
                                   include_rgb=True,
                                   )

        # Connect to camera
        self.camera.connect()

        # Load camera pose and depth scale (from running calibration)
        self.cam_pose = np.loadtxt('saved_data/camera_pose.txt', delimiter=' ')
        self.cam_depth_scale = np.loadtxt(
            'saved_data/camera_depth_scale.txt', delimiter=' ')

        # homedir = os.path.join(os.path.expanduser('~'), "grasp-comms")
        # self.grasp_request = os.path.join(homedir, "grasp_request.npy")
        # self.grasp_available = os.path.join(homedir, "grasp_available.npy")
        # self.grasp_pose = os.path.join(homedir, "grasp_pose.npy")

        self.fig = plt.figure(figsize=(10, 10))

        
        self.init_rgb_img = None
        while True:
            image_bundle = self.camera.get_image_bundle()
            background = self.cam_data.get_rgb(image_bundle['rgb'], False)
            # self.init_rgb_img = cv2.cvtColor(self.init_rgb_img, cv2.COLOR_RGB2BGR)
            self.init_rgb_img = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
            cv2.imshow('camera', self.init_rgb_img)
            keyboard = cv2.waitKey(30)
            if keyboard == 'q' or keyboard == 27:
                break
        
        if not os.path.isfile(os.path.join(self.save_path, "background.png")):
            # saving rgb .png
            print("Saving background ...")
            cv2.imwrite(os.path.join(self.save_path, "background.png"), cv2.cvtColor(image_bundle['rgb'], cv2.COLOR_RGB2BGR))

    def load_model(self):
        print('Loading model... ')
        self.model = torch.load(self.saved_model_path)
        # Get the compute device
        self.device = get_device(force_cpu=False)

    def generate(self):
        self.i += 1

        while True:
            # Get RGB-D image from camera
            image_bundle = self.camera.get_image_bundle()
            rgb = image_bundle['rgb']
            depth = image_bundle['aligned_depth']

            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            # rgb_to_save = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb_to_save = rgb.copy()
            depth_to_save = depth.copy()
            print("Max depth", depth_to_save.max())
            

            rgb_img=self.cam_data.get_rgb(rgb, False)
            depth_img=np.squeeze(self.cam_data.get_depth(depth))
            gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            # cv2.imshow('camera', gray)

            x, _, _ = self.cam_data.get_data(rgb=rgb, depth=depth)

            # Predict the grasp pose using the saved model
            with torch.no_grad():
                xc = x.to(self.device)
                pred = self.model.predict(xc)

            q_img, ang_img, width_img = post_process_output(
                pred['pos'], pred['cos'], pred['sin'], pred['width'])

            fgMask = cv2.absdiff(gray, self.init_rgb_img)
            # fgMask = cv2.cvtColor(fgMask, cv2.COLOR_RGB2GRAY)
            # Otsu's thresholding after Gaussian filtering
            # mask = cv2.adaptiveThreshold(fgMask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
            blur = cv2.GaussianBlur(fgMask,(9,9),0)

            ret3,mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # blur = cv2.GaussianBlur(mask,(9,9),0)
            kernel = np.ones((9,9),np.uint8)
            # mask_img = cv2.dilate(mask,kernel,iterations = 1)
            mask_img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            cnts = cv2.findContours(mask_img.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            if len(cnts) > 0:
                contour = max(cnts, key=cv2.contourArea)
                
                # draw the contours of c
                cv2.drawContours(rgb_img, [contour], -1, (0, 0, 255), 2)

                x,y,w,h = cv2.boundingRect(contour)
                cv2.rectangle(rgb_img,(x,y),(x+w,y+h),(0,255,0),2)
                bbox = np.asarray([x,y,x+w,y+h]).reshape(2,2)

                rect = cv2.minAreaRect(contour)
                minbox = cv2.boxPoints(rect)
                minbox = np.int0(minbox)
                cv2.drawContours(rgb_img,[minbox],0,(255,0,0),2, cv2.FILLED)
                
                rows,cols = rgb_img.shape[:2]
                [vx,vy,x,y] = cv2.fitLine(contour, cv2.DIST_L2,0,0.01,0.01)
                meanline = np.asarray([vx,vy,x + self.delta[0],y + self.delta[1]])
                
                lefty = int((-x*vy/vx) + y)
                righty = int(((cols-x)*vy/vx)+y)
                cv2.line(rgb_img,(cols-1,righty),(0,lefty),(255,255,0),2)


            res = np.hstack((gray, fgMask, blur, mask, mask_img))
            # nn_res = np.hstack((q_img, ang_img, width_img))

            # cv2.imshow('Result', res)
            # cv2.imshow('q_img', q_img)
            # cv2.imshow('ang_img', ang_img)
            # cv2.imshow('width_img', width_img)
            
            grasps = detect_grasps(q_img, ang_img, width_img, mask_img, no_grasps=10)
            grasps_list = []
            for grasp in grasps:
                grasp = grasp.as_gr.points
                grasp[:,[0, 1]] = grasp[:,[1, 0]]
                grasps_list.append(grasp)
                color = list(np.random.random(size=3) * 256)
                cv2.drawContours(rgb_img, [grasp.astype(int)], 0, color=color, thickness=4)

            grasps_array = np.asarray(grasps_list)
            cv2.imshow('camera', rgb_img)
            
            cv2.imshow('depth', depth_to_save)
            # cv2.imshow('depth_img', depth_img)
            # cv2.imshow('depth', cv2.applyColorMap(depth, COLORMAP_JET))
            # cv2.imshow('depth_img', cv2.applyColorMap(depth_img, COLORMAP_JET))
            
            # if self.fig:
            #     plot_grasp(fig=self.fig, rgb_img=rgb_img,grasp_q_img=q_img, grasps=grasps, no_grasps=10, mask_img=mask_img)
            

            keyboard = cv2.waitKey(1000)
            if keyboard == 'q' or keyboard == 27:
                break        

        print("Saving: %04d" % self.i)

        # saving positive grasps
        cposname = "pcd%04dcpos.txt" % self.i
        # print("self.delta", self.delta)
        # print("grasps_array", grasps_array)
        grasps_array = grasps_array.reshape(-1, 2) + self.delta
        np.savetxt(os.path.join(self.save_path, cposname), grasps_array, fmt="%f")

        # saving depth .tiff
        dname = "pcd%04dd.tiff" % self.i
        cv2.imwrite(os.path.join(self.save_path, dname), depth_to_save)


        # saving rgb .png
        rname = "pcd%04dr.png" % self.i
        cv2.imwrite(os.path.join(self.save_path, rname), rgb_to_save)

        # saving contour
        contname = "pcd%04dcont.txt" % self.i
        contour = np.squeeze(contour) + self.delta
        np.savetxt(os.path.join(self.save_path, contname), contour, fmt="%d")

        # saving bbox
        bboxname = "pcd%04dbbox.txt" % self.i
        bbox = bbox + self.delta
        np.savetxt(os.path.join(self.save_path, bboxname), bbox, fmt="%d")

        # saving minbox
        minboxname = "pcd%04dminbox.txt" % self.i
        minbox = minbox + self.delta
        np.savetxt(os.path.join(self.save_path, minboxname), minbox, fmt="%d")

        # saving meanline
        meanlinename = "pcd%04dmeanline.txt" % self.i
        np.savetxt(os.path.join(self.save_path, meanlinename), meanline, fmt="%f")

        return

    def run(self):
       while(True):
            self.generate()
