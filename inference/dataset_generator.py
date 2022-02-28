import os
import time
import cv2
import imutils

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


class DatasetGenerator:
    def __init__(self, saved_model_path, visualize=False, enable_arm=False, include_depth=True,
                                   include_rgb=True, conveyor_speed=None):
        self.saved_model_path = saved_model_path

        self.width = 640
        self.height = 480
        self.output_size = 350
        self.output_width = 250
        self.output_height = 250
        # self.top_left = (150, 200)
        # self.bottom_right = (330, 640)
        self.grip_height = 0.5
        self.conveyor_speed = conveyor_speed

        self.enable_arm = enable_arm

        self.camera = RealSenseCamera(width=self.width,
                                      height=self.height,
                                      fps=30)

        self.saved_model_path = saved_model_path
        self.model = None
        self.device = None

        self.cam_data = CameraData(width=self.width,
                                   height=self.height,
                                   output_size=self.output_size,
                                   output_width=self.output_width,
                                   output_height=self.output_height,
                                   include_depth=include_depth,
                                   include_rgb=include_rgb,
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

        if visualize:
            self.fig = plt.figure(figsize=(10, 10))
        else:
            self.fig = None

        if self.enable_arm:
            self.s = RAS_Connect('/dev/ttyTHS0')
        
        self.init_rgb_img = None
        while True:
            image_bundle = self.camera.get_image_bundle()
            img = self.cam_data.get_rgb(image_bundle['rgb'], False)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.init_rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            cv2.imshow('camera', self.init_rgb_img)
            keyboard = cv2.waitKey(30)
            if keyboard == 'q' or keyboard == 27:
                break

    def load_model(self):
        print('Loading model... ')
        self.model = torch.load(self.saved_model_path)
        # Get the compute device
        self.device = get_device(force_cpu=False)

    def generate(self):
        

        # Get RGB-D image from camera
        image_bundle = self.camera.get_image_bundle()
        rgb = image_bundle['rgb']
        depth = image_bundle['aligned_depth']
        x, _, _ = self.cam_data.get_data(rgb=rgb, depth=depth)
        # print(x.shape)

        # Predict the grasp pose using the saved model
        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.model.predict(xc)

        q_img, ang_img, width_img = post_process_output(
            pred['pos'], pred['cos'], pred['sin'], pred['width'])

        rgb_img=self.cam_data.get_rgb(rgb, False)
        # rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        fgMask = cv2.absdiff(gray, self.init_rgb_img)
        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(fgMask,(9,9),0)
        ret3,mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(mask,(9,9),0)
        kernel = np.ones((9,9),np.uint8)
        mask_img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cnts = cv2.findContours(mask_img.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # print("Contour", c.shape)
        # draw the contours of c
        cv2.drawContours(rgb_img, [c], -1, (0, 0, 255), 2)

        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(rgb_img,(x,y),(x+w,y+h),(0,255,0),2)

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(rgb_img,[box],0,(255,0,0),2)
        
        rows,cols = rgb_img.shape[:2]
        [vx,vy,x,y] = cv2.fitLine(c, cv2.DIST_L2,0,0.01,0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        cv2.line(rgb_img,(cols-1,righty),(0,lefty),(255,255,0),2)


        depth_img=np.squeeze(self.cam_data.get_depth(depth))
        res = np.hstack((gray, mask, blur, mask_img))
        # res = np.hstack((gray, q_img, ang_img, width_img, fgMask, blur, mask, mask_img))

        cv2.imshow('camera', rgb_img)
        cv2.imshow('Result', res)
        cv2.waitKey(30)
        
        grasps = detect_grasps(q_img, ang_img, width_img, mask_img, no_grasps=10)
        
        if self.fig:
            plot_grasp(fig=self.fig, rgb_img=rgb_img,grasp_q_img=q_img, grasps=grasps, no_grasps=10, mask_img=mask_img)

            # plot_results(fig=self.fig,
            #              rgb_img=self.cam_data.get_rgb(rgb, False),
            #              depth_img=np.squeeze(self.cam_data.get_depth(depth)),
            #              grasp_q_img=q_img,
            #              grasp_angle_img=ang_img,
            #              no_grasps=10,
            #              grasp_width_img=width_img,
            #              mask_img=mask_img)
        return

    def run(self):
       while(True):
            self.generate()
