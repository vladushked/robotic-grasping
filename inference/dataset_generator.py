import os
import time
import cv2

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
        self.output_width = 200
        self.output_height = 200
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
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.init_rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            cv2.imshow('Frame', self.init_rgb_img)
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
        img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        fgMask = cv2.absdiff(gray, self.init_rgb_img)
        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(fgMask,(5,5),0)
        ret3,mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((8,8),np.uint8)
        mask_img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        depth_img=np.squeeze(self.cam_data.get_depth(depth))
        res = np.hstack((q_img, ang_img, width_img, mask_img))

        cv2.imshow('Result', res)
        cv2.waitKey(30)
        
        grasps = detect_grasps(q_img, ang_img, width_img, mask_img, no_grasps=10)
        
        if self.fig:
            # plot_grasp(fig=self.fig, rgb_img=self.cam_data.get_rgb(rgb, False), grasps=grasps, grasp_q_img=q_img,
            #            grasp_angle_img=ang_img,
            #            no_grasps=10,
            #            grasp_width_img=width_img)

            plot_results(fig=self.fig,
                         rgb_img=self.cam_data.get_rgb(rgb, False),
                         depth_img=np.squeeze(self.cam_data.get_depth(depth)),
                         grasp_q_img=q_img,
                         grasp_angle_img=ang_img,
                         no_grasps=10,
                         grasp_width_img=width_img,
                         mask_img=mask_img)
        
        
        ## [show]

        

        if len(grasps) == 0:
            return None, None, None

        # Get grasp position from model output
        pos_z = depth[grasps[0].center[0] + self.cam_data.top_left[0],
                      grasps[0].center[1] + self.cam_data.top_left[1]] * self.cam_depth_scale
        pos_x = np.multiply(grasps[0].center[1] + self.cam_data.top_left[1] - self.camera.intrinsics.ppx,
                            pos_z / self.camera.intrinsics.fx)
        pos_y = np.multiply(grasps[0].center[0] + self.cam_data.top_left[0] - self.camera.intrinsics.ppy,
                            pos_z / self.camera.intrinsics.fy)
        print("grasps[0].center[0] + self.cam_data.top_left[0]: ", grasps[0].center[1], self.cam_data.top_left[1])
        print("self.camera.intrinsics.ppx: ", self.camera.intrinsics.ppx)
        print("___CAM POSITION___: ", pos_x, pos_y, pos_z)

        if pos_z == 0:
            return None, None, None

        target = np.asarray([pos_x, pos_y, pos_z])
        target.shape = (3, 1)
        #print('target: ', target)

        # Convert camera to robot coordinates
        camera2robot = self.cam_pose
        target_position = np.dot(
            camera2robot[0:3, 0:3], target) + camera2robot[0:3, 3:]
        
        target_position = target_position[0:3, 0]

        # Convert camera to robot angle
        angle = np.asarray([0, 0, grasps[0].angle])
        angle.shape = (3, 1)
        target_angle = np.dot(camera2robot[0:3, 0:3], angle)

        # Concatenate grasp pose with grasp angle
        grasp_pose = np.append(target_position, target_angle[2])

        # print('grasp_pose: ', grasp_pose)

        

        return grasp_pose, grasps[0].width, grasps[0].length

    def run(self):

        if self.enable_arm:
            while(True):
                print("Resetting position")
                self.s.grip(90)
                time.sleep(2)
                self.s.effectorMovement(-20, 200, 200, 0)
                time.sleep(2)
                tool_position, grasp_width, grasp_length = self.generate()
                if tool_position is None:
                    continue
                
                x = tool_position[0]
                y = tool_position[1]
                z_init = tool_position[2]
                z = tool_position[2] * 0.5
                angle = tool_position[3] * 100
                if tool_position[2] > self.grip_height:
                    z = tool_position[2] - self.grip_height * 0.5

                print("___POSITION___: ", tool_position)
                print("___ANGLE___: ", angle)
                print("___Z___: ", z * 1000)
                print("___LENGTH___", grasp_length)
                print("___WIDTH___", grasp_width)

                if self.conveyor_speed is None:
                    self.s.effectorMovement(x * 1000, y * 1000, z_init * 1000, - angle * 0.5 * 0.62)
                    time.sleep(2)
                    self.s.effectorMovement(x * 1000, y * 1000, z * 1000, - angle * 0.5 * 0.62)
                else:
                    self.s.effectorMovement(0, y * 1000, z_init * 1000, - angle * 0.5 * 0.62)
                    time_to_sleep = x / self.conveyor_speed
                    time.sleep(time_to_sleep)
                    print("___TIME TO SLEEP___", time_to_sleep)
                    self.s.effectorMovement(0, y * 1000, z * 1000, - angle * 0.5 * 0.62)
                time.sleep(0.1)
                self.s.grip(0)
                time.sleep(0.5)
                self.s.effectorMovement(x * 1000, y * 1000, 300, - angle * 0.5 * 0.62)
                time.sleep(1)
                self.s.effectorMovement(-200, 200, 200, 0)
                time.sleep(2)

        else:
            while(True):
                tool_position = self.generate()
                time.sleep(1)
