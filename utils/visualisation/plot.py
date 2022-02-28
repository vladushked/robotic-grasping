from cmath import cos, sin
import warnings
from datetime import datetime
from xml.etree.ElementTree import PI

import cv2

import matplotlib.pyplot as plt
import numpy as np

from utils.dataset_processing.grasp import detect_grasps

warnings.filterwarnings("ignore")


def plot_results(
        fig,
        rgb_img,
        grasp_q_img,
        grasp_angle_img,
        depth_img=None,
        no_grasps=1,
        grasp_width_img=None,
        mask_img=None
):
    """
    Plot the output of a network
    :param fig: Figure to plot the output
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps, mask=mask_img)

    plt.ion()
    plt.clf()
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(rgb_img)
    ax.set_title('RGB')
    ax.axis('off')
    font = {'family': 'cursive',
        'color':  'white',
        'weight': 'bold',
        'size': 12,
        }

    if depth_img is not None:
        ax = fig.add_subplot(2, 3, 2)
        ax.imshow(depth_img, cmap='jet')
        ax.set_title('Depth')
        ax.axis('off')

    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(rgb_img)
    for i, g in enumerate(gs):
        g.plot(ax)
        ax.text(g.center[1], g.center[0], str(grasp_q_img[g.center[0]][g.center[1]]), fontdict = font)

    ax.set_title('Grasp')
    ax.axis('off')

    ax = fig.add_subplot(2, 3, 4)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 3, 5)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 3, 6)
    plot = ax.imshow(grasp_width_img, cmap='jet', vmin=0, vmax=100)
    ax.set_title('Width')
    ax.axis('off')
    plt.colorbar(plot)

    plt.pause(0.1)
    fig.canvas.draw()


def plot_grasp(
        fig,
        grasps=None,
        save=False,
        rgb_img=None,
        grasp_q_img=None,
        grasp_angle_img=None,
        no_grasps=1,
        grasp_width_img=None,
        mask_img=None,
):
    """
    Plot the output grasp of a network
    :param fig: Figure to plot the output
    :param grasps: grasp pose(s)
    :param save: Bool for saving the plot
    :param rgb_img: RGB Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    if grasps is None:
        grasps = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps, mask=mask_img)

    plt.ion()
    plt.clf()

    font = {'family': 'cursive',
        'color':  'white',
        'weight': 'bold',
        'size': 12,
        }
    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    for g in grasps:
        g.plot(ax)
        ax.text(g.center[1], g.center[0], str(grasp_q_img[g.center[0]][g.center[1]]), fontdict = font)
        
    ax.set_title('Grasp')
    ax.axis('off')

    plt.pause(0.1)
    fig.canvas.draw()

    if save:
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.savefig('results/{}.png'.format(time))

    # # img = np.rollaxis(rgb_img,0,3)
    # img = np.copy(rgb_img)

    # for g in grasps:
    #     y1_length = int(g.center[0] + g.width * np.cos(g.angle));
    #     x1_length =  int(g.center[1] + g.width * np.sin(g.angle));
    #     y2_length = int(g.center[0] - g.width * np.cos(g.angle));
    #     x2_length =  int(g.center[1] - g.width * np.sin(g.angle));
    #     color = list(np.random.random(size=3) * 256)
    #     cv2.line(img, (x1_length, y1_length), (x2_length, y2_length), color, 1)
    #     cv2.line(grasp_q_img, (x1_length, y1_length), (x2_length, y2_length), color, 1)
    #     cv2.line(grasp_angle_img, (x1_length, y1_length), (x2_length, y2_length), color, 1)
    #     cv2.line(grasp_width_img, (x1_length, y1_length), (x2_length, y2_length), color, 1)
    # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('RealSense', np.hstack((grasp_q_img, grasp_angle_img, grasp_width_img)))
    # cv2.imshow("RGB", img)
    # cv2.waitKey(1)



def save_results(rgb_img, grasp_q_img, grasp_angle_img, depth_img=None, no_grasps=1, grasp_width_img=None):
    """
    Plot the output of a network
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    ax.set_title('RGB')
    ax.axis('off')
    fig.savefig('results/rgb.png')

    if depth_img.any():
        fig = plt.figure(figsize=(10, 10))
        plt.ion()
        plt.clf()
        ax = plt.subplot(111)
        ax.imshow(depth_img, cmap='gray')
        for g in gs:
            g.plot(ax)
        ax.set_title('Depth')
        ax.axis('off')
        fig.savefig('results/depth.png')

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
    ax.set_title('Grasp')
    ax.axis('off')
    fig.savefig('results/grasp.png')

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)
    fig.savefig('results/quality.png')

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)
    fig.savefig('results/angle.png')

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_width_img, cmap='jet', vmin=0, vmax=100)
    ax.set_title('Width')
    ax.axis('off')
    plt.colorbar(plot)
    fig.savefig('results/width.png')

    fig.canvas.draw()
    plt.close(fig)
