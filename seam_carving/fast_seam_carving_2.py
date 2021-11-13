import re
import numpy as np
import cv2
from numpy.core.fromnumeric import argmax, argmin
import heapq
from tqdm import tqdm

'''
Họ và tên: Lê Quang Huy
MSSV: 19521617
improve seam carving speed

'''

def image_energy(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gX = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0)
    gY = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1)

    g = np.sqrt(gX**2 + gY**2)
    return g

def remove_seam_mask(img: np.ndarray, seam_mask: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        h, w, c = img.shape
        seam_mask = np.dstack([seam_mask] * c)
        ans = img[seam_mask].reshape((h, w - 1, c))

    else: 
        h, w = img.shape  
        ans = img[seam_mask].reshape((h, w - 1))

    return ans

def get_backward_seam(energy: np.ndarray):
    h, w = energy.shape

    cost = energy[0]
    parent = np.empty((h, w), dtype= np.int32)
    base_idx = np.arange(-1, w - 1, dtype= np.int32)

    for r in range(1, h):
        left_shift = np.hstack((cost[1:], np.inf))
        right_shift = np.hstack((np.inf, cost[:-1]))

        min_idx = np.argmin([right_shift, cost, left_shift],\
            axis= 0) + base_idx

        parent[r] = min_idx 
        cost = cost[min_idx] + energy[r]

    c = np.argmin(cost)
    seam = np.empty(h, dtype= np.int32)

    for r in range(h - 1, -1 , -1):
        seam[r] = c 
        c = parent[r, c]

    return seam

def get_seam_mask(gray, seam):
    return ~np.eye(gray.shape[1], dtype= np.bool8)[seam]

def get_backward_seams(gray: np.ndarray, num_seams: int):
    h, w = gray.shape 
    seams_mask = np.zeros((h, w), dtype= np.bool8)
    rows = np.arange(0, h, dtype= np.int32)
    idx_map = np.tile(np.arange(0, w, dtype= np.int32)\
        ).reshape((h, w))
    energy = image_energy(gray)

    for _ in range(num_seams):
        seam = get_backward_seam(energy)
        seams_mask[rows, idx_map[rows, seam]] = True

        seam_mask = get_seam_mask(gray, seam)
