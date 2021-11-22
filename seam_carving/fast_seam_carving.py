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
basic idea: try to delete at most 3 seam line at the same time
'''

def image_energy(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gX = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0)
    gY = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1)

    g = np.sqrt(gX**2 + gY**2)
    return g


def find_minimum_seam(img):
    # dynamic programing
    r, c, _ = img.shape
    energy_map = image_energy(img)

    T = energy_map.copy()
    back_track = np.zeros_like(T, dtype=np.int32)

    for i in range(1, r):
        for j in range(c):
            if j == 0:
                idx = argmin(T[i - 1, j:j+2])
                idx = j + idx
                min_energy = T[i - 1, idx]
                back_track[i, j] = idx
            else:
                idx = argmin(T[i - 1, j - 1: j + 2])
                idx = idx + j - 1
                min_energy = T[i - 1, idx]
                back_track[i, j] = idx

            T[i, j] += min_energy

    return T, back_track


def draw_minimum_seams(img, num_seams):
    r, c, _ = img.shape

    # get T dp and back_track
    T, back_track = find_minimum_seam(img)

    list_val_index = [(T[-1, i], i) for i in range(c)]

    list_val_index.sort()
    count_seams = 0

    # create mask
    mask = np.ones((r, c), dtype=np.bool8)

    # print(mask.shape)
    index = 0
    while len(list_val_index) > index and count_seams < num_seams:
        temp = list_val_index[index]
        index = (index + 100)%c
        j = temp[1]

        flag = True
        temp_mask = mask.copy()
        for i in reversed(range(r)):
            if temp_mask[i, j] == True:
                temp_mask[i, j] = False
                j = back_track[i, j]
            else:
                best_index = j
                maxX = 10**9
                for idx in range(j - 2, j):
                    if T[i, idx] < maxX and temp_mask[i, idx]:
                        best_index = idx
                        maxX = T[i, idx]
                
                for idx in range(j + 1, min(c, j + 3)):
                    if T[i, idx] < maxX and temp_mask[i, idx]:
                        best_index = idx
                        maxX = T[i, idx]
                    
                if best_index != j:
                    j = best_index
                    temp_mask[i, j] = False
                    j = back_track[i, j]
                else:
                    flag = False
                    break

        if flag:
            count_seams += 1
            mask = temp_mask.copy()


    mask = np.stack([mask] * 3, axis=2)
    # img = img[mask].reshape((r, c - count_seams, 3))
    # img[mask == False] = 250

    return mask, count_seams

def crop_col(img, scale_c):
    r, c, _ = img.shape
    remove_c = c - int(scale_c * c)

    sum_of_seams = 0
    # print(remove_c)
    pbar = tqdm(total=remove_c)
    while sum_of_seams < remove_c :
        seams = min(3, remove_c - sum_of_seams)
        mask, count_seams = draw_minimum_seams(img, seams)
        sum_of_seams += count_seams
        img = img[mask].reshape((r, c - count_seams, 3))

        c -= count_seams

        pbar.update(count_seams)

    pbar.close()
    return img

img = cv2.imread('cat.jpeg')
r, c, _ = img.shape 

cv2.imshow('img', img)
# 
cv2.waitKey(0)


cropped_img = crop_col(img, 0.8)
cv2.imshow('cropped_img', cropped_img)
print(cropped_img.shape)
cv2.imwrite('fast_seam_carving_cropped.png', cropped_img)
cv2.waitKey(0)
