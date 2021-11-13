import cv2 
import numpy as np
from tqdm import trange

'''
Họ và tên: Lê Quang Huy
MSSV: 19521617
implement seam carving

'''

def calc_energy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    gray = gray.astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    energy = np.sqrt(grad_x**2 + grad_y**2)
    return energy

def show_img(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)

def minimum_seam(img):
    r, c, _ = img.shape
    energy_map = calc_energy(img)

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int32)

    for i in range(1, r):
        for j in range(0, c):
            # Handle the left edge of the image, to ensure we don't index a -1
            if j == 0:
                idx = np.argmin(M[i-1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack

def carve_column(img):
    r, c, _ = img.shape

    M, backtrack = minimum_seam(img)
    mask = np.ones((r, c), dtype=np.bool8)

    j = np.argmin(M[-1])
    for i in reversed(range(r)):
        mask[i, j] = False
        j = backtrack[i, j]

    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, 3))
    return img

def crop_c(img, scale_c):
    r, c, _ = img.shape
    new_c = int(scale_c * c)

    for i in trange(c - new_c):
        img = carve_column(img)

    return img

img = cv2.imread('cat.jpeg')

crop_img = crop_c(img, 0.8)
show_img('img', img)
show_img('crop', crop_img)

print(crop_img.shape)
cv2.imwrite('seam_carving_cropped.png', crop_img)