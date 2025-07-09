import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import os

def decode(imprefix,start,threshold):
    """
    Given a sequence of 20 images of a scene showing projected 10 bit gray code, 
    decode the binary sequence into a decimal value in (0,1023) for each pixel.
    Mark those pixels whose code is likely to be incorrect based on the user 
    provided threshold.  Images are assumed to be named "imageprefixN.png" where
    N is a 2 digit index (e.g., "img00.png,img01.png,img02.png...")
 
    Parameters
    ----------
    imprefix : str
       Image name prefix
      
    start : int
       Starting index
       
    threshold : float
       Threshold to determine if a bit is decodeable
       
    Returns
    -------
    code : 2D numpy.array (dtype=float)
        Array the same size as input images with entries in (0..1023)
        
    mask : 2D numpy.array (dtype=logical)
        Array indicating which pixels were correctly decoded based on the threshold
    
    """
    
    # we will assume a 10 bit code
    nbits = 10
    gray_bits = []

    # Load the first image to get shape
    first_img = plt.imread(f"{imprefix}{start:02d}_u.png").astype(float)

    if first_img.ndim == 3:
        first_img = first_img.mean(axis=2)

    height, width = first_img.shape
    reliable = np.ones_like(first_img, dtype=bool)

    for i in range(nbits):
        idx1 = start + 2 * i
        idx2 = idx1 + 1

        img1 = plt.imread(f"{imprefix}{idx1:02d}_u.png").astype(float)
        img2 = plt.imread(f"{imprefix}{idx2:02d}_u.png").astype(float)

        if img1.ndim == 3:
            img1 = img1.mean(axis=2)
            img2 = img2.mean(axis=2)

        bit = (img1 > img2).astype(np.uint8)
        gray_bits.append(bit)

        diff = np.abs(img1 - img2)
        reliable &= (diff > threshold)
        
    gray_code = np.stack(gray_bits, axis=0)
    
    # don't forget to convert images to grayscale / float after loading them in

    binary = np.zeros_like(gray_code)
    binary[0] = gray_code[0]
    for i in range(1, nbits):
        binary[i] = np.bitwise_xor(binary[i - 1], gray_code[i])

    powers = 2 ** np.arange(nbits - 1, -1, -1).reshape((nbits, 1, 1))
    code = np.sum(binary * powers, axis=0).astype(float)

    mask = reliable

    return code, mask


def triangulate(pts2L,camL,pts2R,camR):
    """
    Triangulate the set of points seen at location pts2L / pts2R in the
    corresponding pair of cameras. Return the 3D coordinates relative
    to the global coordinate system


    Parameters
    ----------
    pts2L : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N) seen from camL camera

    pts2R : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N) seen from camR camera

    camL : Camera
        The first "left" camera view

    camR : Camera
        The second "right" camera view

    Returns
    -------
    pts3 : 2D numpy.array (dtype=float)
        (3,N) array containing 3D coordinates of the points in global coordinates

    """
    
    # This line will now work because pts2L will be a 2D array
    npts = pts2L.shape[1]

    # Reshape t vectors to be explicitly (3,1) for safe broadcasting
    tL = camL.t.reshape(3, 1)
    tR = camR.t.reshape(3, 1)

    qL = (pts2L - camL.c.reshape(2, 1)) / camL.f.reshape(2, 1)
    qL = np.vstack((qL,np.ones((1,npts))))

    qR = (pts2R - camR.c.reshape(2, 1)) / camR.f.reshape(2, 1)
    qR = np.vstack((qR,np.ones((1,npts))))

    xL = np.zeros((3,npts))
    xR = np.zeros((3,npts))

    for i in range(npts):
        A = np.vstack((camL.R @ qL[:,i],-camR.R @ qR[:,i])).T
        b = tR - tL
        z,_,_,_ = np.linalg.lstsq(A,b,rcond=None)
        xL[:,i] = z[0]*qL[:,i]
        xR[:,i] = z[1]*qR[:,i]
 
    pts3L = camL.R @ xL + tL
    pts3R = camR.R @ xR + tR
    pts3 = 0.5*(pts3L+pts3R)

    return pts3

def compute_object_mask(object_img, background_img, threshold):
    """Computes a mask by thresholding the difference between two images."""
    diff = cv2.absdiff(object_img, background_img)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(diff_gray, int(threshold), 255, cv2.THRESH_BINARY)
    return mask > 0

def reconstruct(imprefixL, imprefixR, threshold, camL, camR,
                object_img, background_img, color_img, mask_threshold=20):

    HL, HmaskL = decode(imprefixL, 0, threshold)
    VL, VmaskL = decode(imprefixL, 20, threshold)
    HR, HmaskR = decode(imprefixR, 0, threshold)
    VR, VmaskR = decode(imprefixR, 20, threshold)

    maskL = HmaskL & VmaskL
    maskR = HmaskR & VmaskR
    print(f"L Valid pixels after decoding: {np.sum(maskL)}")
    print(f"R Valid pixels after decoding: {np.sum(maskR)}")

    object_mask = compute_object_mask(object_img, background_img, mask_threshold)
    print(f"  Object mask has {np.sum(object_mask)} pixels.")

    maskL &= object_mask
    maskR &= object_mask
    print(f"L Valid pixels after object mask: {np.sum(maskL)}")
    print(f"R Valid pixels after object mask: {np.sum(maskR)}")

    if np.sum(maskL) == 0 or np.sum(maskR) == 0:
        return np.empty((2,0)), np.empty((2,0)), np.empty((3,0)), np.empty((3,0))

    CL = 1024 * HL + VL
    CR = 1024 * HR + VR

    CL_valid = CL[maskL]
    CR_valid = CR[maskR]

    matched_codes, submatchL, submatchR = np.intersect1d(CL_valid, CR_valid, return_indices=True)

    yy_validL, xx_validL = np.where(maskL)
    yy_validR, xx_validR = np.where(maskR)

    xx_matchL, yy_matchL = xx_validL[submatchL], yy_validL[submatchL]
    xx_matchR, yy_matchR = xx_validR[submatchR], yy_validR[submatchR]

    pts2L = np.vstack((xx_matchL, yy_matchL))
    pts2R = np.vstack((xx_matchR, yy_matchR))
    
    pts3 = triangulate(pts2L, camL, pts2R, camR)
    
    xL, yL = pts2L[0, :].astype(int), pts2L[1, :].astype(int)
    sampled_colors = color_img[yL, xL]
    colors = sampled_colors[:, ::-1].T / 255.0

    return pts2L, pts2R, pts3, colors