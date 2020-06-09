import cv2
import numpy as np


def auto_canny(image,sigma = 0.33):
    v = np.median(image)
    
    lower = int(max(0,(1.0-sigma)*v))
    upper = int(min(255,(1.0+sigma)*v))
    return cv2.Canny(image,lower,upper)

def dpot(a, b):
    return (a-b) ** 2
def adist(a,b):
    return np.sqrt(dpot(a[0],b[0]) + dpot(a[1],b[1]))
def max_distance(a1, a2, b1, b2):
    dist1 = adist(a1, a2)
    dist2 = adist(b1, b2)
    return int(max([dist1,dist2]))
def sort_points(pts):
    ret = np.zeros((4, 2), dtype = "float32")
    sumF = pts.sum(axis = 1)
    diffF = np.diff(pts, axis = 1)
    ret[0] = pts[np.argmin(sumF)]
    ret[1] = pts[np.argmin(diffF)]
    ret[2] = pts[np.argmax(sumF)]
    ret[3] = pts[np.argmax(diffF)]
    return ret
def get_points(r):
    p1 = (abs(int(r[1][0]/2) - r[0][0]), abs(int(r[1][1]/2) - r[0][1]))
    p2 = (abs(int(r[1][0]/2) + r[0][0]), abs(int(r[1][1]/2) - r[0][1]))
    p3 = (abs(int(r[1][0]/2) - r[0][0]), abs(int(r[1][1]/2) + r[0][1]))
    p4 = (abs(int(r[1][0]/2) + r[0][0]), abs(int(r[1][1]/2) + r[0][1]))
    return p1, p2, p3, p4
def perform_perspective(p1, p2, p3, p4, orig_im):
    pts_src = np.array([list(p1), list(p2), list(p3), list(p4)])
    pts_dst = np.array([[0.0, 0.0],[imw, 0.0],[0.0, imh],[imw, imh]])
    M = cv2.getPerspectiveTransform(pts_src.astype(np.float32), pts_dst.astype(np.float32))
    image = cv2.warpPerspective(orig_im, M, (imw, imh))
    return image
def fix_prespective(image,pts):
    (tl,tr,br,bl) = sort_points(pts)
    maxW = max_distance(br, bl, tr, tl)
    maxH = max_distance(tr, br, tl, bl)
    dst = np.array([[0, 0],[maxW - 1, 0],[maxW - 1, maxH - 1],[0, maxH - 1]], dtype = "float32")
    transform = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst)
    fixed = cv2.warpPerspective(image, transform, (maxW, maxH))
    return fixed