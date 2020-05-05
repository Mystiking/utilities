import cv2 as cv
import numpy as np
import utilities.python.polygons as polygons

'''
Computes the area of the difference between two binary images.
'''
def difference(img1, img2, max_val=255.):
    img_diff = img1 - img2
    img_diff[img_diff != max_val] = 0
    return img_diff

'''
Computes the area of the intersection between two binary images.
'''
def intersection(img1, img2):
    return cv.bitwise_and(img1, img2)

'''
Computes the the union between two binary images.
'''
def union(img1, img2):
    img_union = cv.bitwise_or(img1, img2)
    img_union[img_union > 0] = 1.0
    return img_union

'''
Creates binary proxy image with a transform.
'''
def create_proxy_image(proxy, size=(512, 512, 1), center_transform = None, scale_transform = None):
    proxy_image = np.zeros(size, np.uint8)
    proxy_points = polygons.unpack_proxy(proxy, 2)
    if not (center_transform is None) and not (scale_transform is None):
        proxy_points *= scale_transform
        proxy_points += center_transform
    proxy_points = np.array(proxy_points, np.int32)
    cv.fillPoly(proxy_image, [proxy_points.reshape((-1, 1, 2))], [proxy_points.shape[0]])
    proxy_image[proxy_image > 0] = 1.0
    return proxy_image