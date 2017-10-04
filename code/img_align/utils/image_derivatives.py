
# @brief Utils
# @author Jose M. Buenaposada
# @date 2016/11/12
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr

import cv2
import numpy as np

def computeGrayImageGradients(image):
    """
    Computes the gray levels gradients of a an image

    The input image is converted first to gray levels if it is a color one.
    The returned gradients matrix G. has the horizontal gradient (X direction) in
    the first column and the vertical direction gradient (Y direction) in the
    second column. The image gradients are stored in row-wise order in the gradients
    matrix results

    :param image: np array image (as returned by OpenCV)
    :return: A np array with X gradient in column 0 and Y gradient in column 1
    """

    G = [0.3078, 0.3844, 0.3078] # Gaussian
    DoG = [-0.5, 0.0, 0.5]       # Derivative of a Gaussian

    if (len(image.shape) == 1):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = np.copy(image)

    rows = gray_image.shape[0]
    cols = gray_image.shape[1]
    num_pixels = rows * cols
    gradients  = np.zeros((num_pixels, 2), dtype=np.float64)

    grad_x = cv2.sepFilter2D(gray_image, cv2.CV_32F, np.array(DoG), np.array(G).T)
    grad_y = cv2.sepFilter2D(gray_image, cv2.CV_32F, np.array(G), np.array(DoG).T)

    gradients[:,0] = np.reshape(grad_x, (num_pixels, 1)).T
    gradients[:,1] = np.reshape(grad_y, (num_pixels, 1)).T

    return np.float64(gradients)


def computeGrayImageHessians(image):
    """
    Computes the gray levels second order derivative of a an image

    The input image is converted first to gray levels if it is a color one.
    The image gray levels second derivatives are stored in row-wise order in
    each column of the results matrix H.

    It computes the  \frac{\partial^2 I(\vx)}{\partial \vx \partial \vx}
    (the gradient of the gradients).

    The size of the output matrix is Nx4 being N the number of pixels.
    The first column of the results matrix has is I_xx, x derivative of the x gradient.
    The second column of the results matrix has is I_xy (= I_xy), x derivative of the y gradient.
    The third column of the results matrix has is I_yy, y derivative of the y gradient.

    :param image: np array image (as returned by OpenCV)
    :return: A np array with Nx4 dimensions
    """

    G = [0.3078, 0.3844, 0.3078] # Gaussian
    DoG = [-0.5, 0.0, 0.5]       # Derivative of a Gaussian

    if (len(image.shape) == 1):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = np.copy(image)

    rows = gray_image.shape[0]
    cols = gray_image.shape[1]
    num_pixels = rows * cols
    hessians  = np.zeros(num_pixels, 4)

    # gradients first
    grad_x = cv2.sepFilter2D(gray_image, cv2.CV_32F, np.array(DoG), np.array(G).T)
    grad_y = cv2.sepFilter2D(gray_image, cv2.CV_32F, np.array(G), np.array(DoG).T)

    # Hessian XX
    grad_xx = cv2.sepFilter2D(grad_x, cv2.CV_32F, np.array(DoG), np.array(G).T)

    # Hessian XY = Hessian YX
    grad_xy = cv2.sepFilter2D(grad_x, cv2.CV_32F, np.array(G), np.array(DoG).T)

    # Hessian YY
    grad_yy = cv2.sepFilter2D(grad_y, cv2.CV_32F, np.array(G), np.array(DoG).T)


    hessians[:,0] = grad_xx[:]
    hessians[:,1] = grad_xy[:]
    hessians[:,3] = grad_yy[:]

    return np.float64(hessians)
