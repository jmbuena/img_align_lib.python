import cv2
import numpy as np

def main():
    # Homography transformation.
    pts1 = np.array([[52, 27],
                     [275, 35],
                     [274, 187],
                     [49, 183]], dtype=np.float32)
    pts2 = np.array([[0, 0],
                     [639, 0],
                     [639, 479],
                     [0, 479]], dtype=np.float32)
    pts3 = np.array([[0, 0],
                     [99, 0],
                     [99, 74],
                     [0, 74]], dtype=np.float32)
    H12 = cv2.getPerspectiveTransform(pts1, pts2) 
    H13 = cv2.getPerspectiveTransform(pts1, pts3) 
    im = cv2.imread('book_kk_1.jpg')
    assert(im is not None)
    rect12 = cv2.warpPerspective(im, H12, (640, 480))
    rect13 = cv2.warpPerspective(im, H13, (100, 75))

    cv2.imshow('Template', rect12)
    cv2.imshow('Template lowres', rect13)
    cv2.waitKey()

    cv2.imwrite('book.jpg', rect12)
    print 'Cropped image saved as book.jpg'

    cv2.imwrite('book_lowres.jpg', rect13)
    print 'Cropped image saved as book_lowres.jpg'


if __name__ == "__main__": 
    main()

