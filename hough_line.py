import cv2
import numpy as np
import math

img = cv2.imread('road.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gaussian = cv2.medianBlur(gray, 15)

# canny method
img_canny = cv2.Canny(img_gaussian, 50, 120)
# sobel method
img_sobelx = cv2.Sobel(img_gaussian, cv2.CV_8U, 0, 1, ksize=3)
img_sobely = cv2.Sobel(img_gaussian, cv2.CV_8U, 0, 1, ksize=3)
img_sobel = img_sobelx + img_sobely
# prewitt method
kernelx = np.array([[1, 1, 1],
                    [0, 0, 0],
                    [-1, -1, -1]])
kernely = np.array([[1, 0, -1],
                    [1, 0, -1],
                    [1, 0, -1]])
img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
img_prewit = img_prewittx + img_prewitty
# robert method
roberts_cross_x = np.array([[1, 0],
                            [0, -1]])
roberts_cross_y = np.array([[0, -1],
                            [1, 0]])
img_roberts_x = cv2.filter2D(img_gaussian, -1, roberts_cross_x)
img_roberts_y = cv2.filter2D(img_gaussian, -1, roberts_cross_y)
img_roberts = img_roberts_x + img_roberts_y

# def imshow_components(labels):
#     # Map component labels to hue val
#     label_hue = np.uint8(179*labels/np.max(labels))
#     blank_ch = 255*np.ones_like(label_hue)
#     labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

#     # cvt to BGR for display
#     labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

#     # set bg label to black
#     labeled_img[label_hue==0] = 0

#     cv2.imshow('labeled.png', labeled_img)
#     cv2.waitKey()


def texture_edge_connected_component(img, name):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = 80
    img2 = np.zeros((output.shape), np.uint8)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2


def detect_lines(img3, name):
    img2 = img.copy()
    num = 0
    lines = cv2.HoughLinesP(img3, 1, np.pi/180, 70, None, 50, maxLineGap=150)
    no_of_Lines = 0
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            no_of_Lines = no_of_Lines + 1
            #cv2.line(img2, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.line(img2, (l[0], l[1]), (l[2], l[3]),
                     (0, 0, 255), 1, cv2.LINE_AA)
            num = num+1
        st=""+str(num)
        img2 = cv2.putText(img2,st, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(name, img2)
    # return img


img_canny = texture_edge_connected_component(img_canny, "Canny")
#img_sobel = texture_edge_connected_component(img_sobel, "sobel")
detect_lines(img_canny, "Canny")
detect_lines(img_sobel, "Sobel")
detect_lines(img_prewit, "Prewitt")
detect_lines(img_roberts, "Roberts")
# cv2.imshow("Original Image", img)
# cv2.imshow("Canny Image", img_canny)
# cv2.imshow("Sobel Image", img_sobel)
# cv2.imshow("Prewitt Image", img_prewit)
# cv2.imshow("Roberts Image", img_roberts)

cv2.waitKey(0)
cv2.destroyAllWindows()
