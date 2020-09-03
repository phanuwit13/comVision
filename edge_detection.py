import cv2
import numpy as np
import math
import sys

print('1:Token Image')
print('2:Eye Image')
print('3:Road Image')
x = input("INPUT SELECT : ")
print(x)
i = 0

if x == '1':
    img = cv2.imread('circle.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.medianBlur(gray, 3)
elif x == '2':
    img = cv2.imread('eyes3.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.medianBlur(gray, 19)
elif x == '3':
    img = cv2.imread('road.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.medianBlur(gray, 15)
else:
    sys.exit()


# img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)

# canny method
img_canny = cv2.Canny(img_gaussian, 50, 120 )
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
    min_size = 50
    img2 = np.zeros((output.shape), np.uint8)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2


img_canny = texture_edge_connected_component(img_canny, "Canny")
# img_sobel = texture_edge_connected_component(img_sobel, "sobel")
# img_prewit = texture_edge_connected_component(img_prewit, "Prewitt")
# img_roberts = texture_edge_connected_component(img_roberts, "Roberts")

cv2.imshow("Original Image", img)
cv2.imshow("Canny Image", img_canny)
cv2.imshow("Sobel Image", img_sobel)
cv2.imshow("Prewitt Image", img_prewit)
cv2.imshow("Roberts Image", img_roberts)

cv2.waitKey(0)
cv2.destroyAllWindows()
