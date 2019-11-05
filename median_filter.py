import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("fire.png")
img = img[:, :, ::-1]

mF = cv2.medianBlur(img, 5)

# B, G, R = cv2.split(mF)

# L = 116 * pow((0.299 * R + 0.587 * G + 0.114 * B), 1/3) - 16
# a = 500 * (1.006 * pow(0.607 * R + 0.174 * G + 0.201 * B, 1/3) - pow(0.299 * R + 0.587 * G + 0.114 * B, 1/3))
# b = 200 * (pow(0.299 * R + 0.587 * G + 0.114 * B, 1/3) - 0.846 * pow(0.066 * G + 1.117 * B, 1/3))
#
# lab = cv2.merge((L, a, b))

lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)

L,a,b = cv2.split(lab)

# final_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
# print(a.type())
# a = np.array(a+128,dtype=np.uint8)
plt.subplot(131),plt.imshow(img),plt.title('original image')
plt.xticks([]), plt.yticks([])

ret3,th3 = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


plt.subplot(132),plt.imshow(th3, cmap='Greys_r'),plt.title('grayscale')
plt.xticks([]), plt.yticks([])

# plt.subplot(133),plt.imshow(final_image),plt.title('ori')
# plt.xticks([]), plt.yticks([])
# plt.subplot(143),plt.imshow(b),plt.title('b')
# plt.xticks([]), plt.yticks([])
# plt.subplot(144),plt.imshow(L),plt.title('L')
# plt.xticks([]), plt.yticks([])


plt.show()
