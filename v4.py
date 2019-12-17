import numpy as np
import cv2 as cv
import warnings
warnings.filterwarnings("ignore")

def colour_filter(frame):
	b, g, r = cv.split(frame)
	ret,thresh1 = cv.threshold(r,180,255,cv.THRESH_TOZERO)
	thresh2 = (r > g)*1
	thresh3 = (g > b)*1
	thresh2 = cv.bitwise_and(thresh2, thresh3)
	thresh1 = (np.multiply(thresh1, thresh2)).astype(np.uint8)
	thresh2 = ((g/(r+1)) >= 0.25)
	thresh1 = (np.multiply(thresh1, thresh2)).astype(np.uint8)
	thresh2 = ((g/(r+1)) <= 0.65)
	thresh1 = (np.multiply(thresh1, thresh2)).astype(np.uint8)
	thresh2 = ((b/(r+1)) <= 0.45)
	thresh1 = (np.multiply(thresh1, thresh2)).astype(np.uint8)
	thresh2 = ((b/(r+1)) >= 0.05)
	thresh1 = (np.multiply(thresh1, thresh2)).astype(np.uint8)
	thresh2 = ((b/(g+1)) >= 0.2)
	thresh1 = (np.multiply(thresh1, thresh2)).astype(np.uint8)
	thresh2 = ((b/(g+1)) <= 0.6)
	thresh1 = (np.multiply(thresh1, thresh2)).astype(np.uint8)
	edge = cv.Canny(thresh1, 180, 230)
	dilated = cv.dilate(edge, cv.getStructuringElement(cv.MORPH_RECT,(3,3)), iterations=1)

	return dilated


kernel = np.ones((7, 7), np.float32)/49.0
MIN_THRESHOLD = 0.13
MAX_THRESHOLD = 1.0
cap = None
c = 0
i = 0
while i < 1:
	# file_name = "fire" + str(i) + ".mp4"
	file_name = "no_fire600.mp4"
	# file_name = "fire3.mp4"
	cap = cv.VideoCapture(file_name)

	ret, frame = cap.read()
	frame = cv.resize(frame, (256, 256), interpolation=cv.INTER_AREA)

	cf1 = colour_filter(frame)
	BATCH = 500

	# S = cf1.copy()
	# S_conv = cv.filter2D(S, -1, kernel)
	mm1 = np.zeros((BATCH, 256, 256))
	# mm1[0] = S_conv.copy()
	mm1[0] = cf1.copy()

	# SUM = np.zeros((256, 256), np.float32)
	SUM = cf1.copy()
	counter = 1
	flag = False
	while (True):
		ret, frame = cap.read()
		if frame is None:
			break
		frame = cv.resize(frame, (256, 256), interpolation=cv.INTER_AREA)
		cv.imshow('original', frame)

		cf2 = colour_filter(frame)
		no_of_fire_pixels = np.sum((cf2 > 0))
		# print(no_of_fire_pixels)
		cv.imshow('check1', cf2)

		if (no_of_fire_pixels > 0):
			# print(counter)
			counter = (counter + 1) % BATCH
			diff = abs(cf2 - cf1)
			thresh = (diff > 0)*1.0
			SUM = SUM - (mm1[counter])
			mm1[counter] = thresh.copy()
			# mm1[counter] = cv.filter2D(thresh, -1, kernel)
			SUM += (mm1[counter])
			Max = np.max(SUM) / no_of_fire_pixels
			Mean = np.mean(SUM) / BATCH
			cv.imshow('convolve', mm1[counter])

			if counter == 0:
				flag = True
			if flag and (Mean > 0.01 or Max > 0.5):
				# print("Max:", Max, "Mean:", Mean, "No. of pixels:", no_of_fire_pixels)
				print("Fire Detected")

		cf1 = cf2.copy()
		if cv.waitKey(1) & 0xFF == ord('q'):
			break
	i += 1

cap.release()
cv.destroyAllWindows()