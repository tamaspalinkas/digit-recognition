import numpy as np
import tensorflow as tf
import cv2
import dnn

def main():
	mnist = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	cap = cv2.VideoCapture(0)

	dnn.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	dnn.model.fit(x_train, y_train, epochs=3)

	while (cap.isOpened()):
		ret, img = cap.read()
		img, contours, thresh = get_img_contour_thresh(img)
		if len(contours) > 0:
			contour = max(contours, key=cv2.contourArea)
			if cv2.contourArea(contour) > 2500:
				x, y, w, h = cv2.boundingRect(contour)
				image = thresh[y:y + h, x:x + w]
				image = cv2.resize(image, (28, 28))
				normalized = tf.keras.utils.normalize(image,axis=1)
				prediction = dnn.model.predict([[normalized]])
				print(np.argmax(prediction))

		x, y, w, h = 0, 0, 300, 300
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv2.imshow("Frame", img)
		cv2.imshow("Contours", thresh)
		k = cv2.waitKey(10)
		if k == 27:
			break


def get_img_contour_thresh(img):
	x, y, w, h = 0, 0, 300, 300
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (35, 35), 0)
	ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	thresh1 = thresh1[y:y + h, x:x + w]
	contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
	return img, contours, thresh1

main()
