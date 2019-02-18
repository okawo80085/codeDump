import numpy as np
import tflearn as tfl
import imutils
import cv2

def imToAr(p):

	img = cv2.imread('{0}.jpg'.format(p))

	#img = imutils.resize(img, height=50)

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	#edged = cv2.Canny(blurred, 50, 200, 255)

	return gray.astype(int)

def imToTensor(img):
	x = len(img)
	y = len(img[0]) * x

	arr = np.reshape(img, y)

	return arr/255.0

def info():

	ix = [imToTensor(imToAr('4/{0}'.format(i))) for i in range(10)]
	
	ref = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

	return ix, ref

X, Y = info()

num_classes = 10

l1 = tfl.input_data(shape=[None, 1020])
l2 = tfl.fully_connected(l1, 16, activation='relu')
l2 = tfl.fully_connected(l2, 10, activation='relu')
sm = tfl.fully_connected(l2, num_classes, activation='softmax')

reg = tfl.regression(sm, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.0000001)

net = tfl.DNN(reg, tensorboard_verbose=3)

net.load('net7.tflearn')

net.fit(X, Y, n_epoch=100, shuffle=True, show_metric=True, run_id='ligma2')

E = lambda a, b: [a, b]

def boop(o):
	return o[0]

hh = 0

for i in net.predict(X):
	h = []
	for x in map(E, (i), (range(10))):
		h.append(x)

	pp = sorted(h, key=boop)[-1]

	#print (pp)

	print (pp, hh, pp[1] == hh)
	hh += 1

net.save('net7.tflearn')