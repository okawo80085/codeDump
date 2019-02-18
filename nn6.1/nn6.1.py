import numpy as np
import tflearn as tfl
from tflearn.data_utils import to_categorical
import cv2
import os

def imToAr(p):

	img = cv2.imread(p)

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	img = gray.astype(np.float64)

	size = img.shape[0]*img.shape[1]

	arr = np.reshape(img, size)

	return arr/255.0

def get_format(name):
	return name.rsplit('.')[-1]

def getDataPaths(rootFolder):

	gudFormats = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']

	pathList = [i[0] for i in os.walk(rootFolder)]

	out = []

	for i in pathList:
		for j in os.listdir(i):
			if os.path.isfile(i + '/' + j):
				if get_format(j) in gudFormats:
					out.append(i + '/' + j)

	return out

def pathsToData(paths):
	labls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

	X = np.array([np.zeros(30*34)])
	Y = []

	for i in paths:
		for j in labls:
			if str(j) in i:
				X = np.concatenate((X, [imToAr(i)]), axis=0)
				Y.append(j)

	return X[1:], to_categorical(Y, 10), Y


dataPaths = sorted(getDataPaths('data'))

X, Y, labls = pathsToData(dataPaths)

num_classes = 10

l1 = tfl.input_data(shape=[None, 1020])
l2 = tfl.fully_connected(l1, 16, activation='relu')
l2 = tfl.fully_connected(l2, 10, activation='relu')
sm = tfl.fully_connected(l2, num_classes, activation='softmax')

reg = tfl.regression(sm, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.00001)

net = tfl.DNN(reg, tensorboard_verbose=3)

net.load('net7.tflearn')

net.fit(X, Y, n_epoch=100, shuffle=True, batch_size=5, show_metric=True, run_id='ligma2')

net.save('net7.tflearn')

boop = lambda a: a[0]

for i in zip(net.predict(X), labls):
	h = []
	for x in zip(i[0], range(10)):
		h.append(x)

	pp = sorted(h, key=boop)[-1]

	print (pp, pp[1] == i[1], i[1])

print ('(σ´-ω-`)σ')