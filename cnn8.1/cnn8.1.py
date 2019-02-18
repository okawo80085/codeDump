import numpy as np
import tflearn as tfl
from tflearn.data_utils import to_categorical
import cv2
from PIL import Image
import PIL
import os

def imToAr(p):

	ss = (30, 34)

	img = Image.open(p)
	img = img.resize(ss, resample=PIL.Image.BILINEAR)

	img = np.array(img)

	return img.astype(int)/255.

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
	labls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'z']

	X = np.array([np.zeros((34, 30))])
	Y = []

	for i in paths:
		for j in labls:
			if str(j) in i:
				X = np.concatenate((X, [imToAr(i)]), axis=0)
				Y.append(j)

	Yn = [10 if i=='z' else i for i in Y]

	return X[1:], to_categorical(Yn, 11), Y


dataPaths = sorted(getDataPaths('data'))

X, Y, labls = pathsToData(dataPaths)

n_classes = 11

net = tfl.input_data([None, 34, 30])
net = tfl.conv_1d(net, 25, 5, activation='relu')
net = tfl.max_pool_1d(net, 2)
net = tfl.dropout(net, 0.8)
net = tfl.conv_1d(net, 35, 3, activation='relu')
net = tfl.max_pool_1d(net, 2)
net = tfl.dropout(net, 0.75)
net = tfl.fully_connected(net, 128, activation='relu')
net = tfl.fully_connected(net, 50, activation='relu')
net = tfl.fully_connected(net, n_classes, activation='softmax')

reg = tfl.regression(net, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.00003)

mod = tfl.DNN(reg, tensorboard_verbose=0)

mod.load('conv_nn8.1')

mod.fit(X, Y, n_epoch=100, shuffle=True, show_metric=True, batch_size=11, run_id='ligma4')

mod.save('conv_nn8.1')

names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'z']

boop = lambda a: a[0]

for i in zip(mod.predict(X), labls):
	h = []
	for x in zip(i[0], names):
		h.append(x)

	pp = sorted(h, key=boop)[-1]

	print (pp, pp[1] == i[1], i[1])

print ('(σ´-ω-`)σ')