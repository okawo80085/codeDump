import numpy as np
import tflearn as tfl
import cv2
from PIL import Image
import PIL

def invrt(imm):
	return 255-imm

def imToAr(imgArr):

	ss = (30, 34)

	img = Image.fromarray(imgArr)
	img = img.resize(ss, resample=PIL.Image.BILINEAR)

	img = np.array(img)

	return img.astype(int)/255.

def pred(name):

	X = [imToAr(name)]

	boop = lambda a: a[0]

	names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'Nan']

	for i in mod.predict(X):
		h = []
		for x in zip(i, names):
			h.append(x)

		pp = sorted(h, key=boop)[-1]

	return pp

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

cum = cv2.VideoCapture(0)

while 1:
	ret, frame = cum.read()

	if ret != True:
		break

	imm = cv2.cvtColor(invrt(frame[50:-50, 200:-200]), cv2.COLOR_BGR2GRAY)

	cv2.imshow('hentai', imm)
	pp = pred(imm)

	print (pp)

	k = cv2.waitKey(1)

	if k%256 == 27:
		break

	if k%256 == ord('r'):
		cv2.imwrite('tt/{0}_camview.jpg'.format(pp[1]), frame)
		cv2.imwrite('tt/{0}.jpg'.format(pp[1]), imm)

cum.release()

cv2.destroyAllWindows()
print ('(σ´-ω-`)σ')