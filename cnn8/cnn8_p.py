import numpy as np
import tflearn as tfl
import imutils
import cv2
from PIL import Image
import PIL

def invrt(imm):
	return 255-imm

def imToAr(p):

	ss = (30, 34)

	img = Image.open('{0}.jpg'.format(p))
	img = img.resize(ss, resample=PIL.Image.BILINEAR)
	img.save('{0}.jpg'.format(p), 'JPEG')

	img = cv2.imread('{0}.jpg'.format(p))

	#s = cv2.resize(img, None, fx=30, fy=34)

	img = imutils.resize(img, width=30)

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	#edged = cv2.Canny(blurred, 50, 200, 255)

	return gray.astype(int)

def imToTensor(img):
	arr = np.array(img)

	return arr/255.0

def info():

	ix = [imToTensor(imToAr('t4/{0}'.format(i))) for i in range(10)]

	for i in range(10):
		ix.append(imToTensor(imToAr('6/{0}'.format(i))))

	for i in range(10):
		ix.append(imToTensor(imToAr('t2/{0}'.format(i))))

	for i in range(10):
		ix.append(imToTensor(imToAr('5/{0}'.format(i))))
	
	ref = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]
	ref = ref*5

	last = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

	for i in range(633):
		ix.append(imToTensor(imToAr('t5/{0}'.format(i))))
		ref.append(last)

	return ix, ref

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

def pred(name):

	X = [imToTensor(imToAr(name))]

	E = lambda a, b: [a, b]

	def boop(o):
		return o[0]

	hh = 0

	names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'Nan']

	for i in mod.predict(X):
		h = []
		for x in map(E, (i), (names)):
			h.append(x)

		pp = sorted(h, key=boop)[-1]

		#print (pp)

		#print (pp, hh, pp[1] == hh)
		#hh += 1

	return pp

cum = cv2.VideoCapture(0)

while 1:

	ret, frame = cum.read()

	imm = cv2.cvtColor(invrt(frame[50:-50, 200:-200]), cv2.COLOR_BGR2GRAY)

	cv2.imwrite('tempo.jpg', imm)

	cv2.imshow('hentai', imm)
	pp = pred('tempo')

	print (pp)

	k = cv2.waitKey(1)

	if k%256 == 27:
		break

	if k%256 == 32:
		cv2.imwrite('tt/{0}_camview.jpg'.format(pp[1]), frame)
		cv2.imwrite('tt/{0}.jpg'.format(pp[1]), imm)


cum.release()

cv2.destroyAllWindows()