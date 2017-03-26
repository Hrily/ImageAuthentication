from scipy.misc import imread, imsave, imresize
import numpy as np
import matplotlib.pyplot as plt

K = 1			# Constant for Torus Automorphism

# Function to convert RGB image to grayscale
def to_grayscale(img):
	if len(img.shape) == 2:
		return img
	g = np.sum(img, axis=2);
	g /= 3
	return g

# Function to convert decimaml to bivector
def to_bitvector(img):
	b = np.empty_like(img,  dtype = 'object')
	for i in range(len(img)):
		for j in range(len(img[0])):
			b[i][j] = bin(img[i][j])[2:].zfill(7)
	return b

# Function to get Torus automarphed coordinates
def get_torus_automorphism(x, y):
	global K
	m  = np.array([[1, 1], [K, K+1]]);
	n  = np.array([x, y])
	mn = m.dot(n)
	return mn[0], mn[1]

# Function to xor char
def xor(a, b):
	if not a == b:
		return '1'
	return '0'

# Function to get parity bits
def create_parity(p):
	a = xor(xor(p[0], p[1]), p[3])
	b = xor(xor(p[0], p[2]), p[3])
	c = xor(xor(p[1], p[2]), p[3])
	return a, b, c

# Function to get embedded image
def embed(img):
	for i in range(len(img)):
		for j in range(len(img[0])):
			a, b, c = create_parity(img[i][j])
			_i, _j  = get_torus_automorphism(i, j)
			_i, _j  = _i%len(img), _j%len(img[0])
			s = img[_i][_j]
			img[_i][_j] = s[0:4] + a + b + c
	return img

img = imread('eifel.jpg')
img = to_grayscale(img)
img = to_bitvector(img)

img = embed(img)

print img[0]