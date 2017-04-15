from scipy.misc import imread, imsave, imresize
import numpy as np
import matplotlib.pyplot as plt
import random

# ----- Constants -----

# Constant for Torus Automorphism
K = 1

# ----- End Constants ----

# ----- Helper Functions -----

# Function to convert RGB image to grayscale
def to_grayscale(img):
	if len(img.shape) == 2:
		return img
	g = np.sum(img, axis=2);
	g /= 3
	return g

# Function to get Torus automarphed coordinates
# Refer Formula (1)
def get_torus_automorphism(x, y):
	global K
	m  = np.array([[1, 1], [K, K+1]]);
	n  = np.array([x, y])
	mn = m.dot(n)
	return mn[0], mn[1]

# Function to compute parity bits
# Refer Fig. 1
def create_parity(p):
	p0 = p&(1<<7) > 0	# MSB
	p1 = p&(1<<6) > 0
	p2 = p&(1<<5) > 0
	p3 = p&(1<<4) > 0
	a = p0 ^ p1 ^ p3
	b = p0 ^ p2 ^ p3
	c = p1 ^ p2 ^ p3
	return a*4 + b*2 + c

# Function to put parity bits
def put_parity(p, a):
	p /= 16
	p *= 16
	p += a
	return p

# Function to get parity bits
def get_parity(p):
	a = (p&4) > 0
	b = (p&2) > 0
	c = (p&1) > 0
	return a*4 + b*2 + c

# ----- End Helper Functions -----

# ----- Embed -----

# Function to get embedded image
def embed(img):
	eimg = np.array(img)
	for i in range(len(img)):
		for j in range(len(img[0])):
			a      = create_parity(img[i][j])
			_i, _j = get_torus_automorphism(i, j)
			_i, _j = _i%len(img), _j%len(img[0])
			eimg[_i][_j] = put_parity(img[_i][_j], a)
	return eimg

# ----- End Embed -----

# ----- Detect & Correct -----

# Mapping of parity to data bits
# Refer Fig. 7
map = [
#	 0, 1 
	[0, 6], #0
	[7, 1],
	[5, 3],
	[2, 4],
	[3, 5],
	[4, 2],
	[6, 0],
	[1, 7]  #7
]

# Function to predict pixel using jpeg-ls
# Refer Formula (3)
def jpeg_ls_predict(img, i, j):
	if i == 0 or j == 0:
		return img[i][j]
	a = img[i][j-1]
	b = img[i-1][j]
	c = img[i-1][j-1]
	if c >= max(a, b):
		return min(a, b)
	if c <= min(a, b):
		return max(a, b)
	return int(a) + int(b) - int(c)

# Function to check tampering of pixel
def check_pixel(img, i, j):
	s  = img[i][j]
	sd = s/16
	_i, _j = get_torus_automorphism(i, j)
	_i, _j = _i%len(img), _j%len(img[0])
	op = get_parity(img[_i][_j])
	sp = create_parity(s)
	# Check parities are equal
	if op != sp:
		# Check whether data part is tampered or parity part
		s0 = (sd/8)
		p  = jpeg_ls_predict(img, i, j)
		p0 = int((p&(1<<7))>0)
		if p0 != s0:
			# Data part tampered
			print "Data", (s0, p0), s, p, (sp, op)
			return p0*128 + map[op][p0]*16 + op
		# Parity part tampered
		print "Parity", s, p, (sp, op)
		return p
	return s 

# Function to detect and correct image
def detect_and_correct_error(img):
	for i in range(len(img)):
		for j in range(len(img[0])):
			img[i][j] = check_pixel(img, i, j)
	return img

# ----- End Detect & Correct ----

# ----- Noise -----

NOISYBITS = 10

def add_noise(img):
	n, m = len(img), len(img[0])
	for k in range(NOISYBITS):
		i, j = random.randrange(n), random.randrange(m)
		print i, j
		if i>n-2 or i<1 or j>m-2 or j<1:
			continue
		# img[i-1][j-1] = 0
		img[i-1][j]   = 0
		# img[i-1][j+1] = 0
		img[i][j-1]   = 0
		img[i][j]     = 0
		img[i][j+1]   = 0
		# img[i+1][j-1] = 0
		img[i+1][j]   = 0
		# img[i+1][j+1] = 0
	return img

# ----- End Noise -----

# ----- Test -----

# Read image and convert to grayscale
img = imread('eye.jpg')
img = to_grayscale(img)

# Embed image
eimg = embed(img)
plt.subplot(131)
plt.imshow(eimg, cmap = 'gray')

# Add Noise to image
nimg = add_noise(eimg)
plt.subplot(132)
plt.imshow(eimg, cmap = 'gray')

# Correct noisy image
cimg = detect_and_correct_error(eimg)
plt.subplot(133)
plt.imshow(cimg, cmap = 'gray')

# Plot results
plt.show()