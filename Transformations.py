import numpy as np

#Turn array of 3*n^2 values into matrix of n^2 rows of 3 R,G,B values
""" For example, transform [R, R, R, R, G, G, G, G, B, B, B, B] 
	into 				   [(R,G,B), (R,G,B), (R,G,B), (R,G,B)]"""
def pixelize(img):
	result = []
	num_pixel = int(len(img)/3)
	for i in range(num_pixel):
		result.append(img[i])
		result.append(img[i+num_pixel])
		result.append(img[i+2*num_pixel])

	result = np.array(result).reshape(num_pixel,3)

	return result

#Turn matrix of n^2 rows of 3 R,G,B values into array of 3*n^2 values
""" For example, transform [(R,G,B), (R,G,B), (R,G,B), (R,G,B)] 
	into 				   [R, R, R, R, G, G, G, G, B, B, B, B]"""
def depixelize(img):
	result = []
	num_pixel = img.shape[0]
	for i in range(3):
		for j in range(num_pixel):
			result.append(img[j,i])

	return np.array(result)

#Turn array of 3*n^2 values into n^2 array of grayscale values
""" For example, transform [(R,G,B), (R,G,B), (R,G,B), (R,G,B)] 
	into 				   [Gray, Gray, Gray, Gray]"""
def rgb2gray(img):
	r, g, b = img[0:1024], img[1024:2048], img[2048:3072]
	gray = 0.2989 * r.astype("float") + 0.5870 * g.astype("float") + 0.1140 * b.astype("float")				#Formula to turn rgb to grayscale
	return gray

#Rotate an array of 3*n^2 values, as an image, 90 degrees k times
def rotate(img, k):
	if k == 1:
		img_p = pixelize(img)
		result = np.zeros(img_p.shape)

		n = int(np.sqrt(len(img_p)))

		for i in range(n):
			for j in range(n):
				""" In a regular matrix in form of [i,j], the formula is result[i,j] = img_p[(n-j-1),i]
					But since this is a matrix flattened to an array, the below formula is used"""
				result[i*n+j] = img_p[(n-j-1)*n+i]

		result = depixelize(result)
	else:
		result = img
		for i in range(k):
			result = rotate(result, 1)

	return result

#Mirror an array of 3*n^2 values, as an image, horizontally/vertically
#"vertically" is a boolean. If true, flip vertically, else flip horizontally
def mirror(img, vertically):
	img_p = pixelize(img)
	result = np.zeros(img_p.shape)

	n = int(np.sqrt(len(img_p)))

	for i in range(n):
		for j in range(n):
			if vertically:
				""" In a regular matrix in form of [i,j], the formula is result[i,j] = img_p[(n-i-1),j]
					But since this is a matrix flattened to an array, the below formula is used"""
				result[i*n+j] = img_p[(n-i-1)*n+j]
			else:
				""" In a regular matrix in form of [i,j], the formula is result[i,j] = img_p[i,(n-j-1)]
				But since this is a matrix flattened to an array, the below formula is used"""
				result[i*n+j] = img_p[i*n+(n-j-1)]

	result = depixelize(result)

	return result
