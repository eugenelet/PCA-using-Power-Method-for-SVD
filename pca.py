import numpy as np
from math import sqrt
from random import normalvariate
from numpy.linalg import svd as svd_np
from numpy.linalg import eig
from numpy.linalg import norm
import pylab


def fro_norm(x):
	return sqrt((x**2).sum().astype('float64'))

def normalize_vec(x):
	return x / sqrt((x**2).sum())

def generateUnitVector(n):
	return normalize_vec(np.arange(n))

def cov(A):
    n, m = A.shape
    avg = A.mean(axis=0)
    A = A - avg
    if n >= m:
        B = np.dot(A.T, A)
    else:
        B = np.dot(A, A.T)
    return B / float(A.shape[0])

def svd_1d(A, epsilon=2**(-17)):
	n, m = A.shape
	x = generateUnitVector(min(n,m))  # Due to hardware constraints
	
	lastV = None
	currentV = x
	
	if n > m:
		B = np.dot(A.T, A)
	else:
		B = np.dot(A, A.T)
	
	# print np.max(B)
	iterations = 0
	while True:
		#print iterations
		iterations += 1
		lastV = currentV
		currentV = np.dot(B, lastV)
		currentV = currentV / fro_norm(currentV)
		
		if abs(np.dot(currentV, lastV)) > 1 - epsilon:
			# print("converged in {} iterations!".format(iterations))
			return currentV

def svd(A, k=None, epsilon=2**(-17)):
	'''
		Compute the singular value decomposition of a matrix A
		using the power method. A is the input matrix, and k
		is the number of singular values you wish to compute.
		If k is None, this computes the full-rank decomposition.
	'''
	A = np.array(A, dtype=float)
	n, m = A.shape
	svdSoFar = []

	if k is None:
		k = min(n, m)

	for i in range(k):
		matrixFor1D = A.copy()

		for singularValue, u, v in svdSoFar[:i]:
			matrixFor1D -= singularValue * np.outer(u, v)
		if n > m:
			v = svd_1d(matrixFor1D, epsilon=epsilon)  # next singular vector
			print A.shape, v.shape
			u_unnormalized = np.dot(A, v)
			sigma = fro_norm(u_unnormalized)  # next singular value
			u = u_unnormalized / sigma
			print u.shape
		else:
			u = svd_1d(matrixFor1D, epsilon=epsilon)  # next singular vector
			v_unnormalized = np.dot(A.T, u)
			sigma = fro_norm(v_unnormalized)  # next singular value
			v = v_unnormalized / sigma

		svdSoFar.append((sigma, u, v))

	singularValues, us, vs = [np.array(x) for x in zip(*svdSoFar)]
	return singularValues, us.T, vs


def pca_np(dataMat, topNfeat=9999999):
	meanVals = np.mean(dataMat, axis=0)
	meanRemoved = dataMat - meanVals #remove mean
	covMat = np.cov(meanRemoved, rowvar=0)
	eigVects, eigVals, _ = svd_np(covMat)
	eigValInd = np.argsort(eigVals)            #sort, sort goes smallest to largest
	eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
	redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
	redEigVal = eigVals[eigValInd] # return reduced eig val for whitening
	lowDDataMat = meanRemoved.dot(redEigVects)#transform data into new dimensions
	return lowDDataMat, redEigVects, redEigVal


def pca(dataMat, topNfeat=9999999, epsilon=2**(-17)):
	meanVals = np.mean(dataMat, axis=0)
	meanRemoved = dataMat - meanVals #remove mean
	covMat = np.cov(meanRemoved, rowvar=0)
	eigVals, eigVects, _ = svd(covMat, topNfeat, epsilon=epsilon)
	eigValInd = np.argsort(eigVals)            #sort, sort goes smallest to largest
	eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
	redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
	redEigVal = eigVals[eigValInd] # return reduced eig val for whitening
	lowDDataMat = meanRemoved.dot(redEigVects)#transform data into new dimensions
	return lowDDataMat, redEigVects, redEigVal



# Test frobenius norm
a = np.array([[1,2,3,4], [4,3,2,1]])
print a
print "Frobenius norm: {}".format(fro_norm(a))

# Test vector normalization
a = np.array([3,4,5,6])
b = normalize_vec(a)
print "Inner product of normalized vector: {}".format(b.dot(b))

testMatrix = np.array([
		[2, 5, 3],
		[1, 2, 1],
		[4, 1, 1],
		[3, 5, 2],
		[5, 3, 1],
		[4, 5, 5],
		[2, 4, 2],
		[2, 2, 5],
	], dtype='float64')

#testMatrix = np.random.randint(7, size=(5, 5))
print 'Test self implemented covariance:'

# Self implemented covariance
cov_ = cov(testMatrix)

# numpy implemented covariance
cov_np = np.cov(testMatrix, rowvar=0, ddof=0)

print 'Self implemented:'
print cov_
print 'From numpy'
print cov_np

# Test both PCAs
lowDDataMat, redEigVects, redEigVal = pca(testMatrix, 3)
lowDDataMat_np, redEigVects_np, redEigVal_np = pca_np(testMatrix, 3)

print "self implemented: "
print lowDDataMat
print redEigVal
print "using numpy: "
print lowDDataMat_np
print redEigVal_np

# Test using different epsilon
testMatrix = np.random.randint(7, size=(5, 5))
error = []
lowDDataMat_np, redEigVects_np, redEigVal_np = pca_np(testMatrix, 3)
for i in range(17):
	lowDDataMat, redEigVects, redEigVal = pca(testMatrix, 3, epsilon=2**(-i))
	error.append(((abs(lowDDataMat) - abs(lowDDataMat_np))**2).sum())

pylab.figure()
pylab.plot(error)
pylab.title('Error vs Epsilon')
pylab.xlabel(r'$epsilon (2^{-i})$')
pylab.ylabel('Error')

error = []
for i in range(100):
	X = np.random.randint(7, size=(5, 5))
	lowDDataMat_np, redEigVects_np, redEigVal_np = pca_np(X, 3)
	lowDDataMat, redEigVects, redEigVal = pca(X, 3, epsilon=2**(-17))
	error.append(((abs(lowDDataMat) - abs(lowDDataMat_np))**2).sum())

pylab.figure()
pylab.plot(error)
pylab.title('Error distribution using randomly generated matrices')
pylab.xlabel('matrix')
pylab.ylabel('error')

pylab.show()