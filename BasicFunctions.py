from math import factorial
from operator import mul
from itertools import chain
from subprocess import PIPE, Popen
from functools import partial
import numpy as np


def prod(collection):
	"""
	Product of all elements in the collection.
	elements must support multiplication
	"""
	return reduce(mul, collection)


try:
	from scipy import comb
	nCr = partial(comb, exact=True)
except ImportError:
	def nCr(n,r):
		return prod(range(n-r+1,n+1)) / factorial(r)


def game_size(n,s):
	"""
	Number of profiles in a symmetric game with n players and s strategies.
	"""
	return nCr(n+s-1,n)


def profile_repetitions(p):
	"""
	Number of normal form profiles that correspond to a role-symmetric profile.
	"""
	return prod([factorial(sum(row)) / prod(map(factorial, row)) for row in p])


def average(l):
	"""Arithmetic mean."""
	return sum(l, 0.0) / len(l)


def flatten(l):
	"""Concatenate sublists into a single list."""
	return list(chain(*l))


def one_line(s, line_width=80):
	"""If string s is longer than line width, cut it off and append '...'"""
	if len(s) > line_width:
		s = s[:3*line_width/4] + "..." + s[-line_width/4+3:]
	return s.replace("\n", "")


def leading_zeros(i, m):
	"""Pad the string of integer i with leading zeros to equal length of m."""
	return str(i).zfill(len(str(m)))


def call(proc_name, in_pipe):
	p = Popen(proc_name, shell=True, stdin=PIPE, stdout=PIPE)
	return p.communicate(in_pipe)[0]


tiny = 1e-10


def weighted_least_squares(x, y, weights):
	"""appends the ones for you; puts 1D weights into a diagonal matrix"""
	try:
		A = np.append(x, np.ones([x.shape[0],1]), axis=1)
		W = np.zeros([x.shape[0]]*2)
		np.fill_diagonal(W, weights)
		return y.T.dot(W).dot(A).dot(np.linalg.inv(A.T.dot(W).dot(A)))
	except np.linalg.linalg.LinAlgError:
		z = A.T.dot(W).dot(A)
		for i in range(z.shape[0]):
			for j in range(z.shape[1]):
				z[i,j] += np.random.uniform(-tiny,tiny)
		return y.T.dot(W).dot(A).dot(np.linalg.inv(z))



