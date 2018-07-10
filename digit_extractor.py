import numpy as np
import cv2
import queue


class digit_extractor(object):
	"""
	'digit_extractor' extracts the digit from a cell.
	It relies on the 'Largest Connected Component' algorithm.
	"""

	def __init__(self, image):
		self.graph = self.normalize(image.copy())
		self.W, self.H = self.graph.shape

		self.visited = np.full((self.W, self.H), False)
		self.digit = np.full((self.W, self.H), -1, dtype = np.int32)
		self.build_digit()

	def normalize(self, img):
		return img / np.amax(img) * 255

	def build_digit(self):
		component_id = 0
		A, C = 3 * self.H // 8, 5 * self.H // 8 + 1
		B, D = 3 * self.W // 8, 5 * self.W // 8 + 1

		for i in range(A, C):
			for j in range(B, D):
				if not self.visited[i][j]:
					self.bfs(i, j, component_id)
					component_id += 1

		component_sizes = np.asarray([len(np.argwhere(self.digit == i)) for i in range(component_id)])
		largest = np.argmax(component_sizes)
		self.digit = np.where(self.digit == largest, 1, 0)
		
	def bfs(self, i, j, num):
		q = queue.Queue()
		if self.graph[i][j] >= 220:
			q.put((i,j))
		while not q.empty():
			i, j = q.get()
			if i < 0 or i >= self.H or j < 0 or j >= self.W or self.graph[i][j] <= 170 or self.visited[i][j]:
				continue
			self.digit[i][j] = num
			self.visited[i][j] = True
			for di in [-1, 0, 1]:
				for dj in [-1, 0, 1]:
					if (di != 0 or dj != 0):
						q.put((i + di, j + dj))
