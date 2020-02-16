import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

from game import Game


class Qgrid:
	def __init__(self,grid,gamma,beta):
		#list of cells 
		self.grid = []
		self.gamma = gamma
		self.beta = beta
		sizeI, sizeJ=grid.shape

		#Initialize cell 
		for i in range(sizeI):
			for j in range(sizeJ):
				cell = cell(grid,i,j)
				self.grid.append(cell)

	#Updates Qn for each state Qn-1
	def update_grid(self,trajectoriesXUR,trajectoriesXUX):
		for j in range(len(trajectoriesXUX)):
			i = 0
			for state1,action,reward in trajectoriesXUR:
				x1,y1 = trajectoriesXUX[i]
				u = trajectoriesXUX[i+1]
				x2,y2 = trajectoriesXUX[i+2]

				cellNumberState1 = x1*sizeJ + y1
				cellNumberState2 = x2*sizeJ + y2

				self.grid[cellNumberState1].update_cell(u,reward,self.grid[cellNumberState2])

	def print_grid(self):
		print("format :")
		print("cell number we read from left to right and up to down")
		print("Qvalue of cell Left-Up-Right-Down")
		for i in range(len(self.grid)):
			print("cell number :" + str(i))
			print(str(self.grid[i].cellLeft)+" "+str(self.grid[i].cellUp)+" "+str(self.grid[i].cellRight)+" "+str(self.grid[i].cellDown))
			print("\n")

class cell:
	def __init__(self,positionI,positionJ,alpha):
		self.i = positionI
		self.j = positionJ
		self.alpha = alpha

		i = positionI
		j = positionJ
		
		self.up = 0
		self.down = 0
		self.left = 0
		self.right = 0

	def update_cell(self,action,reward,cell,gamma):	
		if action == "UP":
			self.up = (1 − self.alpha)*self.up + self.alpha*(reward + gamma * cell.get_max())
		if action == "DOWN":
			self.down = (1 − self.alpha)*self.down + self.alpha*(reward + gamma * cell.get_max())
		if action == "LEFT":
			self.left = (1 − self.alpha)*self.left + self.alpha*(reward + gamma * cell.get_max())
		if action == "RIGHT":
			self.right = (1 − self.alpha)*self.right + self.alpha*(reward + gamma * cell.get_max())

	def get_max(self):
		return max(self.up,self.down,self.right,self.left)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--stochastic",help="Add stochasticity to the program",action="store_true")
	args = parser.parse_args()

	array = np.array([
	        [-3, 1, -5, 0, 19],
	        [6, 3, 8, 9, 10],
	        [5, -8, 4, 1, -8],
	        [6, -9, 4, 19, -5],
	        [-20, -17, -4, -3, 9]])

	size_x,size_y=array.shape
	steps = 500
	discount =  0.99
	beta = 0.5
	if args.stochastic ==False:
		beta = 0
	initialI = 0
	initialJ = 0
	policy = "Q"

	vectorScores = np.zeros((size_x,steps))

	plt.figure(figsize=(20,10))


	legend = []
	for i in range(size_x):
		for j in range(size_y):
			for k in range(5):
				game = Game(initialI,initialJ,array,discount,steps, beta,policy)
				game.start_game()
				vectorScores[i][:] = vectorScores[i][:] + game.get_scores()

			vectorScores[i][:] = vectorScores[i][:]/5
			plt.plot(vectorScores[i])
			print(' The expected return for the row '+ str(i+1)+' and column '+str(j+1)+' : ' + str(game.scores[steps - 1]))
			legend.append("row "+str(i+1) + " column " + str(j+1))
			initialJ = initialJ + 1

		initialJ = 0
		initialI = initialI + 1