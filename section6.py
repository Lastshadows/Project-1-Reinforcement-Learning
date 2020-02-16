import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import random

from game import Game

"""
Cell class represents the 4 tuples (state,action)
of a given state X 
"""
class Cell:
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

	#Updates the cell corresponding cell value given the action, the reward, the cell reached 
	#and the value of gamma
	def update_cell(self,action,reward,cell,gamma):	
		if action == "UP":
			self.up = (1-self.alpha)*self.up + self.alpha*(reward + gamma * cell.get_max())
		elif action == "DOWN":
			self.down = (1 - self.alpha)*self.down + self.alpha*(reward + gamma * cell.get_max())
		elif action == "LEFT":
			self.left = (1 - self.alpha)*self.left + self.alpha*(reward + gamma * cell.get_max())
		elif action == "RIGHT":
			self.right = (1 - self.alpha)*self.right + self.alpha*(reward + gamma * cell.get_max())
		else:
			return 0

	#Returns the maximum of all the values for the cell
	def get_max(self):
		return max(self.up,self.down,self.right,self.left)

"""
Qgrid contains a list of cells for each state
"""
class Qgrid:
	def __init__(self,grid,gamma,beta,alpha):
		#list of cells 
		self.grid = []
		self.gamma = gamma
		self.beta = beta
		self.sizeI, self.sizeJ=grid.shape

		#Initialize cell 
		for i in range(self.sizeI):
			for j in range(self.sizeJ):
				cell = Cell(i,j,alpha)
				self.grid.append(cell)

	#Updates Qn for each state Qn-1
	def update_grid(self,trajectoriesXUR,trajectoriesXUX):
		for j in range(len(trajectoriesXUX)):
			i = 0
			for reward,state1,action in trajectoriesXUR[j]:
				#print(trajectoriesXUX[j][i])
				(x2,y2),(x1,y1),u = trajectoriesXUX[j][i]
				#print(reward)
				i = i+1
				cellNumberState1 = x1*self.sizeJ + y1
				cellNumberState2 = x2*self.sizeJ + y2

				self.grid[cellNumberState1].update_cell(u,reward,self.grid[cellNumberState2],self.gamma)

	#print the grid
	def print_grid(self):
		print("format :")
		print("cell number we read from left to right and up to down")
		print("Qvalue of cell Left-Up-Right-Down")
		for i in range(len(self.grid)):
			print("cell number :" + str(i))
			print(str(self.grid[i].left)+" "+str(self.grid[i].up)+" "+str(self.grid[i].right)+" "+str(self.grid[i].down))
			print("\n")

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
	
	policy = "RAND"

	vectorScores = np.zeros((size_x,steps))

	#plt.figure(figsize=(20,10))
	allxur = []
	allxux = []
	nTrajectories=100
	for k in range(nTrajectories):
		initialI = random.randrange(0, 5)
		initialJ = random.randrange(0, 5)
		game = Game(initialI,initialJ,array,discount,steps, beta,policy)
		game.start_game()
		allxur.append(game.rewardFromStateAndAction)
		allxux.append(game.state2FromState1AndAction)

	alpha = 0.05
	Q = Qgrid(array,discount,beta,alpha)
	Q.update_grid(allxur,allxux)
	Q.print_grid()