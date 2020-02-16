import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

from game import Game
class cell:
    def __init__(self):
        self.up = 0
        self.down = 0
        self.right = 0
        self.left = 0

        self.upVector = []
        self.downVector = []
        self.rightVector = []
        self.leftVector  = []

    def add_to_vector(value,move):
        if move == "LEFT":
            self.leftVector.append(value)
        elif move =="RIGHT":
            self.rightVector.append(value)
        elif move =="UP":
            self.upVector.append(value)
        elif move =="DOWN":
            self.downVector.append(value)
        return

    def update_cell(self):
        cumulatedSum = 0
        for i in range(len(upVector)):
            cumulatedSum = cumulatedSum+upVector[i]
        self.up = cumulatedSum/len(upVector)

        cumulatedSum = 0
        for i in range(len(downVector)):
            cumulatedSum = cumulatedSum+downVector[i]
        self.down = cumulatedSum/len(downVector)

        cumulatedSum = 0
        for i in range(len(rightVector)):
            cumulatedSum = cumulatedSum+rightVector[i]
        self.right = cumulatedSum/len(rightVector)

        cumulatedSum = 0
        for i in range(len(leftVector)):
            cumulatedSum = cumulatedSum+leftVector[i]
        self.left = cumulatedSum/len(leftVector)

    def get_value(self,move):
        if move == "LEFT":
            return self.left
        elif move == "RIGHT":
            return self.right
        elif move == "UP":
            return self.up
        elif move == "Down"
            return self.down
        else:
            return -1

class estimator:
    def __init__(self,sizeI,sizeJ,gamma):
        self.reward = []
        for i in range(sizeI):
            for j in range(sizeJ):
                self.reward.append(cell())
        self.sizeI = sizeI
        self.sizeJ = sizeJ
        self.gamma = gamma

    def update_rewards(self,rewardsFromStateAction):
        for j in rewardsFromStateAction:
            i = 0
            for r,x,u in rewardsFromStateAction:
                stateNumber = x[0]*sizeJ+x[1] 
                self.reward[stateNumber].add_to_vector(r,u*(1/self.gamma))

        for i in range(len(self.reward)):
            self.reward[i].update_cell()

    def estimated_value_state_action(self,state,action):
        i,j = state
        stateNumber = i*sizeJ+j
        self.reward[stateNumber].get_value(action)

def reward_state_action(x,u,trajectories):
	reward = 0
	counter = 0
	for trajectory in trajectories:
		for stateT,actionT,rewardT in trajectory:
			if stateT == x and actionT == u:
				reward = reward + rewardT
				counter = counter+1

	if counter >0:
		reward =  reward/counter
	else 
		reward = 0

	return reward

def proba_state1_action_state2(x1,u,x2,trajectories):
	x1uPairDetected = 0
	x1ux2TripletDetected = 0
	proportion = 0

	for trajectory in trajectories:
		for state1,action,state2 in trajectory:
			if state1 == x1 and action == u:
				x1uPairDetected = x1uPairDetected + 1
				if state2 == x2:
					x1ux2TripletDetected = x1ux2TripletDetected +1 

	if x1uPairDetected >= 0:
		proportion = x1ux2TripletDetected/x1uPairDetected

	return proportion


#Check if the position i,j is allowed in grid
def allowed_move(grid,i,j):
    sizeI, sizeJ=grid.shape
    if i >= 0 and i<sizeI and j >= 0 and j<sizeJ:
        return True
    return False

"""
Qgrid 

"""
class Qgrid:
	def __init__(self,grid,gamma,beta):
		#list of supercells 
		self.grid = []
		self.gamma = gamma
		self.beta = beta
		sizeI, sizeJ=grid.shape

		#Initialize superCell 
		for i in range(sizeI):
			for j in range(sizeJ):
				cell = SuperCell(grid,i,j)
				self.grid.append(cell)

		#Update superCell
		for i in range(sizeI):
			for j in range(sizeJ):
				self.grid[(i*sizeJ)+j].set_00_supercell(self.grid[0])
				if j>0:
					self.grid[(i*sizeJ)+j].set_left_supercell(self.grid[(i*sizeJ)+j-1])
				if j<sizeJ-1:
					self.grid[(i*sizeJ)+j].set_right_supercell(self.grid[(i*sizeJ)+j+1])
				if i>0:
					self.grid[(i*sizeJ)+j].set_up_supercell(self.grid[(i-1)*sizeJ+j])
				if i<sizeI-1:
					self.grid[(i*sizeJ)+j].set_down_supercell(self.grid[(i+1)*sizeJ+j])

	#Updates Qn for each state Qn-1
	def update_grid(self):
		for i in range(len(self.grid)):
			self.grid[i].update_inner_cells(self.gamma,self.beta)

		for i in range(len(self.grid)):
			self.grid[i].update_old_cells()

	def print_grid(self):
		print("format :")
		print("cell number we read from left to right and up to down")
		print("Qvalue of cell Left-Up-Right-Down")
		for i in range(len(self.grid)):
			print("cell number :" + str(i))
			print(str(self.grid[i].cellLeft)+" "+str(self.grid[i].cellUp)+" "+str(self.grid[i].cellRight)+" "+str(self.grid[i].cellDown))
			print("\n")

class SuperCell:
	def __init__(self,grid,positionI,positionJ):
		self.i = positionI
		self.j = positionJ

		i = positionI
		j = positionJ
		self.superCell00 = None
		self.superCellRight = None
		self.superCellLeft = None
		self.superCellUp = None
		self.superCellDown = None

		self.cell00 = grid[0][0]
		if allowed_move(grid,i,j-1):
			self.cellLeft = grid[i][j-1]
		else: 
			self.cellLeft = grid[i][j]
			self.superCellLeft = self

		if allowed_move(grid,i,j+1):
			self.cellRight = grid[i][j+1]
		else :
			self.cellRight = grid[i][j]
			self.superCellRight = self

		if allowed_move(grid,i+1,j):
			self.cellDown = grid[i+1][j]
		else :
			self.cellDown = grid[i][j]
			self.superCellDown = self

		if allowed_move(grid,i-1,j):
			self.cellUp = grid[i-1][j]
		else :
			self.cellUp = grid[i][j]
			self.superCellUp = self

		self.oldCellUp = self.cellUp.copy()
		self.oldCellDown = self.cellDown.copy()
		self.oldCellRight = self.cellRight.copy()
		self.oldCellLeft = self.cellLeft.copy()

		self.rewardUp = self.oldCellUp
		self.rewardDown = self.cellDown
		self.rewardRight = self.cellRight
		self.rewardLeft = self.cellLeft

	def set_00_supercell(self,superCell00):
		self.superCell00 = superCell00


	def set_up_supercell(self,superCellUp):
		if self.superCellUp == None :
			self.superCellUp = superCellUp

	def set_down_supercell(self,superCellDown):
		if self.superCellDown == None : 
			self.superCellDown = superCellDown

	def set_right_supercell(self,superCellRight):
		if self.superCellRight == None:
			self.superCellRight = superCellRight

	def set_left_supercell(self,superCellLeft):	
		if self.superCellLeft == None:
			self.superCellLeft = superCellLeft

	def get_maximum(self):
		return max(self.oldCellUp,self.oldCellDown,self.oldCellRight,self.oldCellLeft)

	def update_old_cells(self):
		self.oldCellUp = self.cellUp.copy()
		self.oldCellDown = self.cellDown.copy()
		self.oldCellRight = self.cellRight.copy()
		self.oldCellLeft = self.cellLeft.copy()

	def update_inner_cells(self,gamma,beta):
		self.cellUp = (1-beta)*self.rewardUp + beta*self.cell00 + (1-beta)*gamma * self.superCellUp.get_maximum() + beta*gamma*self.superCell00.get_maximum()
		self.cellDown = (1-beta)*self.rewardDown + beta*self.cell00 + (1-beta)*gamma * self.superCellDown.get_maximum()+ beta*gamma*self.superCell00.get_maximum()
		self.cellRight = (1-beta)*self.rewardRight + beta*self.cell00 + (1-beta)*gamma * self.superCellRight.get_maximum()+ beta*gamma*self.superCell00.get_maximum()
		self.cellLeft = (1-beta)*self.rewardLeft  + beta*self.cell00 +(1-beta)*gamma * self.superCellLeft.get_maximum()+ beta*gamma*self.superCell00.get_maximum()


if __name__ == '__main__':
	array = np.array([
	        [-3, 1, -5, 0, 19],
	        [6, 3, 8, 9, 10],
	        [5, -8, 4, 1, -8],
	        [6, -9, 4, 19, -5],
	        [-20, -17, -4, -3, 9]])

	#array = np.array([
	#		[0,-5,10],
	#		[0,0,0],
	#		])
	parser = argparse.ArgumentParser()
	parser.add_argument("--stochastic",help="Add stochasticity to the program",action="store_true")
	args = parser.parse_args()

	gamma = 0.99
	beta = 0.5
	N = 1000
	if args.stochastic ==False:
		beta = 0
	grid = Qgrid(array,gamma,beta)
	grid.print_grid()
	for i in range (N):
		#print(i)
		grid.update_grid()
	grid.print_grid()