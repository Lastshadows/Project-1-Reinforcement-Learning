import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

from game import Game

# if __name__ == '__main__':

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
	def __init__(self,grid,gamma):
		#list of supercells 
		self.grid = []
		self.gamma = gamma

		sizeI, sizeJ=grid.shape

		#Initialize superCell 
		for i in range(sizeI):
			for j in range(sizeJ):
				cell = SuperCell(grid,i,j)
				self.grid.append(cell)

		#Update superCell
		for i in range(sizeI):
			for j in range(sizeJ):
				if j>0:
					self.grid[(i*sizeJ)+j].set_left_supercell(self.grid[(i*sizeJ)+j-1])
					print("Left of element "+str(i)+" "+str(j)+" : "+str(i)+" "+str(j-1))
				print("cell Left reward: "+str(self.grid[(i*sizeJ)+j].cellLeft))
				if j<sizeJ-1:
					self.grid[(i*sizeJ)+j].set_right_supercell(self.grid[(i*sizeJ)+j+1])
					print("Right of element "+str(i)+" "+str(j)+" : "+str(i)+" "+str(j+1))
				print("cell right reward: "+str(self.grid[(i*sizeJ)+j].cellRight))
				if i>0:
					self.grid[(i*sizeJ)+j].set_up_supercell(self.grid[(i-1)*sizeJ+j])
					print("Up of element "+str(i)+" "+str(j)+" : "+str(i-1)+" "+str(j))
				print("cell up reward: "+str(self.grid[(i*sizeJ)+j].cellUp))
				if i<sizeI-1:
					self.grid[(i*sizeJ)+j].set_down_supercell(self.grid[(i+1)*sizeJ+j])
					print("Down of element "+str(i)+" "+str(j)+" : "+str(i+1)+" "+str(j))
				print("cell down reward: "+str(self.grid[(i*sizeJ)+j].cellDown))
				print("\n\n")

	#Updates Qn for each state Qn-1
	def update_grid(self):
		print("HERRRRE \n\n")
		for i in range(len(self.grid)):
			self.grid[i].update_inner_cells(self.gamma)
			print(i)
			print(str(self.grid[i].cellUp)+" "+str(self.grid[i].cellDown)+" "+str(self.grid[i].cellRight)+" "+str(self.grid[i].cellLeft))
			print("\n\n")



class SuperCell:
	def __init__(self,grid,positionI,positionJ):
		self.i = positionI
		self.j = positionJ

		i = positionI
		j = positionJ

		self.superCellRight = None
		self.superCellLeft = None
		self.superCellUp = None
		self.superCellDown = None

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
		return max(self.cellUp,self.cellDown,self.cellRight,self.cellLeft)

	def update_inner_cells(self,gamma):
		self.cellUp = self.cellUp + gamma * self.superCellUp.get_maximum()
		self.cellDown = self.cellDown + gamma * self.superCellDown.get_maximum()
		self.cellRight = self.cellRight + gamma * self.superCellRight.get_maximum()
		self.cellLeft = self.cellLeft  + gamma * self.superCellLeft.get_maximum()

array = np.array([
        [-3, 1, -5, 0, 19],
        [6, 3, 8, 9, 10],
        [5, -8, 4, 1, -8],
        [6, -9, 4, 19, -5],
        [-20, -17, -4, -3, 9]])

gamma = 0.99

grid = Qgrid(array,gamma)
grid.update_grid()

# 	parser = argparse.ArgumentParser(description='The default policy is RIGHT')
# 	parser.add_argument("--policy",type = str,
# 		help="chose a policy between always :  RIGHT LEFT UP DOWN \n or for random : RAND")

# 	parser.add_argument("--stochastic",help="Add stochasticity to the program",action="store_true")

# 	args = parser.parse_args()

# 	policy = "RIGHT"
# 	if args.policy:
# 		if args.policy=="LEFT" or args.policy=="RIGHT" or args.policy=="UP" or args.policy=="DOWN" or args.policy =="RAND":
# 			policy = args.policy
# 		else : 
# 			print("UNKNOWN policy: "+args.policy)
# 			print("TRY : RIGHT - LEFT - UP - DOWN - RAND")
# 			sys.exit(0)

# 	distribution = proba_distribution(policy)
# 	if distribution == -1:
# 		sys.exit(0)

# 	stats(array,distribution,beta,args.stochasticity)