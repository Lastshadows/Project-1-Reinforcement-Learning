import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import random

from game import Game

class AgentQ:

    """
    Constructor.
    'positionI' is the variable representing the line the agent is on
    'positionJ' is the variable representing the column the agent is on
    'Qgrid' is  estimated Q matrix containing estimated value associated with state action
    'Gamma' is the Gamma exploration variable

    """
    def __init__(self, positionI, positionJ,Qgrid, gamma):

        self.positionI = positionI
        self.positionJ = positionJ

        self.qgrid = Qgrid

        self.score = 0
        self.gamma =  gamma

        self.currMove = "NONE"
        self.currReward = 0

        # first element is the reward, second is a tuple representing the state and the action that lead to the reward
        self.rewardFromStateAndAction = [] # r_x_u
        self.state2FromState1AndAction = []#  x' x u

        if self.grid.allowed_position(self.positionI,self.positionJ)==False:
            print("Bad initial position")

    """
    Updates agents state and grid after one move
    """
    def update_agent(self):
    	#agent position
        i, j = self.positionI,self.positionJ
        #agent moves
        self.move()
        #update reward state action vector
        unalteredReward = self.qgrid.get_unchanged_reward(i,j)
        self.rewardFromStateAndAction.append((unalteredReward, (i,j), self.currMove))

    """
    makes the agent move.
    Agent moves in the direction that maximises Qgrid for the given state
    with a probability gamma of doing a random exploration action
    """
    def move(self):
    	#Random number
        rand = random.uniform(0, 1)
        i = self.positionI
        j = self.positionJ

        #optimal direction to chose according to estimated Q
        direction = self.qgrid[i*self.sizeJ+j].highest_action()

        # exploration chose random direction
        if rand > (1-self.gamma):
            direction = random_direction()

        #update the current Move
        self.currMove = direction

        if (direction == "UP") and (self.grid.allowed_position(i-1,j) is True):
        	self.positionI = i-1
        elif (direction == "DOWN") and (self.grid.allowed_position(i+1,j) is True):
            self.positionI = i+1
        elif (direction == "RIGHT") and (self.grid.allowed_position(i,j+1) is True):
            self.positionJ = j+1
        elif (direction == "LEFT") and (self.grid.allowed_position(i,j-1) is True):
            self.positionJ = j-1

        #update state 2 <- state1 + action vector
        self.state2FromState1AndAction.append(((self.positionI,self.positionJ), (i,j), self.currMove))

        return

    # selects a random direction to move to
    def random_direction(self):
        seed = random.uniform(0, 1)
        if seed <= 0.25:
            direction = "RIGHT"
        elif seed <= 0.5 :
            direction = "LEFT"
        elif seed <= 0.75:
            direction = "UP"
        else :
            direction = "DOWN"
        return direction

    # the agent updates its own cumulated reward at the current time of the Game
    # this score is updated by adding the relevant reward on the grid
    def receive_reward(self):
        self.score = self.score + self.grid.get_reward(self.positionI,self.positionJ)


    # return the current cumulated score of the agent
    def get_score(self):
        return self.score


    # return the current position of the agent
    def get_position(self):
        return (self.positionI , self.positionJ)

    # returns the last move done by the agent
    def get_curr_move(self):
        return self.currMove

    # returns the last move done by the agent
    def get_curr_reward(self):
        return self.currReward

    def get_state2_state1_action(self):
    	return self.state2FromState1AndAction

    # returns the initial reward the agent gets for doing action u from state x .
    # discount factor is not accounted for
    def get_reward_state_action(self):
        return self.rewardFromStateAndAction

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

	def highest_action(self):
		if self.up >= self.right and self.up >= self.down and self.up >= self.left:
			return "UP"

		if self.right >= self.up and self.right >= self.down and self.right >= self.left:
			return "RIGHT"

		if self.down >= self.up and self.down >= self.right and self.down >= self.left:
			return "DOWN"

		if self.left >= self.up and self.left >= self.down and self.left >= self.right:
			return "LEFT"

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
	def __init__(self,grid,gamma,alpha):
		#list of cells
		self.reward = grid
		self.grid = []
		self.gamma = gamma
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

	def optimal_path(self):
		path = []
		for i in range(self.sizeI):
			moves = []
			for j in range(self.sizeJ):
				move = self.grid[i*self.sizeJ+j].highest_action()
				path.append(move)
				moves.append(move)
			print(moves)
			print("\n")

	def get_reward(self,i,j):
		return self.reward[i][j]


	#print the grid
	def print_grid(self):
		print("format :")
		print("cell number we read from left to right and up to down")
		print("Qvalue of cell Left-Up-Right-Down")
		for i in range(len(self.grid)):
			print("cell number :" + str(i))
			print(str(self.grid[i].left)+" "+str(self.grid[i].up)+" "+str(self.grid[i].right)+" "+str(self.grid[i].down))
			print("\n")

"""
This class represents the Game
it has the methods needed to initialize a game, and to run it.
It can also store the scores of each step of the game and return them
"""
class QGame:
    """
    initialize a Game
    'positionI' is the variable representing the line the agent is on
    'positionJ' is the variable representing the column the agent is on
    'rewards' is a reward map of the game and represent the g variable in the domain given
    on the project guidelines
    'discount' is the discount factor
    'steps' is the amount of turn a game takes to end
    """
    def __init__(self,positionI,positionJ,rewards,discount,steps,gamma):
        self.grid = reward
        self.agent = AgentQ(positionI,positionJ,self.grid, gamma)
        self.scores = np.zeros(steps)

        self.iPositions= np.zeros(steps)
        self.jPositions = np.zeros(steps)

        self.moves = []
        self.rewards = []

        self.steps = steps

    # returns all the scores of the game (one for every step) under the form of a
    # one dimensional table
    def get_scores(self):
        return self.scores

    def greedy_game(self):
        return 0

if __name__ == '__main__':
    print("in the main")
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
    policy = "RAND"
    if args.stochastic ==False:
    	beta = 0

    allxur = []
    allxux = []
    nTrajectories=100
    for k in range(nTrajectories):
    	initialI = random.randrange(0, 5)
    	initialJ = random.randrange(0, 5)
    	game = Game(initialI,initialJ,array,discount,steps, beta,policy,False)
    	game.start_game()
    	allxur.append(game.rewardFromStateAndAction)
    	allxux.append(game.state2FromState1AndAction)

    alpha = 0.05
    Q = Qgrid(array,discount,alpha)
    Q.update_grid(allxur,allxux)
    Q.print_grid()
    Q.optimal_path()


    #SECTION 2 :
    ##EXPERIENCE 1 :
    ###TO DONE NORMALLY ONLY FCT CALL
