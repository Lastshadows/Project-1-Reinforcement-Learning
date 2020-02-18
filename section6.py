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
    'epsilon' is the epsilon exploration variable


    """
    def __init__(self, positionI, positionJ,Qgrid, beta, epsilon):

        self.positionI = positionI
        self.positionJ = positionJ

        self.qgrid = Qgrid

        self.score = 0
        self.epsilon =  epsilon
        self.beta = beta

        self.currMove = "NONE"
        self.currReward = 0

        # first element is the reward, second is a tuple representing the state and the action that lead to the reward
        self.rewardFromStateAndAction = ()# r_x_u
        self.state2FromState1AndAction = ()#  x' x u

        if self.qgrid.allowed_position(self.positionI,self.positionJ)==False:
            print("Bad initial position")

    def reset(self,i,j):
        self.positionI = i
        self.positionJ = j
        self.score = 0
        self.currMove = "NONE"
        self.rewardFromStateAndAction = ()
        self.state2FromState1AndAction = ()


    """
    Updates agents state and grid after one move
    """
    def update_agent(self):
        #agent position
        i, j = self.positionI,self.positionJ
        #agent moves
        self.move()
        self.receive_reward()
        #update reward state action vector
        unalteredReward = self.qgrid.get_reward(i,j)
        self.rewardFromStateAndAction = ((unalteredReward, (i,j), self.currMove))

    """
    makes the agent move.
    Agent moves in the direction that maximises Qgrid for the given state
    with a probability epsilon of doing a random exploration action
    """
    def move(self):
        #Random number
        rand = random.uniform(0, 1)
        i = self.positionI
        j = self.positionJ

        #optimal direction to chose according to estimated Q
        direction = self.qgrid.get_cell(i,j).best_action()

        # exploration chose random direction
        if rand > (1-self.epsilon):
            direction = self.random_direction()

        #update the current Move
        self.currMove = direction

        if direction == "UP" and self.qgrid.allowed_position(i-1,j)== True:
            self.positionI = i-1
        elif direction == "DOWN" and self.qgrid.allowed_position(i+1,j)== True:
            self.positionI = i+1
        elif direction == "RIGHT" and self.qgrid.allowed_position(i,j+1)== True:
            self.positionJ = j+1
        elif direction == "LEFT" and self.qgrid.allowed_position(i,j-1)== True:
            self.positionJ = j-1

        # we have a chance beta to get teleported back to the (0,0) cell
        if rand < self.beta:
            self.positionI = 0
            self.positionJ = 0

        #update state 2 <- state1 + action vector
        self.state2FromState1AndAction = ((self.positionI,self.positionJ), (i,j), self.currMove)

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
        self.score = self.score + self.qgrid.get_reward(self.positionI,self.positionJ)
        #print(self.score)
        #print(self.qgrid.get_reward(self.positionI,self.positionJ))


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

    def best_action(self):
        if self.up >= self.right and self.up >= self.down and self.up >= self.left:
            return "UP"

        if self.right >= self.up and self.right >= self.down and self.right >= self.left:
            return "RIGHT"

        if self.down >= self.up and self.down >= self.right and self.down >= self.left:
            return "DOWN"

        if self.left >= self.up and self.left >= self.down and self.left >= self.right:
            return "LEFT"

    # Updates the cell corresponding cell value given the action, the reward, the cell reached
    # and the value of gamma
    def update_cell(self,action,reward,cell,gamma,t):
        if action == "UP":
            self.up = (1-self.alpha(t))*self.up + self.alpha(t)*(reward + gamma * cell.get_max())
        elif action == "DOWN":
            self.down = (1 - self.alpha(t))*self.down + self.alpha(t)*(reward + gamma * cell.get_max())
        elif action == "LEFT":
            self.left = (1 - self.alpha(t))*self.left + self.alpha(t)*(reward + gamma * cell.get_max())
        elif action == "RIGHT":
            self.right = (1 - self.alpha(t))*self.right + self.alpha(t)*(reward + gamma * cell.get_max())
        else:
            return 0

    #Returns the maximum of all the values for the cell
    def get_max(self):
        return max(self.up,self.down,self.right,self.left)

"""
Qgrid contains a list of cells for each state
"""
class Qgrid:
    def __init__(self,grid,alpha,gamma):
        #list of cells
        self.reward = grid
        self.grid = []
        self.gamma = gamma
        self.sizeI, self.sizeJ=grid.shape
        self.t = 0

        #Initialize cell
        for i in range(self.sizeI):
            for j in range(self.sizeJ):
                cell = Cell(i,j,alpha)
                self.grid.append(cell)

    def allowed_position(self,i,j):
        if i >= 0 and i<self.sizeI and j >= 0 and j<self.sizeJ:
            return True
        return False

    def get_cell(self,i,j):
        if self.allowed_position(i,j):
            return self.grid[i*self.sizeJ+j]


    #Updates Qn for each state Qn-1
    def update_grid_trajectories(self,trajectoriesXUR,trajectoriesXUX):
        for j in range(len(trajectoriesXUX)):
            i = 0
            for reward,state1,action in trajectoriesXUR[j]:
                #print(trajectoriesXUX[j][i])
                (x2,y2),(x1,y1),u = trajectoriesXUX[j][i]
                #print(reward)
                i = i+1
                cellNumberState1 = x1*self.sizeJ + y1
                cellNumberState2 = x2*self.sizeJ + y2

                self.grid[cellNumberState1].update_cell(u,reward,self.grid[cellNumberState2],self.gamma,self.t)
                self.t = self.t+1


    def update_grid_single(self,xur,xux):
        (x2,y2),(x1,y1),u = xux
        reward,state1,u = xur
        cellNumberState1 = x1*self.sizeJ + y1
        cellNumberState2 = x2*self.sizeJ + y2
        self.grid[cellNumberState1].update_cell(u,reward,self.grid[cellNumberState2],self.gamma,self.t)
        self.t = self.t+1


    def optimal_path(self):
        path = []
        for i in range(self.sizeI):
            moves = []
            for j in range(self.sizeJ):
                move = self.grid[i*self.sizeJ+j].best_action()
                reward = self.grid[i*self.sizeJ+j].get_max()
                print("J estimated position "+str(i)+" "+str(j)+" : " + str(reward))
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
        print("Qvalue of cell Left-Up-Right-Down\n")
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
    'steps' is the amount of turn a game takes to end
    'alpha' is the learning rate
    'beta' is the chance we have to be teleported back to cell (0,0)
    'epsilon' is the exploration rate
    'gamma' is the discount factor
    """
    def __init__(self,positionI,positionJ,rewards,steps,alpha, beta,epsilon, gamma):
        self.grid = rewards
        self.qgrid=Qgrid(self.grid,alpha,gamma)
        self.agent = AgentQ(positionI,positionJ,self.qgrid,beta, epsilon)
        self.scores = np.zeros(steps)

        self.initialI = positionI
        self.initialJ = positionJ

        self.iPositions= np.zeros(steps)
        self.jPositions = np.zeros(steps)

        self.moves = []
        self.rewards = []

        self.steps = steps

    # returns all the scores of the game (one for every step) under the form of a
    # one dimensional table
    def get_scores(self):
        return self.scores

    def greedy_game(self,n,m):
        trajectoriesXUR= []
        trajectoriesXUX= []
        #NUMBER OF EPISODES
        for i in range(n):
            self.agent.reset(self.initialI,self.initialJ)
            currentTrajectoriesXUX = []
            currentTrajectoriesXUR = []
            scores = np.zeros(m)
            #A TRAJECTORY OF m TRANSITIONS
            for j in range(m):

                scores[j]=self.agent.get_score()
                self.agent.update_agent()

                addedTrajectoryXUX = self.agent.get_state2_state1_action()
                addedTrajectoryXUR = self.agent.get_reward_state_action()

                #print(addedTrajectoryXUR)
                #print(addedTrajectoryXUX)

                currentTrajectoriesXUX.append(addedTrajectoryXUX)
                currentTrajectoriesXUR.append(addedTrajectoryXUR)

                self.qgrid.update_grid_single(addedTrajectoryXUR,addedTrajectoryXUX)
            if i < 10:
                plt.plot(scores)
            trajectoriesXUX.append(currentTrajectoriesXUX)
            trajectoriesXUR.append(currentTrajectoriesXUR)

        plt.savefig('scores.png')



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

    steps = 1000
    nTrajectories=1000
    policy = "RAND"

    gamma =  0.99
    beta = 0.5
    alpha = 0.05
    epsilon = 0.25


    if args.stochastic ==False:
        beta = 0

    allxur = []
    allxux = []

    alpha1 = lambda t : 0.05
    alpha2 = lambda t : pow(0.8,t)*0.05

    for k in range(nTrajectories):
        initialI = random.randrange(0, 5)
        initialJ = random.randrange(0, 5)
        game = Game(initialI,initialJ,array,gamma,steps, beta,policy, False)
        game.start_game()
        allxur.append(game.rewardFromStateAndAction)
        allxux.append(game.state2FromState1AndAction)

    Q = Qgrid(array,alpha1,gamma)
    Q.update_grid_trajectories(allxur,allxux)
    Q.print_grid()
    Q.optimal_path()

    #SECTION 2 :
    ##EXPERIENCE 1 :

    qgame = QGame(0,3,array,steps,alpha1,beta, epsilon, gamma)
    qgame.greedy_game(100,1000)

    #EXPERIENCE 2 : 
    qgame = QGame(0,3,array,steps,alpha2,beta, epsilon, gamma)
    qgame.greedy_game(100,1000)