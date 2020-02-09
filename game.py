import numpy as np
import matplotlib.pyplot as plt

"""
The agent class represents the artificial autonomous intelligent agent. It
possesses all the data needed to take decisions based on the state it is in,
and possesses methods allowing him to act upon these decisions
"""
class Agent:

    """
    Constructor.
    'positionI' is the variable representing the line the agent is on
    'positionJ' is the variable representing the column the agent is on
    'grid' is a reward map of the game and represent the g variable in the domain given
    on the project guidelines
    'randomFactor' is the beta variable given in the guidelines and is used for the stochastic
    factor when chosing an action.

    """
    def __init__(self, positionI, positionJ,grid, randomFactor,policy):
        self.positionI = positionI
        self.positionJ = positionJ
        self.grid = grid
        self.score = 0
        self.randomFactor =  randomFactor
        self.policyType = policy

        if self.grid.allowed_position(self.positionI,self.positionJ)==False:
            print("Bad initial position")

    """
    makes the agent move in a given direction.
    "UP", "DOWN", "LEFT", "RIGHT" and "NONE" are the only possible directions
    they make the agent move in the corresponding direction on the board.
    If the agent cannot move in the given position, it will simply stay still.
    """
    def move(self,direction):
        i = self.positionI
        j = self.positionJ

        if direction == "UP" and self.grid.allowed_position(i-1,j)== True:
            self.positionI = i-1
        elif direction == "DOWN" and self.grid.allowed_position(i+1,j)== True:
            self.positionI = i+1
        elif direction == "RIGHT" and self.grid.allowed_position(i,j+1)== True:
            self.positionJ = j+1
        elif direction == "LEFT" and self.grid.allowed_position(i,j-1)== True:
            self.positionJ = j-1
        elif direction == "NONE" and self.grid.allowed_position(i,j)== True:
            self.positionJ = j
            self.positionI = i

        return


    def policy(self):
        if self.policyType == 0:
            return self.policy_right()
        elif self.policyType == 1: 
            return self.policy_rand()
        else :
            return 

    def policy_rand(self):
        seed = np.random.random_sample()
        #print(seed)
        if seed >= 0.25:
            self.move("RIGHT")
        elif seed >= 0.5 :
            self.move("LEFT")
        elif seed >= 0.75:
            self.move("UP")
        else :
            self.move("DOWN")

        self.receive_reward()
        self.grid.update_reward()

    """
    makes the agent move from on tile according to a policy, and updates the
    rewards grid.
    """
    def policy_right(self):
        self.move("RIGHT")
        self.receive_reward()
        self.grid.update_reward()


    """
    the agent updates its own cumulated reward at the current time of the Game
    this score is updated by adding the relevant reward on the grid
    """
    def receive_reward(self):
        self.score = self.score + self.grid.get_reward(self.positionI,self.positionJ)

    """
    return the current cumulated score of the agent
    """
    def get_score(self):
        return self.score

    """
    return the current position of the agent
    """
    def get_position(self):
        return (self.positionI , self.positionJ)


"""
This class represents the grid the agent evolves through. It has all the
informations needed to compute the instanteneous reward the agent gets at a
given time, and updates these based on the discount factor.
"""
class Grid:

    def __init__(self,rewards, discountFactor):
        self.rewards = rewards
        self.discount = discountFactor

    """
    updates the rewards at a given time across the board by multiplying it by
    the discount factor
    """
    def update_reward(self):
        self.rewards = self.rewards * self.discount # ?

    """
    returns the current reward for a given position on the Grid
    i is the rank of the desired position
    j is the column of the desired position

    """
    def get_reward(self,i,j):
        return self.rewards[i][j]

    """
    checks if a position is legal or not.
    i is the rank if the tested position
    j is the column if the tested position
    returns true if the position is legal, false otherwise
    """
    def allowed_position(self,i,j):
        sizeI, sizeJ=self.rewards.shape
        if i >= 0 and i<sizeI and j >= 0 and j<sizeJ:
            return True
        return False

"""
This class represents the Game
it has the methods needed to initialize a game, and to run it.
It can also store the scores of each step of the game and return them
"""
class Game:
    """
    initialize a Game
    'positionI' is the variable representing the line the agent is on
    'positionJ' is the variable representing the column the agent is on
    'rewards' is a reward map of the game and represent the g variable in the domain given
    on the project guidelines
    'discount' is the discount factor
    'steps' is the amount of turn a game takes to end
    'randomFactor' is the probability that the agent fails to move and stays
    still instead
    """
    def __init__(self,positionI,positionJ,rewards,discount,steps, randomFactor):
        self.grid = Grid(rewards,discount)
        self.agent = Agent(positionI,positionJ,self.grid, randomFactor,0)
        self.scores = np.zeros(steps)
        self.iPositions= np.zeros(steps)
        self.jPositions = np.zeros(steps)
        self.trajectory = []
        self.steps = steps

    """
    returns all the scores of the game (one for every step) under the form of a
    one dimensional table
    """
    def get_scores(self):
        return self.scores

    """
    launches a game
    """
    def start_game(self):
        for i in range(self.steps):

            self.scores[i] = self.agent.get_score()
            self.iPositions[i], self.jPositions[i] =  self.agent.get_position()
            self.agent.policy()

        # zip together the i and j vectors to have a general trajectory list
        self.trajectory =  list(zip(self.iPositions, self.jPositions))
