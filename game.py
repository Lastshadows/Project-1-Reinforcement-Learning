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
    'position_x' is the variable representing the line the agent is on
    'position_y' is the variable representing the column the agent is on
    'grid' is a reward map of the game and represent the g variable in the domain given
    on the project guidelines
    'randomFactor' is the beta variable given in the guidelines and is used for the stochastic
    factor when chosing an action.

    """
    def __init__(self, position_x, position_y,grid, randomFactor):
        self.position_x = position_x
        self.position_y = position_y
        self.grid = grid
        self.score = 0
        self.randomFactor =  randomFactor

        if self.grid.allowed_position(self.position_x,self.position_y)==False:
            print("Bad initial position")

    """
    makes the agent move in a given direction.
    "UP", "DOWN", "LEFT", "RIGHT" and "NONE" are the only possible directions
    they make the agent move in the corresponding direction on the board.
    If the agent cannot move in the given position, it will simply stay still.
    """
    def move(self,direction):
        x = self.position_x
        y = self.position_y

        if direction == "UP" and self.grid.allowed_position(x-1,y)== True:
            self.position_x = x-1
        elif direction == "DOWN" and self.grid.allowed_position(x+1,y)== True:
            self.position_x = x+1
        elif direction == "RIGHT" and self.grid.allowed_position(x,y+1)== True:
            self.position_y = y+1
        elif direction == "LEFT" and self.grid.allowed_position(x,y-1)== True:
            self.position_y = y-1
        elif direction == "NONE" and self.grid.allowed_position(x,y)== True:
            self.position_y = y
            self.position_x = x

        return

    """
    the agent updates its own cumulated reward at the current time of the Game
    this score is updated by adding the relevant reward on the grid
    """
    def receive_reward(self):
        self.score = self.score + self.grid.get_reward(self.position_x,self.position_y)

    """
    return the current cumulated score of the agent
    """
    def get_score(self):
        return self.score


"""
This class represents the grid the agent evolves through. It has all the
informations needed to compute the instanteneous reward the agent gets at a
given time, and updates these based on the discount factor.
"""
class Grid:

    def __init__(self,rewards, discount_factor):
        self.rewards = rewards
        self.discount = discount_factor

    """
    updates the rewards at a given time across the board by multiplying it by
    the discount factor
    """
    def update_reward(self):
        self.rewards = self.rewards * self.discount # ?

    """
    returns the current reward for a given position on the Grid
    x is the rank if the desired position
    y is the column if the desired position

    """
    def get_reward(self,x,y):
        return self.rewards[x][y]

    """
    checks if a position is legal or not.
    x is the rank if the tested position
    y is the column if the tested position
    returns true if the position is legal, false otherwise
    """
    def allowed_position(self,x,y):
        size_x,size_y=self.rewards.shape
        if x >= 0 and x<size_x and y >= 0 and y<size_y:
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
    'position_x' is the variable representing the line the agent is on
    'position_y' is the variable representing the column the agent is on
    'rewards' is a reward map of the game and represent the g variable in the domain given
    on the project guidelines
    'discount' is the discount factor
    'steps' is the amount of turn a game takes to end
    'randomFactor' is the probability that the agent fails to move and stays
    still instead
    """
    def __init__(self,position_x,position_y,rewards,discount,steps, randomFactor):
        self.grid = Grid(rewards,discount)
        self.agent = Agent(position_x,position_y,self.grid, randomFactor)
        self.scores = np.zeros(steps)
        self.steps = steps

    def policy(self):
        self.agent.move("RIGHT")
        self.agent.receive_reward()
        self.grid.update_reward()

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
            self.policy()
