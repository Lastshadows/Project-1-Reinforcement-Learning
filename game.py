import numpy as np
import matplotlib.pyplot as plt

class Agent:
    """
    this is a test for git
    this is another test
    """
    def __init__(self, position_x, position_y,grid, randomFactor):
        self.position_x = position_x
        self.position_y = position_y
        self.grid = grid
        self.score = 0
        self.randomFactor =  randomFactor

        if self.grid.allowed_position(self.position_x,self.position_y)==False:
            print("Bad initial position")

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

    def receive_reward(self):
        self.score = self.score + self.grid.get_reward(self.position_x,self.position_y)

    def get_score(self):
        return self.score


class Grid:
    def __init__(self,rewards, discount_factor):
        self.rewards = rewards
        self.discount = discount_factor

    def update_reward(self):
        self.rewards = self.rewards * self.discount # ?

    def get_reward(self,x,y):
        return self.rewards[x][y]

    def allowed_position(self,x,y):
        size_x,size_y=self.rewards.shape
        if x >= 0 and x<size_x and y >= 0 and y<size_y:
            return True
        return False

class Game:
    def __init__(self,position_x,position_y,rewards,discount,steps, randomFactor):
        self.grid = Grid(rewards,discount)
        self.agent = Agent(position_x,position_y,self.grid, randomFactor)
        self.scores = np.zeros(steps)
        self.steps = steps

    def policy(self):
        self.agent.move("RIGHT")
        self.agent.receive_reward()
        self.grid.update_reward()

    def get_scores(self):
        return self.scores

    def start_game(self):
        for i in range(self.steps):
            self.scores[i] = self.agent.get_score()
            self.policy()
