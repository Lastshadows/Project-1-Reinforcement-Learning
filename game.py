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
    def __init__(self, positionI, positionJ,grid, randomFactor,policy ):

        self.positionI = positionI
        self.positionJ = positionJ

        self.grid = grid
        self.initialGrid = grid.unchangedRewards

        self.score = 0
        self.randomFactor =  randomFactor
        self.policyType = policy

        self.currMove = "NONE"
        self.currReward = 0
        # first element is the reward, second is a tuple represetning the state and the action that lead to the reward
        self.rewardFromStateAndAction = ()
        self.state2FromState1AndAction = ()

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


    # according to the set policy, return the corresponding policy
    def policy(self):
        if self.policyType == 0:
            return self.policy_rand()
        elif self.policyType == 1:
            return self.policy_right()
        elif self.policyType == 2:
            return self.policy_left()
        elif self.policyType == 3:
            return self.policy_up()
        elif self.policyType == 4:
            return self.policy_down()
        else :
            return

    # selects a random direction to move to and updates the rewards
    def policy_rand(self):
        seed = np.random.rand()
        if seed <= 0.25:
            self.currMove = "RIGHT"
        elif seed <= 0.5 :
            self.currMove = "LEFT"
        elif seed <= 0.75:
            self.currMove = "UP"
        else :
            self.currMove = "DOWN"

        # computation of r(x,u)
        i, j = self.positionI,self.positionJ
        self.move(self.currMove)
        unalteredReward = self.grid.get_unchanged_reward(i,j)
        self.rewardFromStateAndAction = ( (unalteredReward, (i,j), self.currMove) )

        # saving data for computation of p(x'|x,u)
        i2,j2 =  self.positionI,self.positionJ
        self.state2FromState1AndAction = ((i2,j2), (i,j), self.currMove)

        self.receive_reward()
        self.currReward =  self.grid.get_reward(self.positionI,self.positionJ)
        self.grid.update_reward()


    #makes the agent move to the right, and updates the
    #rewards grid.
    def policy_right(self):

        self.currMove = "RIGHT"

        # computation of r(x,u)
        i, j = self.positionI,self.positionJ
        self.move(self.currMove)
        unalteredReward = self.grid.get_unchanged_reward(i,j)
        self.rewardFromStateAndAction = ( (unalteredReward, (i,j), self.currMove) )

        #saving data for computation of p(x'|x,u)
        i2,j2 =  self.positionI,self.positionJ
        self.state2FromState1AndAction = ((i2,j2), (i,j), self.currMove)

        # update of the agent position, the cumulated score and the reward grid
        self.receive_reward()
        self.currReward =  self.grid.get_reward(self.positionI,self.positionJ)
        self.grid.update_reward()

    """
    makes the agent move to the left, and updates the
    rewards grid.
    """
    def policy_left(self):
        self.currMove = "LEFT"
        self.move(self.currMove)
        self.receive_reward()
        self.currReward =  self.grid.get_reward(self.positionI,self.positionJ)
        self.grid.update_reward()

    """
    makes the agent move to the up, and updates the
    rewards grid.
    """
    def policy_up(self):
        self.currMove = "UP"
        self.move(self.currMove)
        self.receive_reward()
        self.currReward =  self.grid.get_reward(self.positionI,self.positionJ)
        self.grid.update_reward()

    """
    makes the agent move to the down, and updates the
    rewards grid.
    """
    def policy_down(self):
        
        self.currMove = "DOWN"
        self.move(self.currMove)
        self.receive_reward()
        self.currReward =  self.grid.get_reward(self.positionI,self.positionJ)
        self.grid.update_reward()


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

    # returns the initial reward the agent gets for doing action u from state x .
    # discount factor is not accounted for
    def get_r_x_u(self):
        return self.rewardFromStateAndAction

"""
This class represents the grid the agent evolves through. It has all the
informations needed to compute the instanteneous reward the agent gets at a
given time, and updates these based on the discount factor.
"""
class Grid:

    def __init__(self,rewards, discountFactor):
        self.rewards = rewards
        self.unchangedRewards = rewards
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

    def get_unchanged_reward(self, i,j):
        return self.unchangedRewards[i][j]

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
    def __init__(self,positionI,positionJ,rewards,discount,steps, randomFactor,policy):
        self.grid = Grid(rewards,discount)
        policyType = self.policy_definition(policy)
        self.agent = Agent(positionI,positionJ,self.grid, randomFactor,policyType)
        self.scores = np.zeros(steps)
        self.iPositions= np.zeros(steps)
        self.jPositions = np.zeros(steps)
        self.moves = []
        self.rewards = []
        self.trajectory = []
        self.rewardFromStateAndAction = []
        self.state2FromState1AndAction = []
        self.steps = steps

    def policy_definition(self, direction):
        policyType = -1
        if direction == "RAND":
            policyType = 0
        elif direction == "RIGHT":
            policyType = 1
        elif direction == "LEFT":
            policyType = 2
        elif direction == "UP":
            policyType = 3
        elif direction == "DOWN":
            policyType = 4
        else:
            print("error unknown direction: "+direction)
        return policyType

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

            self.moves.append(self.agent.get_curr_move())
            self.rewards.append(self.agent.get_curr_reward())

            self.rewardFromStateAndAction.append(self.agent.rewardFromStateAndAction)
            self.state2FromState1AndAction.append(self.agent.state2FromState1AndAction)

        # zip together the i and j vectors to have a general trajectory list
        self.trajectory =  list(zip(self.iPositions, self.jPositions))
        self.trajectory = list(zip(self.trajectory, self.moves))
        self.trajectory = list(zip(self.trajectory, self.rewards))
