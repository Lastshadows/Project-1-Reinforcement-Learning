import numpy as np
import matplotlib.pyplot as plt
import random

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
    'beta' is the beta variable given in the guidelines and is used for the stochastic
    factor when chosing an action.

    """
    def __init__(self, positionI, positionJ,grid, beta,policy ):

        self.positionI = positionI
        self.positionJ = positionJ

        self.grid = grid
        self.initialGrid = grid.unchangedRewards

        self.score = 0
        self.beta =  beta
        self.policyType = policy

        self.currMove = "NONE"
        self.currReward = 0
        # first element is the reward, second is a tuple represetning the state and the action that lead to the reward
        self.rewardFromStateAndAction = () # r_x_u
        self.state2FromState1AndAction = ()#  x' x u

        if self.grid.allowed_position(self.positionI,self.positionJ)==False:
            print("Bad initial position")

    """
    Updates agents state and grid after one move
    """
    def update_agent(self):
        # computation of r(x,u)
        i, j = self.positionI,self.positionJ
        self.move(self.currMove)
        unalteredReward = self.grid.get_unchanged_reward(i,j)
        self.rewardFromStateAndAction = ( (unalteredReward, (i,j), self.currMove) )

        # saving data for computation of p(x'|x,u)
        i2,j2 =  self.positionI,self.positionJ
        self.state2FromState1AndAction = ((i2,j2), (i,j), self.currMove)

        # updating remaining data
        self.receive_reward()
        self.currReward =  self.grid.get_reward(self.positionI,self.positionJ)
        self.grid.update_reward()

    """
    makes the agent move in a given direction.
    "UP", "DOWN", "LEFT", "RIGHT" and "RESET" are the only possible directions
    they make the agent move in the corresponding direction on the board.
    If the agent cannot move in the given position, it will simply stay still.
    """
    def move(self,direction):

        rand = random.uniform(0, 1)
        i = self.positionI
        j = self.positionJ

        # if unlucky, goes back to tile (0,0)
        if rand > (1-self.beta):
            direction = "RESET"

        if direction == "UP" and self.grid.allowed_position(i-1,j)== True:
            self.positionI = i-1
        elif direction == "DOWN" and self.grid.allowed_position(i+1,j)== True:
            self.positionI = i+1
        elif direction == "RIGHT" and self.grid.allowed_position(i,j+1)== True:
            self.positionJ = j+1
        elif direction == "LEFT" and self.grid.allowed_position(i,j-1)== True:
            self.positionJ = j-1
        elif direction == "RESET" and self.grid.allowed_position(i,j)== True:
            self.positionJ = 0
            self.positionI = 0

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
        seed = random.uniform(0, 1)
        if seed <= 0.25:
            self.currMove = "RIGHT"
        elif seed <= 0.5 :
            self.currMove = "LEFT"
        elif seed <= 0.75:
            self.currMove = "UP"
        else :
            self.currMove = "DOWN"
        self.update_agent()

    #makes the agent move to the right, and updates the
    #rewards grid.
    def policy_right(self):
        self.currMove = "RIGHT"
        self.update_agent()


    # makes the agent move to the left, and updates the
    # rewards grid.

    def policy_left(self):
        self.currMove = "LEFT"
        self.update_agent()

    """
    makes the agent move to the up, and updates the
    rewards grid.
    """
    def policy_up(self):
        self.currMove = "UP"
        self.update_agent()

    """
    makes the agent move to the down, and updates the
    rewards grid.
    """
    def policy_down(self):
        self.currMove = "DOWN"
        self.update_agent()


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

    # takes a state x, an action u and a proba beta as arguments
    # x is a tuple representing the position i,j on the board
    # u is one of the following : "UP", "DOWN", "RIGHT", "LEFT"
    # beta is the probability the agent has to  go back to the (0,0) state when he tries to move  and should be between 0 and 1
    # this functions returns the expected reward at a given time and state in the game for a given move
    def compute_r_x_u(self,x,u,beta):

        resetChance = beta
        normalMoveChance =  1 - beta
        i,j = x
        direction =  u

        # if the movement was allowed, the i and j indexes are changed. Otherwise we hit a wall -> i j dont change we stay there
        if direction == "UP" and self.allowed_position(i-1,j)== True:
            i= i-1
        elif direction == "DOWN" and self.grid.allowed_position(i+1,j)== True:
            i = i+1
        elif direction == "RIGHT" and self.grid.allowed_position(i,j+1)== True:
            j = j+1
        elif direction == "LEFT" and self.grid.allowed_position(i,j-1)== True:
            j = j-1

        expectedReward = (self.rewards[i][j] * normalMoveChance) + (self.rewards[0][0]*resetChance)

        return expectedReward

    # this fucntion returns the proba to get to state x' (xprime) from state x while doing action u
    # u is one of the following : "UP", "DOWN", "RIGHT", "LEFT"
    # beta is the proba to go back to (0,0) instead of doing the action u
    def compute_proba_xprime_x_u(self, xprime, x, u, beta):

        resetChance = beta
        normalMoveChance =  1 - beta
        i,j = x # current state
        direction =  u
        i_prime, j_prime = xprime # x' state
        i_moved, j_moved =  x

        # computation of the state we would reach if u was applied to x
        if direction == "UP" and self.allowed_position(i-1,j)== True:
            i_moved = i-1
        elif direction == "DOWN" and self.grid.allowed_position(i+1,j)== True:
            i_moved = i+1
        elif direction == "RIGHT" and self.grid.allowed_position(i,j+1)== True:
            j_moved = j+1
        elif direction == "LEFT" and self.grid.allowed_position(i,j-1)== True:
            j_moved = j-1

        # if x' state not allowed, proba to get there is 0
        if self.allowed_position(i_prime,j_prime) is not True:
            return 0

        # if x' isnt (0.0) and isnt the desired state, then probability to reach it is 0
        if ( not(i_moved == i_prime and j_moved == j_prime) and not (i_prime == 0 and j_prime == 0) ):
            return 0

        # if the desired state is the (0.0) state, and is also x', the proba is 1 (either we reach it the normal way or we reach it through stochasticity)
        if ( (i_moved == i_prime  and j_moved == j_prime) and (i_prime == 0 and j_prime == 0) ):
            return 1

        # if x' is the desired state (x + u), then the proba is just the nomal move chance to reach its target
        if ( (i_moved == i_prime  and j_moved == j_prime) ):
            return normalMoveChance

        # if x' is (0.0) and x' is not the desired state (x +u), then the proba is beta
        if (i_prime == 0 and j_prime == 0) and not (i_prime == i_moved and j_prime == j_moved):
            return resetChance

        # if we get here, unexpected case, return -1
        print(" ERROR smth unexpected happened in p(x'|x,u)")
        return -1




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
    'beta' is the probability that the agent fails to move and stays
    still instead
    """
    def __init__(self,positionI,positionJ,rewards,discount,steps, beta,policy):
        self.grid = Grid(rewards,discount)
        policyType = self.policy_definition(policy)
        self.agent = Agent(positionI,positionJ,self.grid, beta,policyType)
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


    # returns all the scores of the game (one for every step) under the form of a
    # one dimensional table
    def get_scores(self):
        return self.scores


    # launches a game
    def start_game(self):
        for i in range(self.steps):

            self.scores[i] = self.agent.get_score()
            self.iPositions[i], self.jPositions[i] =  self.agent.get_position()

            self.agent.policy()

            self.moves.append(self.agent.get_curr_move())
            self.rewards.append(self.agent.get_curr_reward())

            self.rewardFromStateAndAction.append(self.agent.rewardFromStateAndAction)
            self.state2FromState1AndAction.append(self.agent.state2FromState1AndAction)

        # zip together all the elements vectors that together make up a trajectory into
        # the trajectory list
        self.trajectory =  list(zip(self.iPositions, self.jPositions))
        self.trajectory = list(zip(self.trajectory, self.moves))
        self.trajectory = list(zip(self.trajectory, self.rewards))
