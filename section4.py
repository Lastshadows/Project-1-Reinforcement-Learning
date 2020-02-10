import numpy as np
import matplotlib.pyplot as plt
from game import Game


array = np.array([
        [-3, 1, -5, 0, 19],
        [6, 3, 8, 9, 10],
        [5, -8, 4, 1, -8],
        [6, -9, 4, 19, -5],
        [-20, -17, -4, -3, 9]])

steps = 1000
discount =  0.99
beta = 0.5

initialI = 0
initialJ = 0

game = Game(initialI,initialJ,array,discount,steps, beta, "RAND")
game.start_game()

rewards =  game.rewardFromStateAndAction
stateToState = game.state2FromState1AndAction

for i in stateToState:
    print(i)

# compute p(x'|x,u) using a dictionnaries to count
# a first dictionnary will use the tuple (x',x, u) as a key and the number of time it appeared
# as associated element.
# a second dict will use the (x,u) tuple as key and the associated value will be the number of time this
# couple appeared
# from there we can infer p(x'|x,u)

ssa = {} # state state action = ssa
for sts in stateToState:
    if sts in ssa: # if the element already existed
        print(" oh quel belle opération !")
