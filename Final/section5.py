import numpy as np
import matplotlib.pyplot as plt
from game import Game


array = np.array([
        [-3, 1, -5, 0, 19],
        [6, 3, 8, 9, 10],
        [5, -8, 4, 1, -8],
        [6, -9, 4, 19, -5],
        [-20, -17, -4, -3, 9]])

steps = 100
discount =  0.99
beta = 0.5

initialI = 0
initialJ = 0

game = Game(initialI,initialJ,array,discount,steps, beta, "RAND")
game.start_game()

rewards =  game.rewardFromStateAndAction # r_x_u
stateToState = game.state2FromState1AndAction # x' x u

for i in stateToState:
    print(i)

# compute p(x'|x,u) using a dictionnaries to count
# a first dictionnary will use the tuple (x',x, u) as a key and the number of time it appeared
# as associated element.
# a second dict will use the (x,u) tuple as key and the associated value will be the number of time this
# couple appeared
# from there we can infer p(x'|x,u)

# first dict
s_s_a = {} # state state action = s_s_a
for s_t_s in stateToState:

    if s_t_s in s_s_a: # if the element already existed, increment the counter
        s_s_a[s_t_s] += 1
        
    else: # if first time we encounter the (x' x u) we add it to the dict and initialize counter to 1
        s_s_a[s_t_s] = 1

# second dict
s_a = {} # state action
for s_t_s in stateToState:
    xprime, x, u = stateToState
    x_u = (x,u)

    if x_u in s_a: # if the element already existed, increment the counter
        s_a[x_u] += 1

    else: # if first time we encounter the (x' x u) we add it to the dict and initialize counter to 1
        s_a[x_u] = 1
