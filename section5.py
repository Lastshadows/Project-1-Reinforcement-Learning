import numpy as np
import matplotlib.pyplot as plt
from game import Game
import game as G
import random

def squared_error(a,b):
    return (a-b)**2

array = np.array([
        [-3, 1, -5, 0, 19],
        [6, 3, 8, 9, 10],
        [5, -8, 4, 1, -8],
        [6, -9, 4, 19, -5],
        [-20, -17, -4, -3, 9]])

steps_per_game = 50000
discount =  0.99
beta = 0.5

size_x,size_y=array.shape
trajectory = []
nb_of_games = 1

##################### CONVERGENCE SPEED TO P AND R FUNCTIONS ###################################################

print("computing the convergence speed to r and p from the size of the trajectory used \n")

# we run many games to gather data with our random agent, MDP is set to false
for i in range(nb_of_games):
    initialI = random.randrange(0, size_y)
    initialJ = random.randrange(0, size_x)
    game = Game(initialI,initialJ,array,discount,steps_per_game, beta, "RAND", False)
    game.start_game()
    trajectory = trajectory + game.trajectory_MDP

print("initial random agent games ended \n")
print("")
print(" computing all MDP proba and rewards\n")

# creating the domain and actions possible
U = ["UP", "DOWN", "RIGHT", "LEFT"]
X = []
for i in range(size_x):
    for j in range(size_y):
        X.append( (i,j) )

# collecting all rewards and proba expectations from MDP model and from known statistical model
MDP_rewards = []
stat_rewards = []
MDP_proba = []
stat_proba = []
countR = 0
countP =0
r_diff = 0
pDiff = 0
xAxis = []

for x in X:
    for u in U:
        MDP_r = G.reward_state_action_MDP(x,u,trajectory)
        stats_r = game.grid.compute_r_x_u(x,u,beta)

        MDP_rewards.append(MDP_r)
        stat_rewards.append(stats_r)
        r_diff += (squared_error(MDP_r, stats_r))
        countR = countR + 1

        for x2 in X:
            xAxis.append(countP)
            MDP_P =  G.proba_state1_action_state2_MDP(x,u,x2,trajectory)
            stats_P =  game.grid.compute_proba_xprime_x_u(x2, x, u, beta)

            MDP_proba.append(MDP_P)
            stat_proba.append(stats_P)
            pDiff += (squared_error(MDP_P, stats_P))
            countP += 1



print("plot of the precision of the MDP for R \n ")

fig, ax = plt.subplots()
ax.plot(MDP_rewards, label = "MDP")
plt.plot(stat_rewards, label = "stats")
ax.legend()
plt.title("R estimation,  for " + str(nb_of_games) +" games and " + str(steps_per_game) + " steps per game")
plt.xlabel("state-action duo")
plt.ylabel("estimated reward")
plt.show()


fig, ax = plt.subplots()
plt.plot(xAxis, MDP_proba, 'go', label = "MDP")
plt.plot(xAxis, stat_proba, 'ro', label = "stats")
ax.legend()
plt.title("P estimation, for " + str(nb_of_games) +" games and " + str(steps_per_game) + " steps per game")
plt.xlabel("state-action-state triple")
plt.ylabel("estimated proba")
plt.show()


print(" average squared error on r = " + str(r_diff/countR))
print(" average squared error on p = " + str(pDiff/countP))
print("")
print("")


print("computing both expected returns for each position ")
# display both expected return for each starting position side to side
steps_per_game = 100
vectorScores = np.zeros((size_x,steps_per_game))
vectorScoresMDP = np.zeros((size_x,steps_per_game))
policy = "Q"

for i in range(size_x):
    for j in range(size_y):
        for k in range(3):
            game = Game(initialI,initialJ,array,discount,steps_per_game, beta,policy, False)
            game.start_game()
            vectorScores[i][:] = vectorScores[i][:] + game.get_scores()

            print("done with stat Q game")
            game_MDP = Game(initialI,initialJ,array,discount,steps_per_game, beta, "Q", True)
            # we set the new game's grid with the experienced trajectory of earlier
            game_MDP.grid.set_prev_trajectory(trajectory)
            game_MDP.start_game()
            vectorScoresMDP[i][:] = vectorScoresMDP[i][:] + game.get_scores()
            print("done with MDP Q game")

        vectorScores[i][:] = vectorScores[i][:]/5
        vectorScoresMDP[i][:] = vectorScoresMDP[i][:]/5

        print(' The expected return for the row '+ str(i+1)+' and column '+str(j+1)+' with normal stats is : ' + str(game.scores[steps_per_game - 1])) # ???
        print(' The expected return for the row '+ str(i+1)+' and column '+str(j+1)+' with MDP is : ' + str(game_MDP.scores[steps_per_game - 1])) # ???
        print("")
        initialJ = initialJ + 1

    initialJ = 0
    initialI = initialI + 1
