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
game = Game(3,0,array,discount,steps, beta)
game.start_game()
plt.plot(game.get_scores())
plt.savefig('Evolution of scores during the game 1 (1000 steps) .png')

for i in range(steps):
    print(game.trajectory[i])
