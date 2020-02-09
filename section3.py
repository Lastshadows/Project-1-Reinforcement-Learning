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

size_x,size_y=array.shape
vectorScores = np.zeros((size_x,steps))

initialI = 0
initialJ = 0

plt.figure(figsize=(20,10))

legend = []

print('The chosen policy is always move RIGHT')

for i in range(size_x):
	game = Game(initialI,initialJ,array,discount,steps, beta)
	game.start_game()
	vectorScores[i][:] = game.get_scores()
	plt.plot(vectorScores[i])
	print(' The expected return for the'+ str(i+1)+' row : ' + str(game.scores[steps - 1]))
	legend.append("row "+str(i+1))
	initialI = initialI + 1 

plt.legend(legend)
plt.title('Evolution of scores during the game (1000 steps) for the different initial positions using the always right policy')
plt.savefig('Evolution of scores during the game (1000 steps) for the different initial positions.png')
