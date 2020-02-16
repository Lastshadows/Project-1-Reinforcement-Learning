import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

from game import Game

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--stochastic",help="Add stochasticity to the program",action="store_true")
	args = parser.parse_args()

	array = np.array([
	        [-3, 1, -5, 0, 19],
	        [6, 3, 8, 9, 10],
	        [5, -8, 4, 1, -8],
	        [6, -9, 4, 19, -5],
	        [-20, -17, -4, -3, 9]])

	size_x,size_y=array.shape
	steps = 500
	discount =  0.99
	beta = 0.5
	if args.stochastic ==False:
		beta = 0
	initialI = 0
	initialJ = 0
	policy = "Q"

	vectorScores = np.zeros((size_x,steps))

	plt.figure(figsize=(20,10))


	legend = []
	for i in range(size_x):
		for j in range(size_y):
			for k in range(5):
				game = Game(initialI,initialJ,array,discount,steps, beta,policy)
				game.start_game()
				vectorScores[i][:] = vectorScores[i][:] + game.get_scores()

			vectorScores[i][:] = vectorScores[i][:]/5
			plt.plot(vectorScores[i])
			print(' The expected return for the row '+ str(i+1)+' and column '+str(j+1)+' : ' + str(game.scores[steps - 1]))
			legend.append("row "+str(i+1) + " column " + str(j+1))
			initialJ = initialJ + 1

		initialJ = 0
		initialI = initialI + 1

	"""
	nb_run = 10
	score = 0
	scores = np.zeros(steps)

	for k in range(nb_run):
		game = Game(initialI,initialJ,array,discount,steps, beta,policy)
		game.start_game()
		scores += game.get_scores()

	scores = scores/nb_run
	print(scores)




	x = (initialI , initialJ )
	u = "UP"
	xp =  (initialI , initialJ +2)

	r = game.grid.compute_r_x_u(x,u, beta)
	p = game.grid.compute_proba_xprime_x_u(xp, x, u, beta)
	q = game.grid.Q_function( 2, x, u, beta)

	print(q)
	print(r)
	"""
