import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

from game import Game

if __name__ == '__main__':

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
	policy = "Q"


	nb_run = 1
	score = 0
	scores = np.zeros(steps)

	for k in range(nb_run):
		game = Game(initialI,initialJ,array,discount,steps, beta,policy)
		game.start_game()
		scores += game.get_scores()


	scores = scores/nb_run
	print(scores)

	"""
	x = (initialI , initialJ )
	u = "UP"
	xp =  (initialI , initialJ +2)

	r = game.grid.compute_r_x_u(x,u, beta)
	p = game.grid.compute_proba_xprime_x_u(xp, x, u, beta)
	q = game.grid.Q_function( 2, x, u, beta)

	print(q)
	print(r)
	"""
