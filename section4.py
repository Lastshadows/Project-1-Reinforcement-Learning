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

	steps = 1000
	discount =  0.99
	beta = 0.5
	if args.stochastic ==False:
		beta = 0
	initialI = 0
	initialJ = 0
	policy = "Q"

	nb_run = 10
	score = 0
	scores = np.zeros(steps)

	for k in range(nb_run):
		game = Game(initialI,initialJ,array,discount,steps, beta,policy)
		game.start_game()
		scores += game.get_scores()

	scores = scores/nb_run
	print(scores)
