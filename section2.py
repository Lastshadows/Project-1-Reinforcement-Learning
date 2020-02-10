import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

from game import Game


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='The default policy is RIGHT')
	parser.add_argument("--policy",type = str,
		help="chose a policy between always :  RIGHT LEFT UP DOWN \n or for random : RAND")
	args = parser.parse_args()

	policy = "RIGHT"
	if args.policy:
		if args.policy=="LEFT" or args.policy=="RIGHT" or args.policy=="UP" or args.policy=="DOWN" or args.policy =="RAND":
			policy = args.policy
		else : 
			print("UNKNOWN policy: "+args.policy)
			print("TRY : RIGHT - LEFT - UP - DOWN - RAND")
			sys.exit(0)

	array = np.array([
	        [-3, 1, -5, 0, 19],
	        [6, 3, 8, 9, 10],
	        [5, -8, 4, 1, -8],
	        [6, -9, 4, 19, -5],
	        [-20, -17, -4, -3, 9]])

	steps = 1000
	discount =  0.99
	beta = 0.5
	initialI = 3
	initialJ = 0
	game = Game(initialI,initialJ,array,discount,steps,beta,policy)
	game.start_game()

	for i in range(steps):
	    print(game.trajectory[i])