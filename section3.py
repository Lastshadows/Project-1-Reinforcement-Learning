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

	size_x,size_y=array.shape

	parser = argparse.ArgumentParser(description='The default policy is RIGHT')
	parser.add_argument("--policy",type = str,
		help="chose a policy between always :  RIGHT LEFT UP DOWN \n or for random : RAND")
	parser.add_argument("--stochastic",help="Add stochasticity to the program",action="store_true")
	args = parser.parse_args()

	policy = "RIGHT"
	if args.policy:
		if args.policy=="LEFT" or args.policy=="RIGHT" :
			policy = args.policy
			size_y = 1
		elif args.policy=="UP" or args.policy=="DOWN":
			policy = args.policy
			size_x = 1
		elif args.policy == "RAND":
			policy = args.policy
		else :
			print("UNKNOWN policy: "+args.policy)
			print("TRY : RIGHT - LEFT - UP - DOWN - RAND")
			sys.exit(0)
	else :
		policy = "RIGHT"
		#size_y = 1
	steps = 1000
	discount =  0.99
	beta = 0.5

	if args.stochastic==False:
		beta = 0

	vectorScores = np.zeros((size_x,steps))

	initialI = 0
	initialJ = 0

	plt.figure(figsize=(20,10))


	legend = []
	for i in range(size_x):
		for j in range(size_y):
			for k in range(10):
				game = Game(initialI,initialJ,array,discount,steps, beta,policy)
				game.start_game()
				vectorScores[i][:] = vectorScores[i][:] + game.get_scores()

			vectorScores[i][:] = vectorScores[i][:]/10
			plt.plot(vectorScores[i])
			print(' The expected return for the row '+ str(i+1)+' and column '+str(j+1)+' : ' + str(game.scores[steps - 1]))
			legend.append("row "+str(i+1) + " column " + str(j+1))
			initialJ = initialJ + 1

		initialJ = 0
		initialI = initialI + 1


	plt.legend(legend)
	#plt.title('Evolution of scores during the game (1000 steps) for the different initial positions using the '+policy+' policy')
	plt.savefig('Evolution of scores during the game (1000 steps) for the different initial positions.png')
