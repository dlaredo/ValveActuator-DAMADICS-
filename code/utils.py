from random import shuffle

def random_mini_batches(X, y, minibatch_size=256, seed=None):


	n_m = X.shape[1]
	shuffled = [i for i in range(n_m)]
	shuffle(shuffled)
	num_minibatches = int(n_m/minibatch_size)
	minibatches = []

	'''print(n_m)
	print(num_minibatches)
	print(num_minibatches*minibatch_size)
	print(n_m-num_minibatches*minibatch_size)

	print(X.shape)
	print(y.shape)
	print(X)
	print(y)'''

	for i in range(num_minibatches):
		indices = shuffled[i*minibatch_size:(i+1)*minibatch_size]
		X_mini = X[:,indices]
		y_mini = y[:,indices]
		minibatches.append((X_mini, y_mini))

	#add the last minibatch
	indices = shuffled[num_minibatches*minibatch_size:]
	X_mini = X[:,indices]
	y_mini = y[:,indices]
	minibatches.append((X_mini, y_mini))

	return minibatches