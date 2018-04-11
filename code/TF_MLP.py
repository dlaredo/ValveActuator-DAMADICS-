import tensorflow as tf
import utils


class MLPClassifier:

	def __init__(self, hidden_layer_sizes=(100,), alpha=1e-4, batch_size=256, learning_rate_init=1e-3, minibatch_size=256, num_epochs=200, seed=None):

		self.coeffs_ = {}
		self.intercepts_ = {}
		self.hidden_layer_sizes_ = hidden_layer_sizes
		self.alpha = alpha
		self.batch_size = batch_size
		self.learning_rate_init = learning_rate_init
		self._n_hidden_layers_ = len(self.hidden_layer_sizes_)
		self.n_layers_ = self._n_hidden_layers_+2
		self.minibatch_size = minibatch_size
		self.num_epochs = num_epochs
		self.seed_ = seed if seed == None 


	def create_placeholders(self, n_x, n_y):
	    """
	    Creates the placeholders for the tensorflow session.
	    
	    Arguments:
	    n_x -- scalar, number of features
	    n_y -- scalar, number of classes
	    
	    Returns:
	    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
	    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
	    
	    Tips:
	    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
	      In fact, the number of examples during test/train is different.
	    """

	    ### START CODE HERE ### (approx. 2 lines)
	    X = tf.placeholder(tf.float32, shape=(n_x, None), name='X')
	    Y = tf.placeholder(tf.float32, shape=(n_y, None), name='Y')
	    ### END CODE HERE ###
	    
	    return X, Y


	def initialize_parameters(self, n_x, n_y):
	    """
	    Initializes parameters to build a neural network with tensorflow.

	    Arguments:
	    n_x -- scalar, number of features
	    n_y -- scalar, number of classes
	    
	    Returns:
	    parameters -- a dictionary of tensors containing coeffs and intercepts
	    """
	    
	    # to make the outputs deterministic
	    if self.seed_ != None:
	    	tf.set_random_seed(seed)

	    sizes = []
	    sizes.append(n_x)
	    sizes += list(self.hidden_layer_sizes_)
	    sizes.append(1)							#only works for binary classification so far

	    for l in range(self.n_layers_-1):

	    	#print(sizes[l])
	    	#print(sizes[l+1])

	    	self.coeffs_["W"+str(l+1)] = tf.get_variable("W"+str(l+1), [sizes[l+1], sizes[l]], initializer = tf.contrib.layers.xavier_initializer(seed=self.seed_))
	    	self.intercepts_["b"+str(l+1)] = tf.get_variable("b"+str(l+1), [sizes[l+1],1], initializer = tf.zeros_initializer())


	def forward_propagation(self, X):
		"""
		Implements the forward propagation for the model: LINEAR -> RELU ... -> LINEAR -> RELU
		
		Arguments:
		X -- input dataset placeholder, of shape (input size, number of examples)
		coeffs_, intercepts_ -- python dictionary containing your parameters the shapes are given in initialize_parameters

		Returns:
		Z -- the output of the last LINEAR unit
		"""

		A = X

		for l in range(self.n_layers_-1):
			W = self.coeffs_["W"+str(l+1)]
			b = self.intercepts_["b"+str(l+1)]
			Z = tf.add(tf.matmul(W, A), b)
			A = tf.nn.relu(Z)

		return Z


	def compute_cost(self, Z, Y):
		"""
    	Computes the cost
    	
    	Arguments:
    	Z -- output of forward propagation (output of the last LINEAR unit)
    	Y -- "true" labels vector placeholder, same shape as Z
    	
    	Returns:
    	cost - Tensor of the cost function
    	"""

    	# to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
		logits = tf.transpose(Z)
		labels = tf.transpose(Y)

		#works only with binary classification so far
		cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

		return cost


	def fit(self, X, y):
		"""
    	Fit the coeffs_ and intercepts_ to the training data X, y
    	
    	Arguments:
    	X -- input dataset placeholder, of shape (input size, number of examples)
    	Y -- "true" labels vector placeholder, of shape (input size, number of classes)
    	
    	Returns:
    	None
    	"""

		n_x, n_m = X.shape
		n_y = y.shape[0]
		print_cost = True
		costs = []

		seed = self.seed_

		#Define the tensorflow graph
		X_tf, y_tf = self.create_placeholders(n_x, n_y)
		self.initialize_parameters(n_x, n_y)

		print(self.coeffs_)
		print(self.intercepts_)

		Z = self.forward_propagation(X_tf)
		cost = self.compute_cost(Z, y_tf)

		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_init).minimize(cost)
		init = tf.global_variables_initializer()

		with tf.Session() as sess:
			sess.run(init)

			#do the training loop
			for epoch in range(self.num_epochs):

				epoch_cost = 0
				num_minibatches = int(n_m/self.minibatch_size)

				if self.seed_ != None
					seed = seed+1
				
				minibatches = utils.random_mini_batches(X, y, self.minibatch_size, seed)

				for minibatch in minibatches:

					(minibatch_X, minibatch_y) = minibatch
					_, minibatch_cost = sess.run([optimizer, cost], feed_dict={X_tf:minibatch_X, y_tf:minibatch_y})

					epoch_cost += minibatch_cost / num_minibatches

				# Print the cost every epoch
				if print_cost == True and epoch % 100 == 0:
					print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
				if print_cost == True and epoch % 5 == 0:
					costs.append(epoch_cost)


