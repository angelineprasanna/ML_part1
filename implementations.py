import numpy as np

def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE
    N = len(y)
    e = y - np.dot(tx,w).flatten()
    loss = 1/(2*N)*np.dot(e.T,e)
    return loss
    # ***************************************************
    #raise NotImplementedError

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute gradient vector
    N = y.shape[0]

    e = y - tx @ w
    gradient = (-1/N)* tx.T@ e

    return gradient
    # ***************************************************
    #raise NotImplementedError

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    #ws = [initial_w]
    #losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        # ***************************************************
        #raise NotImplementedError
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        w = w - gamma * gradient
        # ***************************************************
        #raise NotImplementedError

        # store w and loss
        #ws.append(w)
        #losses.append(loss)

    return w, loss
    
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.

    Args:
        y: numpy array of shape=(B, )
        tx: numpy array of shape=(B,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """

    # ***************************************************
    # INSERT YOUR CODE HERE
    error = y - tx @ w
    gradient = - (tx.T @ error)/len(y)

    return gradient
    # TODO: implement stochastic gradient computation. It's the same as the usual gradient.
    # ***************************************************
    #raise NotImplementedError


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    #jai enlevé le batch_size
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    # Define parameters to store w and loss
    #ws = [initial_w]
    #losses = []
    w = initial_w
    batch_size = 1

  for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        for mini_y, mini_tx in batch_iter(y, tx, batch_size, num_batches=1):

            grad = compute_stoch_gradient(mini_y, mini_tx, w)
            w = w - gamma * grad

            loss = compute_loss(mini_y, mini_tx, w)

            #losses.append(loss)
            #ws.append(w)
        # TODO: implement stochastic gradient descent.
        # ***************************************************
        #raise NotImplementedError



    return w, loss

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    # ***************************************************
    # COPY YOUR CODE FROM EX03 HERE
    # least squares: TODO
    gram_matrix = tx.T @ tx
    cross_term = tx.T @ y
    weights  = np.linalg.solve(gram_matrix, cross_term)

    error = y - tx @ weights
    mse = np.mean(error**2) / 2
    # returns optimal weights, MSE
    # ***************************************************
    #raise NotImplementedError

    return weights, mse

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    # ***************************************************
    # COPY YOUR CODE FROM EX03 HERE
    D = tx.shape[1]
    N = tx.shape[0]
    I = np.eye(D)
    lambda_prime = lambda_*2*N
    w = np.linalg.solve(tx.T @ tx + lambda_prime*I, tx.T @ y)

    error = y -tx @ w
    mse = np.mean(error**2) / 2
    ridge_loss = mse + lambda_ * np.sum(weights**2)

    return w, ridge_loss
    # ridge regression: TODO
    # ***************************************************
    #raise NotImplementedError

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    >>> sigmoid(np.array([0.1]))
    array([0.52497919])
    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """
    #raise NotImplementedError
    sigmoid = 1 / (1 + np.exp(-t))
    return sigmoid


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(4).reshape(2, 2)
    >>> w = np.c_[[2., 3.]]
    >>> round(calculate_loss(y, tx, w), 8)
    1.52429481
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
    #raise NotImplementedError
    #sig = sigmoid(tx.dot(w))
    #loss = -np.mean(y * np.log(sig) + (1 - y) * np.log(1 - sig))

    
    xTw = np.dot(tx, w)
    term1 = np.sum(np.dot(y.T, xTw))  # y_n * x_n^T * w
    term2 = np.sum(np.log(1 + np.exp(xTw)))  # log(1 + e^(x_n^T * w))
    loss = (term2 - term1) / y.shape[0]

    return loss


test(calculate_loss)

def calculate_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> calculate_gradient(y, tx, w)
    array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]])
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
    #raise NotImplementedError("Calculate gradient")
    sig = sigmoid(tx.dot(w))
    gradient = (np.dot(tx.T, sig-y))/y.shape[0]
    return gradient 


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """#return the loss, gradient of the loss, and hessian of the loss.
    return the last weight vector and the loss

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        w: last weight vector, shape=(D,1)
        loss: scalar number
        #gradient: shape=(D, 1)
        #hessian: shape=(D, D)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> loss, gradient, hessian = logistic_regression(y, tx, w)
    >>> round(loss, 8)
    0.62137268
    >>> gradient, hessian
    (array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]]), array([[0.28961235, 0.3861498 , 0.48268724],
           [0.3861498 , 0.62182124, 0.85749269],
           [0.48268724, 0.85749269, 1.23229813]]))
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient, and Hessian: TODO
    # ***************************************************
    #raise NotImplementedError
    w = initial_w
    
    for iter in range(max_iters):
        loss = calculate_loss(y, tx, w)
        gradient = calculate_gradient(y, tx, w)
        #hessian = calculate_hessian(y, tx, w)
        w = w - gamma * gradient


    return w, loss




def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """return the weights and the loss .

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        w: last weight vector
        loss: scalar number
        #gradient: shape=(D, 1)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> lambda_ = 0.1
    >>> loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    >>> round(loss, 8)
    0.62137268
    >>> gradient
    array([[-0.08370763],
           [ 0.2467104 ],
           [ 0.57712843]])
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient, and Hessian: TODO
    w = intial_w
    for iter in range(max_iters):
        loss = calculate_loss(y, tx, w)
        gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient

    return w, loss
    # ***************************************************
    #raise NotImplementedError


