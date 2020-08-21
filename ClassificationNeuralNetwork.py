import numpy as np
from scipy.optimize import minimize


class NeuralNetwork:
    def __init__(self, structure: list, lmd, data: str, label: str, partition: list = None):
        """
        Initializes the neural network with given parameters

        data and label should be saved in numpy format on disk, and both
        should have 2 dimensions. each training example should be one row.

        partition divides the data into 3 groups. tran, cross validation,
        and test. these groups used later for validation and testing.

        :param list structure: a list of number of nodes in each layer
        :param lmd: lambda
        :param data: the address of data on disk
        :param label: the address of label on disk
        :param partition: a list of size of each partition in percent
        """

        self.structure = structure
        self.n_layer = len(structure)  # number of layers

        self.data = np.load(data)
        m = self.data.shape[0]  # number of training examples

        # read the label data, and convert them such that for each label
        # we have a vector that all elements are zero except the i-th element
        # that i is the label
        temp_label = np.load(label)
        k = np.unique(temp_label).size
        self.label = np.zeros([k, m])
        self.label[temp_label.T.astype(np.int), np.arange(m)] = 1

        # sets the default partition if partition is not given
        if partition is None:
            self.partition = [60., 20., 20.]  # train, cv, test percent
        else:
            self.partition = partition
        self.c1 = 0
        self.c2 = 0
        self.cal_partition()
        self.lmd = lmd  # lambda

        # random initialization of theta
        self.theta = [np.ones([1])] * (self.n_layer - 1)
        for i in range(self.n_layer - 1):
            init_epsilon = (6 / (structure[i + 1] + structure[i])) ** 0.5  # for random initialization
            self.theta[i] = np.random.random([structure[i + 1], structure[i] + 1]) * 2 * init_epsilon - init_epsilon

    def cal_partition(self):
        """
        calculates the size of train, cv and test portion of data
        [:c1, c1:c2, c2:]

        :return: None
        """

        m = self.data.shape[0]
        self.c1 = round(self.partition[0] * m / 100.)
        self.c2 = round((self.partition[0] + self.partition[1]) * m / 100.)

    @staticmethod
    def sigmoid(g: np.ndarray) -> np.ndarray:
        """
        sigmoid function

        :param np.ndarray g: an array of numbers
        :return np.ndarray: sigmoid of g
        """

        return 1. / (1. + np.exp(-g))

    def forward_prop(self, x: np.ndarray):
        """
        Forward Propagation

        :param np.ndarray x: examples
        :return: output
        """

        # if the input has one dimension, add a new dimension
        if len(x.shape) == 1:
            a = np.expand_dims(x, 0)
        else:
            a = x

        a = a.T
        for i in range(self.n_layer - 1):
            z = np.insert(a, 0, values=1.0, axis=0)  # add bias unit
            a = self.sigmoid(self.theta[i] @ z)

        return a

    def cost(self, group: str):
        """
        Calculates the cost with respect to current theta

        :param str group: train, cv (cross validation) or test
        :return: cost
        """

        # sets the data and label and does forward propagation
        if group == 'train':
            m = self.c1
            h_theta = self.forward_prop((self.data[:self.c1, :]))
            y = self.label[:, :self.c1]
        elif group == 'cv':
            m = self.c2 - self.c1
            h_theta = self.forward_prop((self.data[self.c1:self.c2, :]))
            y = self.label[:, self.c1:self.c2]
        elif group == 'test':
            m = self.data.shape[0] - self.c2
            h_theta = self.forward_prop((self.data[self.c2:, :]))
            y = self.label[:, self.c2:]
        else:
            print('Error: opt should be train, cv or test')
            return 0

        j = -(np.multiply(y, np.log(h_theta)) + np.multiply((1 - y), np.log(1 - h_theta))).sum()

        # regularization term
        reg = 0
        for t in self.theta:
            reg = reg + (t[:, 1:] ** 2).sum()

        return (j + self.lmd * reg / 2.) / m

    def grad(self):
        """
        Calculates the back propagation

        :return: gradient (back propagation)
        """

        L = self.n_layer  # number of layers
        x = self.data[:self.c1, :]  # input data
        y = self.label[:, :self.c1]  # labels
        m = self.c1  # number of training examples

        # set Delta = 0 for all layers
        big_delta = [np.ones([1])] * (L - 1)
        for i in range(L - 1):
            big_delta[i] = np.zeros([self.structure[i + 1], self.structure[i] + 1])

        # forward propagation
        # since we need the a data for the rest of the calculation,
        # we do forward propagation here
        a = [np.ones([1])] * L
        a[0] = x.T
        for i in range(L - 1):
            a[i] = np.insert(a[i], 0, values=1.0, axis=0)
            a[i + 1] = self.sigmoid(self.theta[i] @ a[i])

        # compute small delta for last layer
        small_delta = [np.ones([1])] * L
        small_delta[-1] = (a[-1] - y)

        # compute small delta for the rest of layers
        for i in range(L - 2):
            index = L - i - 2
            small_delta[index] = (self.theta[index][:, 1:].T @ small_delta[index + 1]) * (
                    a[index][1:] * (1 - a[index][1:]))

        # compute big delta and gradient with regularization
        D = [np.ones([1])] * (L - 1)
        for i in range(L - 1):
            big_delta[i] = big_delta[i] + (small_delta[i + 1] @ a[i].T)
            D[i] = big_delta[i]
            D[i][:, 1:] = D[i][:, 1:] + self.lmd * self.theta[i][:, 1:]
            D[i] = D[i] / m

        return D

    def grad_check(self, layer, i, j):
        """
        This function is used to test the grad function correctness

        :param layer: the layer we want to check
        :param i: the row of layer we want to check
        :param j: the column of layer we want to check
        :return: None
        """

        # computer grad with grad function
        t = self.theta
        g1 = self.grad()
        print('grad:\t', g1[layer][i, j])

        # compute grad manually
        e = 0.0001
        t[layer][i, j] = t[layer][i, j] + e
        j1 = self.cost('train')
        t[layer][i, j] = t[layer][i, j] - 2 * e
        j2 = self.cost('train')
        t[layer][i, j] = t[layer][i, j] + e
        grad_check = (j1 - j2) / (2 * e)
        print('check:\t', grad_check)

    def gradient_descent(self, itr, alp=1.):
        """
        Gradient descent algorithm

        :param itr: number of iteration
        :param alp: alpha
        :return: None
        """

        c = self.cost('train')

        print('initial cost: ', c)

        for i in range(itr):
            g = self.grad()
            for j in range(self.n_layer - 1):
                self.theta[j] = self.theta[j] - (alp * g[j])

            if (i + 1) % 10 == 0:
                print('iter: ', i + 1, '\tcost(train): ', self.cost('train'))

        print('final cost: ', self.cost('train'))

    def cost_flat(self, x):
        """
        Used in scipy.optimize minimize function because this
        function needs a cost function with one flat input

        x is necessary because of minimize function, but we do not
        use because theta will be updated in grad_flat function

        :param x: not used
        :return: cost of train
        """

        return self.cost('train')

    def grad_flat(self, x):
        """
        Used in scipy.optimize minimize function because this
        function needs a grad function with one flat input

        :param x: flatten input data
        :return: flatten grad
        """

        # sets theta with rolled x
        index = 0
        for j in range(self.n_layer - 1):
            t = self.theta[j].shape
            self.theta[j] = np.reshape(x[index: index + t[0] * t[1]], t)
            index = index + t[0] * t[1]

        # compute grad
        g = self.grad()

        # unrolls the grad
        g_unroll = np.zeros(x.shape)
        index = 0
        for j in range(self.n_layer - 1):
            t = self.theta[j].shape
            g_unroll[index: index + t[0] * t[1]] = np.reshape(g[j], [1, t[0] * t[1]])
            index = index + t[0] * t[1]

        return g_unroll

    # train with built in function
    def train_helper(self, itr):
        """
        A helper function used in train.
        This function uses L-BFGS-B algorithm for minimization.

        :param itr: number of iteration
        :return: None
        """

        # count number of elements of thetas
        n = 0
        for i in range(self.n_layer - 1):
            s = self.theta[i].shape
            n = n + s[0] * s[1]

        # unrolls initial theta
        x0 = np.zeros(n)
        index = 0
        for j in range(self.n_layer - 1):
            s = self.theta[j].shape
            x0[index: index + s[0] * s[1]] = np.reshape(self.theta[j], [1, s[0] * s[1]])
            index = index + s[0] * s[1]

        res = minimize(self.cost_flat, x0, method='L-BFGS-B', jac=self.grad_flat, options={'maxiter': itr, 'disp': False})

        # rolls the thetas
        index = 0
        for j in range(self.n_layer - 1):
            s = self.theta[j].shape
            self.theta[j] = np.reshape(res.x[index: index + s[0] * s[1]], s)
            index = index + s[0] * s[1]

    def train(self, itr, batch=50):
        """
        Uses train_helper for minimization and prints the results.

        :param itr: number of iteration
        :param batch: number of iteration for each printing results
        :return: None
        """

        print('initial cost: ', self.cost('train'))

        num_iter = itr // batch
        for i in range(num_iter):
            self.train_helper(batch)
            print('itr: ', (i + 1) * batch, '\tcost: ', self.cost('train'))

        if itr % batch != 0:
            self.train_helper(itr % batch)
            print('itr: ', itr, '\tcost: ', self.cost('train'))

    def test(self):
        """
        Tests the neural network with current theta on the test partition,
        and prints the percent of correct predictions.

        :return: None
        """

        a = self.forward_prop(self.data[self.c2:, :])
        m = a.shape[1]  # number of examples

        pred = np.zeros(a.shape)
        pred[np.argmax(a, 0), np.arange(m)] = 1

        # counts number correct predictions
        p = np.abs(pred - self.label[:, self.c2:]).sum(0)
        p = round(((p == 0).sum() * 100 / m), 2)

        print('accuracy: ', p, '%')
