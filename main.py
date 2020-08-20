from ClassificationNeuralNetwork import NeuralNetwork

nn = NeuralNetwork([784, 50, 10], 0.1, 'num_class_data.npy', 'num_class_label.npy', [60., 20., 20.])

nn.test()
nn.train(1000, 200)
nn.test()
