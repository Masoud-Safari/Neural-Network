# Neural-Network
Implementation of basic neural network concepts in python. :brain:  

With this implementation, you can quickly build and train a neural network. :relaxed:  

This implementation consists of:  
- vectorized forward propagation and backpropagation  
- training (gradient descent and built-in L-BFGS-B in scipy)  
- testing the neural network  

---
### Sample Code
A neural network with 3 layers (input: 784 units, hidden layer: 50 units, output: 10 units), and lambda = 0.1.  
Trained for 1000 iterations.  
The num_class_data.npy and num_class_label.npy are consist of 5000 handwritten numbers from [the MNIST database](http://yann.lecun.com/exdb/mnist/).  

```python
from ClassificationNeuralNetwork import NeuralNetwork

nn = NeuralNetwork([784, 50, 10], 0.1, 'num_class_data.npy', 'num_class_label.npy')

nn.test()
nn.train(1000, 200)
nn.test()
```

Result:
```
accuracy:  9.8 %
initial cost:  8.122171329467614
itr:  200 	cost:  1.9672215322429254
itr:  400 	cost:  1.5490956701271907
itr:  600 	cost:  1.3867609010798139
itr:  800 	cost:  1.2941751160581383
itr:  1000 	cost:  1.2421255738642054
accuracy:  74.7 %
```

---
### How To Use:
You need to install numpy and scipy.
The input files should have been saved in numpy format.  
Both data and labels should have two dimensions, and each training example should be one row.  
