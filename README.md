## About
A fully connected feedforward neural network. Cost function: mean squared error, activation functions: exponential linear unit in hidden layers and sigmoid in output layer. Supports mini-batch gradient descent with variable batch size as well as stochastic gradient descent and batch gradient descent. User can freely choose the amount of layers and the amount of neurons in each layer. Each neuron has its own bias.
Should be capable of at least basic classification problems such as handwritten digits recognition if provided with such training data.
## How to use
1.

```java
import monika.Network;
```

and create a new Network object. For example to create a network with an input layer of 5 neurons, two hidden layers with 10 neurons each, a third hidden layer of 5 neurons and an output layer of 2 neurons, call the constructor in the following way:

```java
Network nn = new Network(5, 10, 10, 5, 2);
```

2.
Train the network using the train method. You have to give arguments in the following order: training data, epochs, batch size and learning rate. Epochs is an integer and refers to how many times you want to go through the whole training data during training. Batch size is also an integer and refers to how many training examples should be processed before updating the models parameters (note that batch size should be smaller or equal to the length of the training data and batch size -1 sets batch size to maximum that is the length of the training data i.e. batch gradient descent). Learning rate is a float and the bigger it is the faster the model learns i.e. in how big steps the model's parameters are updated after each batch (something around 0.1 may be a good starting point). Training data is a three dimensional array with the following structure: The main array contains arrays whose length must be 2 and these arrays contain an array of inputs and an array of expected outputs. In other words training data contains training examples and each training example contains an array of inputs and an array of expected outputs (note that the inputs array must be the same length as the input layer and the expected outputs array must be the same length as the output layer. Also both inputs and outputs should be normalized to the range from 0 to 1). The method returns an array of average errors per epoch.

```java
float[] errors = nn.train(data, epochs, batchSize, learningRate);
```

4.
After training you can use the network by using the forward method. Provide an array of inputs as an argument and the method will return an array of outputs.

```java
float[] outputs = nn.forward(inputs);
```
