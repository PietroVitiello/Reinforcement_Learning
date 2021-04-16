import numpy as np
import torch
from matplotlib import pyplot as plt


# Turn on interactive mode for PyPlot, to prevent the displayed graph from blocking the program flow
plt.ion()


# Create a Network class, which inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=10)
        self.layer_2 = torch.nn.Linear(in_features=10, out_features=10)
        self.output_layer = torch.nn.Linear(in_features=10, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# Main entry point
if __name__ == "__main__":

    # Create some input data with 3 dimensions
    input_data = np.random.uniform(0, 1, [100, 3]).astype(np.float32)
    # Create some label data with 2 dimensions
    label_data = np.zeros([100, 2], dtype=np.float32)
    # Create a function which maps the input data to the labels. This is the function which the neural network will try to predict.
    for i in range(100):
        label_data[i, 0] = 1 + input_data[i, 0] * input_data[i, 0] + input_data[i, 1] * input_data[i, 2]
        label_data[i, 1] = input_data[i, 0] * input_data[i, 1] - input_data[i, 2] * 3

    # Create the neural network
    network = Network(input_dimension=3, output_dimension=2)
    # Create the optimiser
    optimiser = torch.optim.Adam(network.parameters(), lr=0.001)

    # Create lists to store the losses and epochs
    losses = []
    iterations = []

    # Create a graph which will show the loss as a function of the number of training iterations
    fig, ax = plt.subplots()
    ax.set(xlabel='Iteration', ylabel='Loss', title='Loss Curve for Torch Example')

    # Loop over training iterations
    for training_iteration in range(1000):
        # Set all the gradients stored in the optimiser to zero.
        optimiser.zero_grad()
        # Sample a mini-batch of size 5 from the training data
        # NOTE: when just training on a single example on each iteration, the NumPy array (and Torch tensor) still needs to have two dimensions: the mini-batch dimension, and the data dimension. And in this case, the mini-batch dimension would be 1, instead of 5. This can be done by using the torch.unsqueeze() function.
        minibatch_indices = np.random.choice(range(100), 5)
        minibatch_inputs = input_data[minibatch_indices]
        minibatch_labels = label_data[minibatch_indices]
        # Convert the NumPy array into a Torch tensor
        minibatch_input_tensor = torch.tensor(minibatch_inputs)
        minibatch_labels_tensor = torch.tensor(minibatch_labels)
        # Do a forward pass of the network using the inputs batch
        # NOTE: when training a Q-network, you will need to find the prediction for a particular action. This can be done using the "torch.gather()" function.
        network_prediction = network.forward(minibatch_input_tensor)
        # Compute the loss based on the label's batch
        loss = torch.nn.MSELoss()(network_prediction, minibatch_labels_tensor)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the network parameters.
        loss.backward()
        # Take one gradient step to update the network
        optimiser.step()
        # Get the loss as a scalar value
        loss_value = loss.item()
        # Print out this loss
        print('Iteration ' + str(training_iteration) + ', Loss = ' + str(loss_value))
        # Store this loss in the list
        losses.append(loss_value)
        # Update the list of iterations
        iterations.append(training_iteration)
        # Plot and save the loss vs iterations graph
        ax.plot(iterations, losses, color='blue')
        plt.yscale('log')
        plt.show()
        fig.savefig("loss_vs_iterations.png")

