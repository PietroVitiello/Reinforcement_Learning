# Import some modules from other libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import math
import cv2
import os

# Import the environment module
from env import Environment
from replay_buffer import ReplayBuffer
from q_value_visualiser import QValueVisualiser


# The Agent class allows the agent to interact with the environment.
class Agent:

    # The class initialisation function.
    def __init__(self, environment):
        # Set the agent's environment.
        self.environment = environment
        # Create the agent's current state
        self.state = None
        # Create the agent's total reward for the current episode.
        self.total_reward = None
        # Reset the agent.
        self.reset()

    # Function to reset the environment, and set the agent to its initial state. This should be done at the start of every episode.
    def reset(self):
        # Reset the environment for the start of the new episode, and set the agent's state to the initial state as defined by the environment.
        self.state = self.environment.reset()
        # Set the agent's total reward for this episode to zero.
        self.total_reward = 0.0

    # Function to make the agent take one step in the environment.
    def step(self):
        # Choose the next action.
        discrete_action = self._choose_next_action()
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, distance_to_goal = self.environment.step(self.state, continuous_action)
        # Compute the reward for this paction.
        reward = self._compute_reward(distance_to_goal)
        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # Return the transition
        return transition

    def greedy_step(self, network):
        
        discrete_action = self.best_action(network)
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        next_state, distance_to_goal = self.environment.step(self.state, continuous_action, True)

        reward = self._compute_reward(distance_to_goal)
        transition = (self.state, discrete_action, reward, next_state)

        self.state = next_state
        self.total_reward += reward

        return transition

    # Function for the agent to choose its next action
    def _choose_next_action(self):
        action = np.random.randint(4)
        return action

    def best_action(self, network):
        q_values = network.get_q_values(self.state)
        action = np.argmax(q_values.tolist()[0])
        return action

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:
            continuous_action = np.array([0.1, 0], dtype=np.float32) # 0: R --> Move 0.1 to the right, and 0 upwards
        elif discrete_action == 1:
            continuous_action = np.array([-0.1, 0], dtype=np.float32) # 1: L --> Move 0.1 to the left, and 0 upwards
        elif discrete_action == 2:
            continuous_action = np.array([0, 0.1], dtype=np.float32) # 2: U --> Move 0 laterally, and 0.1 upwards
        else:
            continuous_action = np.array([0, -0.1], dtype=np.float32) # 3: D --> Move 0 laterally, and 0.1 downwards
        return continuous_action

    # Function for the agent to compute its reward. In this example, the reward is based on the agent's distance to the goal after the agent takes an action.
    def _compute_reward(self, distance_to_goal):
        reward = float(0.1*(1 - distance_to_goal))
        return reward


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input_):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input_))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        # Create target network with all weights to zero
        self.target_network = Network(input_dimension=2, output_dimension=4)
        self.zero_target()

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transitions, gamma=0, do_target=False):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transitions, gamma, do_target)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transitions, gamma=0, use_target=False):
        input_tensor = torch.tensor([transition[0] for transition in transitions], dtype=torch.float32)
        actions = torch.tensor([transition[1] for transition in transitions], dtype=torch.int64)
        rewards = torch.tensor([transition[2] for transition in transitions], dtype=torch.float32)
        next_tensor = torch.tensor([transition[3] for transition in transitions], dtype=torch.float32)

        network_prediction = self.q_network.forward(input_tensor)
        network_prediction = torch.gather(network_prediction, dim=1, index=actions.unsqueeze(-1)).squeeze(-1)

        if use_target == True:
            future_q =  self.target_network.forward(next_tensor)
            future_q = future_q.detach()
            best_future_actions = torch.max(future_q, 1).indices
            future_returns = future_q.gather(dim=1, index=best_future_actions.unsqueeze(-1)).squeeze(-1)
            labels_tensor = rewards + gamma * future_returns
        elif gamma == 0:
            labels_tensor = rewards
        else:
            future_q =  self.q_network.forward(next_tensor)
            best_future_actions = torch.max(future_q, 1).indices
            future_returns = future_q.gather(dim=1, index=best_future_actions.unsqueeze(-1)).squeeze(-1)
            labels_tensor = rewards + gamma * future_returns
        

        #print(f"\n\n\n\nReward: {rewards} \nFuture q val: {future_q} \nBest actions: {best_future_actions} \nLabels: {labels_tensor}")

        loss = torch.nn.MSELoss()(network_prediction, labels_tensor)

        # print(network_prediction.size())
        # print(labels_tensor.size())
        # print('\n\n\n')

        return loss
    
    def get_q_values(self, state):
        input_tensor = torch.tensor(state).unsqueeze(0)
        q_values = self.q_network.forward(input_tensor)

        return q_values

    def target_update(self):
        q_dictionary = self.q_network.state_dict()
        self.target_network.load_state_dict(q_dictionary)

    def zero_target(self):
        for name, params in self.target_network.named_parameters():
            tensor_size = list(params.size())
            params.data.copy_(torch.zeros(tensor_size))



def q_visual(environment, dqn, save_id):
    q_visualiser = QValueVisualiser(environment, magnification=500)
    q_values = np.zeros((10, 10, 4))

    for row in range(10):
        for col in range(10):
            x = col/10 + 0.05
            y = row/10 + 0.05
            state = (y, x)
            q_values[row, col, :] = dqn.get_q_values(state).tolist()[0]
    
    q_visualiser.draw_q_values(q_values)
    q_values_image = q_visualiser.get_image()
    path = os.getcwd()
    cv2.imwrite(os.path.join(path, '..\\Figures\\QValuesImage'+save_id+'.png'), q_values_image)
    cv2.waitKey(0)

def greedy_policy_visual(environment, dqn, agent, save_id):
    agent.reset()
    environment.do_display(True)
    
    for step_num in range(20):
        transition = agent.greedy_step(dqn)
        time.sleep(0.2)
    print("Greedy trace completed")
    policy_image = environment.get_image()
    path = os.getcwd()
    cv2.imwrite(os.path.join(path, '..\\Figures\\GreedyPolicyImage'+save_id+'.png'), policy_image)
    cv2.waitKey(0)

# Main entry point
if __name__ == "__main__":

    show_loss = 0
    show_env = 0
    show_q_values = 1
    show_best_policy = 1
    which = 4  #1 reward, no batch | 2 reward, batch | 3 Bellman | 4 Bellman, target

    if show_loss : plt.ion()

    # Create an environment.
    # If display is True, then the environment will be displayed after every agent step. This can be set to False to speed up training time. The evaluation in part 2 of the coursework will be done based on the time with display=False.
    # Magnification determines how big the window will be when displaying the environment on your monitor. For desktop monitors, a value of 1000 should be about right. For laptops, a value of 500 should be about right. Note that this value does not affect the underlying state space or the learning, just the visualisation of the environment.
    environment = Environment(display=show_env, magnification=500)
    
    # Initialise agent and network
    agent = Agent(environment)
    dqn = DQN()

    # Used to plot the loss against episode
    fig, ax = plt.subplots()
    ax.set(xlabel='Episode', ylabel='Loss', title='Loss Curve using Bellman equation and Target network')
    episode_counter = 0
    episodes = []
    losses = []

    # Training parameters
    n_steps = 20
    n_episodes = 100
    N = 10 # Number of epochs after which target gets updated
    gamma = 0.9
    do_target = True if which==4 else False

    # Initialise buffer
    replay_buffer = ReplayBuffer(5000)
    batch_size = 100
    additional_episodes = math.ceil(batch_size/n_steps) if which!=1 else 0
    n_episodes += additional_episodes

    # Loop over episodes
    while episode_counter < n_episodes:
        
        agent.reset()
        loss_avg = 0

        if which == 1:
            for step_num in range(n_steps):
                
                transition = [agent.step(), ]
                loss = dqn.train_q_network(transition)
                loss_avg += loss

                if show_env: time.sleep(0.1)

        elif which == 2:
            for step_num in range(n_steps):
                
                transition = agent.step()
                replay_buffer.add(transition)

                if episode_counter >= additional_episodes:
                    batch_transitions = replay_buffer.sample(batch_size)
                    loss = dqn.train_q_network(batch_transitions)
                    loss_avg += loss

                if show_env: time.sleep(0.1)

        else:
            for step_num in range(n_steps):
                
                transition = agent.step()
                replay_buffer.add(transition)

                if episode_counter >= additional_episodes:
                    batch_transitions = replay_buffer.sample(batch_size)
                    loss = dqn.train_q_network(batch_transitions, gamma, do_target)
                    loss_avg += loss

                if show_env: time.sleep(0.1)

        episode_counter += 1
        print(f'Episode: {episode_counter}')
        if (episode_counter - additional_episodes) % N == 0:
            dqn.target_update()
            print("update")

        if loss_avg != 0:
            losses.append(loss_avg/20)
            episodes.append(episode_counter - additional_episodes)

        if show_loss:
            ax.plot(episodes, losses, color='blue')
            plt.yscale('log')
            plt.show()
            plt.pause(0.001)
    
    if not show_loss:
        std = np.std(losses)
        print(f"\n\nVariance: {std**2}\n")

        ax.plot(episodes, losses, color='blue')
        plt.yscale('log')
        plt.show()

    #Stuff for image saving names
    if which == 1:
        save_id = '_none'
    elif which == 2:
        save_id = '_batch'
    elif which == 3:
        save_id = '_bellman'
    elif which == 4:
        save_id = '_target'
    
    if show_q_values: q_visual(environment, dqn, save_id)
    if show_best_policy: greedy_policy_visual(environment, dqn, agent, save_id)
