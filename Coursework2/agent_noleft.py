############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import collections

import numpy as np
import torch

class ReplayBuffer():

    def __init__(self, size, alpha=1, do_prioritised=False):
        self.buffer = np.zeros((size, 6))
        self.maxlen = size
        self.item_pointer = 0
        self.num_items = 0
        self.do_prioritised = do_prioritised
        self.alpha = alpha

        if do_prioritised:
            self.weights = np.zeros(size)
            self.probs = np.zeros(size)
            self.sampled_idx = None

    def add(self, transition):
        self.buffer[self.item_pointer, :] = transition

        if self.do_prioritised:
            if self.weights[0] == 0:
                max_weight = 1
            else:
                max_weight = np.max(self.weights)
            self.weights[self.item_pointer] = max_weight
            self.update_probabilities(self.alpha)

        # Update how many transitions are stored in the buffer
        if self.num_items < self.maxlen:
            self.num_items += 1

        # Update the pointer
        if self.item_pointer == (self.maxlen - 1):
            self.item_pointer = 0
        else:
            self.item_pointer += 1

    def sample(self, N):
        if self.do_prioritised:
            return self.prioritised_sample(N)
        else:
            return self.random_sample(N)
    
    # Classic replay buffer
    def random_sample(self, N):
        batch = np.zeros((N, 6))
        indeces = np.random.choice(self.num_items, size=N, replace=False)
        batch = np.array([self.buffer[i,:] for i in indeces])

        return batch

    # Prioritised experience replay
    def prioritised_sample(self, N):
        self.sampled_idx = np.random.choice(self.num_items, size=N, replace=False, p=self.probs[0:self.num_items])
        batch = np.array([self.buffer[i,:] for i in self.sampled_idx])

        return batch

    def update_probabilities(self, alpha=1):
        weights_sum = np.sum(self.weights**alpha)
        self.probs = self.weights**alpha / weights_sum

    def update_weights(self, deltas, constant=0.001):
        for i in range(len(self.sampled_idx)):
            idx = int(self.sampled_idx[i])
            self.weights[idx] = abs(deltas[i]) + constant

    def get_len(self):
        return self.num_items

# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 128 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=128)
        self.layer_2 = torch.nn.Linear(in_features=128, out_features=128)
        self.output_layer = torch.nn.Linear(in_features=128, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input_):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input_))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self, learning_rate):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=3)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        # Create target network with all weights to zero
        self.target_network = Network(input_dimension=2, output_dimension=3)
        self.zero_target()

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transitions, gamma=0, do_target=False, do_double_q=False):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss, delta = self._calculate_loss(transitions, gamma, do_target)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # clipping gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(),1)
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item(), delta

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transitions, gamma=0, use_target=False, use_double_q=False):
        input_tensor = torch.from_numpy(transitions[:, [0,1]].astype(np.float32))
        actions = torch.from_numpy(transitions[:, 2].astype(np.int64))
        rewards = torch.from_numpy(transitions[:, 3].astype(np.float32))
        next_tensor = torch.from_numpy(transitions[:, [4,5]].astype(np.float32))

        if use_target == True:
            future_q_search_arg =  self.target_network.forward(next_tensor)
            future_q_search_arg = future_q_search_arg.detach()
            best_future_actions = torch.max(future_q_search_arg, 1).indices
            if use_double_q:
                future_q = self.q_network.forward(next_tensor)
            else:
                future_q = future_q_search_arg
            future_q = future_q.detach()
            future_returns = future_q.gather(dim=1, index=best_future_actions.unsqueeze(-1)).squeeze(-1)
            labels_tensor = rewards + (gamma * future_returns)
        elif gamma == 0:
            labels_tensor = rewards
        else:
            future_q =  self.q_network.forward(next_tensor)
            best_future_actions = torch.max(future_q, 1).indices
            future_returns = future_q.gather(1, index=best_future_actions.unsqueeze(-1)).squeeze(-1)
            labels_tensor = rewards + (gamma * future_returns)

        network_prediction = self.q_network.forward(input_tensor).gather(1, index=actions.unsqueeze(-1)).squeeze(-1)
        delta = labels_tensor - network_prediction
        loss = torch.nn.MSELoss()(network_prediction, labels_tensor)

        return loss, delta

    def get_q_values(self, state):
        input_tensor = torch.tensor(state).unsqueeze(0)
        q_values = self.q_network.forward(input_tensor)
        q_values = q_values.detach()

        return q_values

    def target_update(self):
        q_dictionary = self.q_network.state_dict()
        self.target_network.load_state_dict(q_dictionary)

    def zero_target(self):
        for name, params in self.target_network.named_parameters():
            tensor_size = list(params.size())
            params.data.copy_(torch.zeros(tensor_size))


class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 250
        self.episode_floor = 250
        self.episode_decay = 200
        self.episode_length_update = 1
        self.episode_counter = -1
        self.do_decay_episode_length = False

        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        self.steps_counter = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None

        # Define parameters used during training
        lr = 0.0025
        self.gamma = 0.91
        self.alpha_prioritised_buffer = 0.5

        # Calculate reward
        self.do_personalised_reward = True
        self.penalize_walls = True
        # self.personalised_rewards = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
        self.personalised_rewards = [7, 6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2]
        # self.personalised_rewards = [4, 3, 2.5, 2, 1.5, 1, 0.8, 0.7, 0.6, 0.5, 0.15]

        # Episode with higher epsilon
        self.do_increase_epsilon = True
        self.epsilon_threshold = 0.1
        self.increase_epsilon_occurrence = 10
        self.increase_epsilon_episodes = 2
        self.increased_epsilon = 0.15
        self.prev_epsilon = 1000
        self.epsilon_increase_episode_counter = -1

        # Epsilon
        self.epsilon_which = 2
        self.epsilon = 1
        self.epsilon_min = 0
        self.epsilon_step = 0.000015
        self.epsilon_decay_factor = 0.98
        self.epsilon_update = 300

        # Initialise an empy replay buffer
        self.do_prioritised = True
        self.replay_buffer = ReplayBuffer(5000, self.alpha_prioritised_buffer, self.do_prioritised)
        self.batch_size = 100
        self.prioritised_constant = 0.001

        # Initialise network
        self.dqn = DQN(lr)
        self.target_update = 1 #Number of episodes after which target gets updated
        self.do_target = True
        self.do_double_q = True

    def epsilon_decay(self, which):
        # Epsilon decay after number of steps by a factor with minimum
        if which == 0:
            if (self.num_steps_taken % self.epsilon_update == 0) and (self.epsilon > self.epsilon_min):
                self.epsilon = self.epsilon * self.epsilon_decay_factor

        # Epsilon decay after number of steps by a step with minimum
        if which == 1:
            if (self.num_steps_taken % self.epsilon_update == 0) and (self.epsilon > self.epsilon_min):
                self.epsilon = self.epsilon - self.epsilon_step

        # Epsilon decay after each episode by a factor with minimum
        if which == 2:
            if (self.steps_counter == self.episode_length) and (self.epsilon > self.epsilon_min):
                self.epsilon = self.epsilon * self.epsilon_decay_factor
                
                # Do periodic episodes with higher epsilon
                if self.do_increase_epsilon and ((self.epsilon < self.epsilon_threshold) or self.prev_epsilon != 1000):
                    if self.epsilon_increase_episode_counter == -1:
                        self.epsilon_increase_episode_counter = 1
                    elif self.epsilon_increase_episode_counter <= self.increase_epsilon_occurrence:
                        self.epsilon_increase_episode_counter += 1
                        self.prev_epsilon = self.epsilon
                    elif self.epsilon_increase_episode_counter <= (self.increase_epsilon_occurrence+self.increase_epsilon_episodes):
                        self.epsilon = self.increased_epsilon
                        self.epsilon_increase_episode_counter += 1
                    else:
                        self.epsilon_increase_episode_counter = 1
                        self.epsilon = self.prev_epsilon
                        self.prev_epsilon == 1000
                        self.epsilon_decay(self.epsilon_which)

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.steps_counter % self.episode_length == 0:
            self.steps_counter = 0
            self.episode_counter += 1
            print(f'\nEpisode {self.episode_counter+1} and Epsilon {self.epsilon}')

            if self.do_decay_episode_length:
                if ((self.episode_counter+1)%self.episode_length_update==0) and (self.episode_length>self.episode_floor):
                    self.episode_length -= self.episode_decay

            if self.episode_counter % self.target_update == 0:
                self.dqn.target_update()
                print('Target updated')

            return True
        else:
            return False

    def epsilon_greedy_action(self):
        q_values = self.dqn.get_q_values(self.state)
        best_action = np.argmax(q_values.tolist()[0])

        toss = np.random.uniform(0, 1)
        if toss > self.epsilon:
            return best_action
        else:
            return np.random.randint(3)
    
    def best_action(self, state):
        q_values = self.dqn.get_q_values(state)
        action = np.argmax(q_values.tolist()[0])
        return action

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:
            continuous_action = np.array([0.02, 0], dtype=np.float32) # 0: R --> Move 0.1 to the right, and 0 upwards
        elif discrete_action == 1:
            continuous_action = np.array([0, -0.02], dtype=np.float32) # 0: D --> Move 0 laterally, and 0.1 downwards
        elif discrete_action == 2:
            continuous_action = np.array([0, 0.02], dtype=np.float32) # 0: U --> Move 0 laterally, and 0.1 upwards
            
        return continuous_action

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # action = np.random.uniform(low=-0.01, high=0.01, size=2).astype(np.float32)
        self.num_steps_taken += 1
        self.steps_counter += 1
        if self.num_steps_taken % 50 == 0:
            print(f'Step {self.steps_counter}')
        
        self.state = state

        action = self.epsilon_greedy_action()
        self.action = action
        action = self._discrete_action_to_continuous(action)

        return action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
        reward = self.calculate_reward(distance_to_goal, self.state, next_state)

        transition = (self.state[0], self.state[1], self.action, reward, next_state[0], next_state[1])

        self.replay_buffer.add(transition)

        if self.replay_buffer.get_len() >= self.batch_size:
            batch_transitions = self.replay_buffer.sample(self.batch_size)
            loss, delta = self.dqn.train_q_network(batch_transitions, self.gamma, self.do_target, self.do_double_q)

            if self.do_prioritised:
                self.replay_buffer.update_weights(delta, self.prioritised_constant)

            self.epsilon_decay(self.epsilon_which)

    def calculate_reward(self, distance, current_state, next_state):
        if self.do_personalised_reward:
            ranges = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5])
            range_state = np.argmax(ranges > distance)
            reward_multiplier = self.personalised_rewards[range_state]
            if self.penalize_walls and (current_state == next_state).all():
                reward_multiplier /= 2
            return reward_multiplier * (1.5 - distance)
        else:
            return (1 - distance)

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        action = self.best_action(state)
        action = self._discrete_action_to_continuous(action)
        return action
