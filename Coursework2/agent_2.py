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

import numpy as np
import torch
import collections
import time

class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 250
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # Setting dimension of action space (degrees of freedom the agent has)
        self.action_space = 4
        # Initializing epoch number
        self.epoch = 0
        # Select minibatch size
        self.minibatch_size = 100
        # Initialize epsilon
        self.epsilon = 1
        # Initialize helper epsilon variable
        self.prev_epsilon = 1
        # Initializing episode epsilon-decay rate 
        self.eps_decay_ep = 0.0005
        # Initializing step epsilon-decay rate 
        self.eps_decay_step = 0
        # Initializing minimum step epsilon value
        self.min_step_eps = 0.01
        # Initialize minibatch array
        self.minibatch = []
        # Initializing weighting of reward depending on distance to goal
        self.weight_reward = [0.5, 0.6, 0.7, 0.8, 1, 1.5, 2, 2.5, 3, 4]
        ####################################################################
        
        ################## PRIORITESD EXPERIENCE REPLAY BUFFER ##################
        # Initialize buffer size
        self.buffer_size = 5000
        # Decide whether we want to use Prioritised Experience Replay
        self.prioritised_replay = True
        # Initialize ReplayBuffer object
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.prioritised_replay)
        # Initialize weights for Prioritised Experience Replay
        self.weights = np.zeros((self.buffer_size, ))
        # Initialize probabilities for Prioritised Experience Replay
        self.probs = np.zeros((self.buffer_size, ))
        self.probs[0:self.minibatch_size] = 1/self.minibatch_size
        # Initialize parameter epsilon for determining weight.
        self.constant_weight = 0.0001
        # Initialize parameter alpha for weighting probabilities for Prioritised Experience Replay
        self.alpha = 0.01
        self.deltas = []
        ####################################################################

        ################## EPSILON DECAY MODES ##################
        self.dacay_ep_and_step = False
        self.decay_step_only = True
        self.decay_ep_only = False
        ####################################################################

        # Decide whether to train the grredy policy every n steps
        self.train_greedy = False
        # Decide whether to penilize reward for hitting the wall
        self.penalize_wall = True
        # Decide whether to weight reward based on x location of the agent
        self.weighted_reward = True
        # Decide whether to use Double-Q Learning or not
        self.double_q_learning = True
        # Initialize DQN object
        self.dqn = DQN(lr=0.001, action_space=self.action_space, double_q=self.double_q_learning)
        ####################################################################
        
    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            return True
        else:
            return False

    # Function to update the weights of the target network at the end of every episode
    def update_target_network_weights(self):
        # TODO: This can be changed to a larger period (in terms of epochs instead)
        if self.has_finished_episode() and self.num_steps_taken != 0:
            self.dqn.update_target_network()

    # Function to update epsilon in different modes
    def update_epsilon(self):
        if self.dacay_ep_and_step:
            # Update EPISODE epsilon value 
            if self.has_finished_episode():
                self.prev_epsilon = self.prev_epsilon - self.eps_decay_ep*self.prev_epsilon
                self.epsilon = self.prev_epsilon
            # Update STEP epsilon decay rate
            self.eps_decay_step = (self.prev_epsilon - self.min_step_eps)/self.episode_length
            self.epsilon = self.epsilon - self.eps_decay_step*self.epsilon

        elif self.decay_step_only:
            # Update STEP epsilon decay rate
            self.eps_decay_step = 0.000015
            if self.num_steps_taken > 230*self.episode_length:
                self.eps_decay_step = 0.00006
                self.epsilon = self.epsilon - self.epsilon * self.eps_decay_step
            else:
                self.eps_decay_step = 0.000015
                self.epsilon = self.epsilon - self.epsilon * self.eps_decay_step
            
        elif self.decay_ep_only:
            # Update EPISODE epsilon decay rate
            if self.has_finished_episode():
                self.eps_decay_ep = 0.03
                self.epsilon = self.epsilon - self.eps_decay_ep*self.epsilon

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # train on greedy policy every n steps
        if self.train_greedy:
            if self.num_steps_taken % 50000 == 0 and self.num_steps_taken > 0:
                action = self.get_action_from_prediction(state)
                
        # choosing action based on epsilon-greedy policy
        rand_count = np.random.rand(1)
        if rand_count > self.epsilon:
            action = self.get_action_from_prediction(state)
        else:
            action = np.random.choice(np.arange(4))

        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action
        # Update step epsilon value
        self.update_epsilon()
        # Update epoch number
        if self.has_finished_episode():
            self.epoch += 1
            print('EPSILON: ', self.epsilon)
            print('EPOCH: ', self.epoch)
        return self._discrete_action_to_continuous(action)

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
        if self.weighted_reward:
            idx = 0
            spaces = [i/10 for i in range(10)]
            for j, space in enumerate(spaces):
                if (self.state[0]) > 0.1:
                    if space - (self.state[0])> 0:
                        idx = j
                        break
                    else:
                        idx = 0
            k = self.weight_reward[idx]
            if distance_to_goal < 0.1:
                k = 7
            if distance_to_goal < 0.03:
                if (self.num_steps_taken - ((self.epoch)*self.episode_length)) < 150:
                    print('STEP TO REACH GOAL: ', self.num_steps_taken - ((self.epoch)*self.episode_length))
            if self.penalize_wall:
                if (np.array(self.state) == np.array(next_state)).all():
                    k = 0.01
                
            reward = k*0.1*(1 - distance_to_goal)
        else:
            reward = 1 - distance_to_goal

        # Create a transition
        transition = (self.state[0], self.state[1], self.action, reward, next_state[0], next_state[1])
        # Adding transition to the replay buffer
        self.replay_buffer.add_transition(transition, self.num_steps_taken)
        # if prioritised replay is used 
        if self.prioritised_replay:
            # Adding weight to weights deque
            self.add_weight()
            # Carrying out update of the Q values 
            delta = self.update()
            # Updating buffer weights
            self.update_buffer_weights(self.minibatch, delta)
            # Updating probabiliy vector:
            self.update_prob_vector()
        else:
            self.update()
        # Updating weights of target network
        self.update_target_network_weights()

    # Function that carries out update on the DQN
    def update(self):
        # carrying out update if the size of the replay buffer is larger than minibacth size
        if self.num_steps_taken >= self.minibatch_size:
            self.minibatch = self.replay_buffer.get_minibatch(self.minibatch_size, self.probs, self.num_steps_taken)
            loss, delta = self.dqn.train_q_network(self.minibatch)
            return delta

    # Function that adds the weight of the transition to the weight array
    def add_weight(self):
        # Assigning a random weight bewteen 0 and 1 to the very first weight
        if self.num_steps_taken == 1:
            self.weights[self.num_steps_taken-1] = np.random.rand(1)
        elif self.num_steps_taken <= self.buffer_size:
            self.weights[self.num_steps_taken-1] = max(self.weights)
        else:
            self.weights[(self.num_steps_taken-1) % self.buffer_size] = max(self.weights)
    
    # Function that updates the weights of the buffer after each loss update
    def update_buffer_weights(self, minibatch, delta):
        if self.num_steps_taken >= self.minibatch_size:
            # retrieve indeces of the transition present in the minibatch inside the buffer
            idxs = self.replay_buffer.get_minibatch_indices()
            for j, i in enumerate(idxs):
                self.weights[i] = abs(delta[j]) + self.constant_weight

    # Function that updates the probability vector of the buffer after each loss update
    def update_prob_vector(self):
        if self.num_steps_taken >= self.minibatch_size:
            self.probs = self.weights**(self.alpha)/np.sum(self.weights**self.alpha)

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        # Getting Q-values for the input state
        state_prediction = self.dqn._get_state_prediction(state).unsqueeze(0)
        # retrieving action corresponding to highest Q-value
        max_q_val, action = torch.max(state_prediction, 1)
        return self._discrete_action_to_continuous(action)

    # Function to get the greedy DISCRETE action for a particular state
    def get_action_from_prediction(self, state):
        # Getting Q-values for the input state
        state_prediction = self.dqn._get_state_prediction(state).unsqueeze(0)
        # retrieving action corresponding to highest Q-value
        max_q_val, action = torch.max(state_prediction, 1)
        return action

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        # MOVE 
        if discrete_action == 0:
            # Move 0.2 to the right, and 0 upwards
            continuous_action = np.array([0.02, 0], dtype=np.float32)
        if discrete_action == 1:
            # Move 0 to the right, and -0.2 upward
            continuous_action = np.array([0, -0.02], dtype=np.float32)
        if discrete_action == 2:
            # Move -0.2 to the right, and 0 upward
            continuous_action = np.array([-0.02, 0], dtype=np.float32)
        if discrete_action == 3:
            # Move 0 to the right, and 0.2 upward
            continuous_action = np.array([0, 0.02], dtype=np.float32)
        return continuous_action 

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
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output

# The DQN class determines how to train the above neural network.
class DQN:
    # The class initialisation function.
    def __init__(self, lr, action_space, double_q):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=action_space)
        # Create a target network
        self.target_network = Network(input_dimension=2, output_dimension=action_space)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.double_q = double_q
   
    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, minibatch):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss, delta = self._calculate_loss(minibatch)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # clipping gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(),1)
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item(), delta
    
    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, minibatch, gamma=0.9):
        # retrieving predictions
        exp_reward_tensors, predictions, next_state_label, delta = self.get_preds_from_batch(minibatch)
        # Calculate loss using predictions
        loss = torch.nn.MSELoss()(predictions, exp_reward_tensors + gamma*(next_state_label))
        return loss, delta
    
    # Function that computes the predictions used for loss calculation
    def get_preds_from_batch(self, minibatch, gamma=0.95):
        # retrive transition elements from minibatch and convert to torch.tensor objects
        input_state_tensors = torch.from_numpy(minibatch[:, [0,1]])
        action_reward_tensors =  torch.from_numpy(minibatch[:, [2]].astype(np.int64))
        exp_reward_tensors = torch.from_numpy(minibatch[:, [3]])
        successor_state_tensors =  torch.from_numpy(minibatch[:, [4,5]])

        # reformatting expected reward tensors
        exp_reward_tensors = exp_reward_tensors.squeeze(-1)
        
        # Getting predicted Q values tensor using q-network for input states
        pred_tensor = self.q_network.forward(input_state_tensors)
        predictions = torch.gather(pred_tensor, dim=1, index=action_reward_tensors).squeeze(-1)
    
        # Getting predicted Q values tensor from successor state using target-network 
        with torch.no_grad():
            next_state_prediction = self.target_network.forward(successor_state_tensors)
            next_state_prediction = next_state_prediction.detach()

        #Getting predicted Q values tensor from successor state using Q network (DOUBLE-Q)
        if self.double_q:
            with torch.no_grad():
                next_state_prediction_Q = self.q_network.forward(successor_state_tensors)
        
        # Getting expected best action from successor states 
        if self.double_q:
            next_state_label_Q, indices_Q =  torch.max(next_state_prediction_Q, 1)
        else:
            next_state_label, indices = torch.max(next_state_prediction, 1)
        
        # Decide which delta to return based on whether we are using DOUBLE-Q or not
        if self.double_q:
            Q_preds = torch.gather(next_state_prediction, dim=1, index=indices_Q.unsqueeze(-1)).squeeze(-1)
            delta = exp_reward_tensors + gamma*(Q_preds) - predictions
        else:
            delta = exp_reward_tensors + gamma*(next_state_label) - predictions

        if self.double_q:
            return exp_reward_tensors, predictions, Q_preds, delta.detach().numpy()
        else:
            return exp_reward_tensors, predictions, next_state_label, delta.detach().numpy()

    # Function that resturns the predictions for a given state
    def _get_state_prediction(self, input_state):
        input_state_tensor = torch.tensor(input_state, dtype=torch.float32)
        return self.q_network.forward(input_state_tensor)

    # Function that update the weight of the target network 
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# The ReplayBuffer class takes care minibatches
class ReplayBuffer:
    # The class initialisation function.
    def __init__(self, buffer_size, prioritised_replay):
        self.buffer = np.zeros((buffer_size, 6), dtype=np.float32)
        self.buffer_size = buffer_size
        self.prioritised_replay = prioritised_replay
        
    # Function to add transition to minibatch deque object 
    def add_transition(self, transition, idx):
        if idx <= self.buffer_size:
            self.buffer[idx-1, :] = transition
        else:
            idx_ = (idx-1) % self.buffer_size
            self.buffer[idx_, :] = transition
    
    # Function to return minibatch list type
    def get_minibatch(self, minibatch_size, p_vector, max_idx):
        self.minibatch_indices = np.zeros((minibatch_size, ), dtype=np.int64)
        if self.prioritised_replay:
            # Choose minimbatch indices based on probability vector p_vector
            if max_idx <= self.buffer_size:
                self.minibatch_indices = np.random.choice(max_idx, minibatch_size, replace=False, p=(p_vector)[0:(max_idx)])
            else:
                self.minibatch_indices = np.random.choice(self.buffer_size, minibatch_size, replace=False, p=(p_vector)[0:self.buffer_size])
        else:
            self.minibatch_indices = np.random.randint(len(self.buffer), size=minibatch_size)

        # print(np.transpose(self.minibatch_indices))
        minibatch_inputs = np.array([self.buffer[i, :] for i in self.minibatch_indices])
        return minibatch_inputs  

    # Utility Function that return the current minibatch indices of the buffer 
    def get_minibatch_indices(self):
        return self.minibatch_indices

    # Utility Function to get size of the buffer
    def get_size(self):
        return len(self.buffer)