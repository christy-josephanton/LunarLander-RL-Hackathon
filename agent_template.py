# agent_template.py

import gymnasium as gym
from state_discretizer import StateDiscretizer
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random


# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class DQN(nn.Module):

    def __init__(self, num_states, num_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(num_states, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class LunarLanderAgent:
    def __init__(self, render_mode=None, debug=False):
        """
        Initialize your agent here.

        This method is called when you create a new instance of your agent.
        Use this method to Initializes the environment, the agent’s model (e.g., Q-table or neural network),
        and the optional state discretizer for Q-learning. Add any necessary initialization for model parameters here.
        """
        # TODO: Initialize your agent's parameters and variables

        # Initialize environment
        self.env = gym.make('LunarLander-v3', render_mode=render_mode)
        self.debug = debug # enables print statements

        # Set learning parameters
        self.epsilon = 1.0        # Initial exploration rate
        self.epsilon_min = 0.01        # Initial exploration rate
        self.epsilon_decay = 0.995   # Exploration decay rate
        
        # Initialize any other parameters and variables
        self.learning_rate =0.001
        self.batch_size = 64
        self.mem_size = 10
        self.gamma = 0.99
        self.update_dqn_target_step = 10
        self.step = 0

        # Initialize Neutral Networks
        self.dqn = DQN(self.env.observation_space.shape[0], self.env.action_space.n)
        self.dqn_target = DQN(self.env.observation_space.shape[0], self.env.action_space.n)

        # Match two DQN Network weights/biases
        self.dqn_target.load_state_dict(self.dqn.state_dict())

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.learning_rate)

        self.best_params = self.dqn.state_dict()

        # Initialize Replay Buffer for batching
        self.replay_buffer = deque(maxlen=10000)

        pass

    def select_action(self, state):
        """
        Given a state, select an action to take. The function should operate in training and testing modes,
        where in testing you will need to shut off epsilon-greedy selection.

        Args:
            state (array): The current state of the environment.

        Returns:
            int: The action to take.
        """
             
        if (np.random.rand() < self.epsilon): #if less than epsilon, explore (random action)
            return self.env.action_space.sample() 
        
        else: #if not, select action based off memory
    
            # convert to state to tensor, pass to model, get prediction, return
            state_tensor = torch.tensor(state, dtype=torch.float32)  # Convert state to tensor
            with torch.no_grad():
                q_values = self.dqn(state_tensor)  # get prediction
            return int(np.argmax(q_values.numpy()))  # convert tensor->np, get action


            
    def train(self, num_episodes):
        """
         Contains the main training loop where the agent learns over multiple episodes.

        Args:
            num_episodes (int): Number of episodes to train for.
        """
        all_rewards = []
        best_reward = -np.inf

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            # decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            #add to master list and find average for last 100
            all_rewards.append(total_reward)
            average_reward = np.mean(all_rewards[-100:])

            if self.debug: print(f"Episode {episode + 1}/{num_episodes} - Reward: {total_reward:.10f}, Average Reward: {average_reward:.10f}, Epsilon: {self.epsilon:.5f}")

            # autosave the best model
            if average_reward >= best_reward :
                if self.debug: print("Autosaved")
                best_reward = average_reward
                self.best_params = self.dqn.state_dict()

        
        pass

    def update(self, state, action, reward, next_state, done):
        """
        Update your agent's knowledge based on the transition.

        Args:
            state (array): The previous state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (array): The new state after the action.
            done (bool): Whether the episode has ended.
        """
        # Add step to the replay buffer
        self.replay_buffer.append((state, action, reward, next_state, done))

        # train if enough samples are available
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch of transitions from the replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # convert data to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).view(-1, 1)  # column vector
        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)  # column vector
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).view(-1, 1)  # column vector

        # Compute current Q-values
        q_values = self.dqn(states)
        current_q_values = q_values.gather(1, actions)  # Gather Q-values corresponding to actions

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.dqn_target(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Update parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.step += 1
        if self.step % self.update_dqn_target_step == 0:
            self.dqn_target.load_state_dict(self.dqn.state_dict())

        pass
    
    def test(self, num_episodes = 100):
        """
        Test your agent locally before submission to get a hint of the expected score.

        Args:
            num_episodes (int): Number of episodes to test for.
        """
        # TODO: Implement your testing loop here
        # Make sure to:
        # Store the cumulative rewards (return) in all episodes and then take the average 

        rewards = []   
        self.epsilon = 0 #no chance of explore
        for episode in range(num_episodes):
            state, info = self.env.reset()
            reward_accum = 0
            done = False
            while not done:
                action = self.select_action(state)
                state, reward, terminated, truncated, info = self.env.step(action)
                reward_accum+=reward

                done = terminated or truncated
        
            if self.debug: print(f"TEST Episode {episode + 1}/{num_episodes} - Reward: {reward_accum:.10f}, Average Reward: {(np.mean(rewards) if rewards else 0.0):.10f}, Epsilon: {self.epsilon:.5f}")

            rewards.append(reward_accum)

        return np.mean(rewards)

    def save_agent(self, file_name):
        """
        Save your agent's model to a file.

        Args:
            file_name (str): The file name to save the model.
        """
        torch.save(self.best_params, file_name)

        pass

    def load_agent(self, file_name):
        """
        Load your agent's model from a file.

        Args:
            file_name (str): The file name to load the model from.
        """
        self.dqn.load_state_dict(torch.load(file_name))
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        if self.debug: print(f"Model loaded from {file_name}.")

        pass

if __name__ == '__main__':


    # agent = LunarLanderAgent()
    agent = LunarLanderAgent(debug=True)
    #agent = LunarLanderAgent(render_mode="human", debug=True) # much slower but shows visuals 

    agent_model_file = 'model.pkl'  # Set the model file name

    # Example usage:
    # Uncomment the following lines to train your agent and save the model

    num_training_episodes = 200  # Define the number of training episodes
    print("Training the agent...")
    agent.train(num_training_episodes)
    print("Training completed.")

    # Save the trained model
    agent.save_agent(agent_model_file)
    print("Model saved.")

    #Test Trained Model
    test_agent = LunarLanderAgent(debug=True)
    #test_agent = LunarLanderAgent(render_mode="human", debug=True) # much slower but shows visuals 
    num_testing_episodes = 100  # Define the number of testing episodes
    test_agent.load_agent(agent_model_file)
    average_reward = test_agent.test(num_testing_episodes)
    print('Average Test Reward:',average_reward)