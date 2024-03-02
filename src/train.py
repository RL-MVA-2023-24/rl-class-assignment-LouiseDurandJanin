from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
from tqdm import tqdm
import joblib
import torch
import random
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from copy import deepcopy
from evaluate import evaluate_HIV, evaluate_HIV_population

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

class DQN_model(nn.Module):
    def __init__(self, nb_neurons, state_dim, nb_actions):
        super(DQN_model, self).__init__()
        self.nb_neurons = nb_neurons 
        self.state_dim = state_dim
        self.relu = nn.ReLU()
        self.nb_actions = nb_actions
        self.linear1 = nn.Linear(self.state_dim, self.nb_neurons)
        self.linear2 = nn.Linear(self.nb_neurons, self.nb_neurons)
        self.linear3 = nn.Linear(self.nb_neurons, self.nb_neurons)
        self.linear4 = nn.Linear(self.nb_neurons, self.nb_actions)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x




    
class ProjectAgent:
    def __init__(self):
        self.device = "cpu"
        self.env = env
        self.gamma = 0.95
        self.batch_size = 256
        self.nb_actions = 4
        self.memory = ReplayBuffer(100000, self.device)
        self.epsilon_max = 1.
        self.epsilon_min = 0.01
        self.epsilon_stop = 1000
        self.epsilon_delay = 20
        self.epsilon_step = 0.005
        self.state_dim = self.env.observation_space.shape[0]
        self.nb_neurons=256
        self.model = DQN_model(self.nb_neurons, self.state_dim, self.nb_actions)
        self.target_model = deepcopy(self.model).to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.nb_gradient_steps =  1
        self.update_target_strategy = 'replace'
        self.update_target_freq = 50
        self.update_target_tau = 0.005
    
        self.path = "model3.pth"
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    def greedy_action(self,network, state):
        with torch.no_grad():
            Q = network(torch.Tensor(state).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()
        
    def train(self, env, max_episode, nb_samples):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        best=0

        # Fill replay memory with random steps
        
        for _ in range(nb_samples):
            # select epsilon-greedy actio
            action = self.env.action_space.sample()
            next_state, reward, done, trunc, _ = self.env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            if done:
                state, _ = self.env.reset()
            else:
                state = next_state
        print("Sampling done")
        
        
        while episode < max_episode:
            # update epsilon
            
            if step > self.epsilon_delay:
              epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
              action = env.action_space.sample()
            else:
              action = self.greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            #self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            #print("step", step)
            step += 1
            if done or trunc or step==200:
                print("Episode", episode, "finished")
                episode += 1
                step=0
                if episode % 20==0:
                  self.save("model3.pth")
                  print("Episode ", '{:3d}'.format(episode), 
                        ", epsilon ", '{:6.2f}'.format(epsilon), 
                        ", length replay memory ", '{:5d}'.format(len(self.memory)), 
                        ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                        sep='')
                  print("saved")

               
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return

    

    
    def act(self, observation, use_random=False):
 
        if use_random:
            action = self.env.action_space.sample()
        else:
            action = self.greedy_action(self.model, observation)

        return action

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        self.path = path

    def load(self):
        if self.path:
            self.model.load_state_dict(torch.load(self.path))
        else:
            print(f"File not found at path: {self.path}. Skipping loading.")

if __name__ =="__main__":
    agent = ProjectAgent()
    agent.load()
    # Train the agent
    max_episode = 800  # You can adjust this value
    episode_returns = agent.train(env,max_episode, 8000)

    # Save the trained model
    agent.save("model_final_5layers.pth")

