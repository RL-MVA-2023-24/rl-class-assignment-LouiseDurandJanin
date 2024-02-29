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
def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()
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
    
class ProjectAgent:
    def __init__(self):
        self.device = "cpu"
        self.env = env
        self.gamma = 0.95
        #self.batch_size = 100
        self.batch_size = 20
        self.nb_actions = 4
        #self.memory = ReplayBuffer(100000, self.device)
        self.memory = ReplayBuffer(1000000, self.device)
        self.epsilon_max = 1.
        self.epsilon_min = 0.01
        self.epsilon_stop = 1000
        self.epsilon_delay = 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.state_dim = self.env.observation_space.shape[0]
        self.nb_neurons=24
        self.model = torch.nn.Sequential(nn.Linear(self.state_dim, self.nb_neurons),
                          nn.ReLU(),
                          nn.Linear(self.nb_neurons, self.nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(self.nb_neurons, self.nb_actions)).to(self.device)
        #self.target_model = deepcopy(self.model).to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        #self.nb_gradient_steps =  1
        #self.update_target_strategy = 'replace'
        #self.update_target_freq = 20
        #self.update_target_tau = 0.005
    
        self.path = "model_final.pth"
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.model(Y).max(1)[0].detach()
            #update = torch.addcmul(R, self.gamma, 1-D, QYmax)
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        best = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)

            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # train
            self.gradient_step()

            # next transition
            step += 1
            if done or trunc:
                episode += 1
                current_score_agent = evaluate_HIV(self)
                score_pop = evaluate_HIV_population(self)
                if current_score_agent > best:
                    best = current_score_agent
                    print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                       ", score agent ", '{:4.1f}'.format(current_score_agent),
                        ", score population ", '{:4.1f}'.format(score_pop),
                      sep='')
                    self.save("model.pth")
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
            action = greedy_action(self.model, observation)

        return action

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        self.path = path

    def load(self):
        if self.path:
            self.model.load_state_dict(torch.load(self.path))
        else:
            print(f"File not found at path: {self.path}. Skipping loading.")

'''
agent = ProjectAgent()
agent.load()
# Train the agent
max_episode = 10  # You can adjust this value
episode_returns = agent.train(env,max_episode)

# Save the trained model
agent.save("model_final.pth")
'''