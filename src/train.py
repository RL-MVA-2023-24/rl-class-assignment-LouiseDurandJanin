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
import matplotlib.pyplot as plt 
import torch.nn.functional as F

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
        self.linear4 = nn.Linear(self.nb_neurons, self.nb_neurons)
        self.linear5 = nn.Linear(self.nb_neurons, self.nb_actions)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        return x




    
class ProjectAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.path = "best_model-5.pt"
        self.nb_actions = 4
        self.gamma = 0.95
        self.batch_size = 800
        buffer_size = 100000
        self.memory = ReplayBuffer(buffer_size,self.device)
        self.epsilon_max =  1.
        self.epsilon_min =  0.05
        self.epsilon_stop =  20000
        self.epsilon_delay = 100
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.nb_neurons = 512
        self.state_dim = env.observation_space.shape[0]
        self.model = DQN_model(self.nb_neurons, self.state_dim, self.nb_actions)
        self.target_model = deepcopy(self.model).to(self.device)
        self.criterion = torch.nn.SmoothL1Loss()
        lr = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = 8
        self.update_target_strategy = 'replace'
        self.update_target_freq = 50
        self.update_target_tau = 0.005
        self.monitoring_nb_trials = 50

    def MC_eval(self, env, nb_trials):   # NEW NEW NEW
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in range(nb_trials):
            x,_ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            best=0
            while not (done or trunc):
                a = greedy_action(self.model, x)
                y,r,done,trunc,_ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma**step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)
    
    def V_initial_state(self, env, nb_trials):   # NEW NEW NEW
        with torch.no_grad():
            for _ in range(nb_trials):
                val = []
                x,_ = env.reset()
                val.append(self.model(torch.Tensor(x).unsqueeze(0).to(self.device)).max().item())
        return np.mean(val)
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            #QYmax = self.target_model(Y).max(1)[0].detach()
            #update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            #QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            #loss = self.criterion(QXA, update.unsqueeze(1))
            #self.optimizer.zero_grad()
            #loss.backward()
            #self.optimizer.step() 
            q_online = self.model(Y)
            action_q_online = torch.argmax(q_online, dim=1)
            q_target = self.target_model(Y)
            ddqn_q = torch.sum(q_target * F.one_hot(action_q_online, 4), dim=1)
            expected_q = R + self.gamma * ddqn_q * (1.0 - D)

            main_q = torch.sum(self.model(X) * F.one_hot(A.long(), num_classes = 4), dim=1)

            loss = self.criterion(expected_q.detach(), main_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, env, max_episode):
        episode_return = []
        MC_avg_total_reward = []   # NEW NEW NEW
        MC_avg_discounted_reward = []   # NEW NEW NEW
        V_init_state = []   # NEW NEW NEW
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        best=0
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
                    target_state_dict[key] = tau*model_state_dict + (1-tau)*target_state_dict
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                # Monitoring
                self.save(self.path)
                print("model saved")
                score = evaluate_HIV(agent=self, nb_episode=1)
                if self.monitoring_nb_trials>0:
                    if score > best:
                        best = score
                        self.save("best_model.pt")
                        print(" New model saved")
                    #MC_dr, MC_tr = self.MC_eval(env, self.monitoring_nb_trials)    # NEW NEW NEW
                    #V0 = self.V_initial_state(env, self.monitoring_nb_trials)   # NEW NEW NEW
                    #MC_avg_total_reward.append(MC_tr)   # NEW NEW NEW
                    #MC_avg_discounted_reward.append(MC_dr)   # NEW NEW NEW
                    #V_init_state.append(V0)   # NEW NEW NEW
                    episode_return.append(episode_cum_reward)   # NEW NEW NEW
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward),
                           ", score ", '{:6.2f}'.format(score),
                          #", MC tot ", '{:6.2f}'.format(MC_tr),
                         # ", MC disc ", '{:6.2f}'.format(MC_dr),
                         # ", V0 ", '{:6.2f}'.format(V0),
                          sep='')
                else:
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                          sep='')

                
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state

    

    
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
            self.model.load_state_dict(torch.load(self.path, map_location=torch.device('cpu')))
        else:
            print(f"File not found at path: {self.path}. Skipping loading.")

if __name__ =="__main__":
    config = {'nb_actions': 4,
          'learning_rate': 0.001,
          'gamma': 0.95,
          'buffer_size': 1000000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 1000,
          'epsilon_delay_decay': 20,
          'batch_size': 256,
          'gradient_steps': 2,
          'update_target_strategy': 'replace', # or 'ema'
          'update_target_freq': 50,
          'update_target_tau': 0.005,
          'criterion': torch.nn.SmoothL1Loss(),
          'monitoring_nb_trials': 50}
    agent = ProjectAgent()
    agent.load()
    # Train the agent
    max_episode = 800  # You can adjust this value
    # Save the trained model


    ep_length, disc_rewards, tot_rewards, V0 = agent.train(env, 800)
    '''
    plt.plot(ep_length, label="training episode length")
    plt.plot(tot_rewards, label="MC eval of total reward")
    plt.legend()
    plt.figure()
    plt.plot(disc_rewards, label="MC eval of discounted reward")
    plt.plot(V0, label="average $max_a Q(s_0)$")
    plt.legend();
    '''

