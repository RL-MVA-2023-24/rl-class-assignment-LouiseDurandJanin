from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
from tqdm import tqdm
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):
        super(ProjectAgent, self).__init__()
        self.env = env 
        self.S, self.A, self.R, self.S2, self.D = self.collect_samples()
        print(self.S.shape[0])
        self.Qfunction = self.rf_fqi()
        self.path = None

    def collect_samples(self, horizon = int(200), disable_tqdm=False, print_done_states=False):
        env = self.env
        s, _ = env.reset()
        #dataset = []
        S = []
        A = []
        R = []
        S2 = []
        D = []
        for _ in tqdm(range(horizon), disable=disable_tqdm):
            a = env.action_space.sample()
            s2, r, done, trunc, _ = env.step(a)
            #dataset.append((s,a,r,s2,done,trunc))
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(done)
            if done or trunc:
                s, _ = env.reset()
                if done and print_done_states:
                    print("done!")
            else:
                s = s2
        S = np.array(S)
        A = np.array(A).reshape((-1,1))
        R = np.array(R)
        S2= np.array(S2)
        D = np.array(D)
        return S, A, R, S2, D
    
    def rf_fqi(self, iterations=10, nb_actions=4, gamma=.9, disable_tqdm=False):
        nb_samples = self.S.shape[0]
        Qfunctions = []
        SA = np.append(self.S,self.A,axis=1)
        for iter in tqdm(range(iterations), disable=disable_tqdm):
            if iter==0:
                value=self.R.copy()
            else:
                Q2 = np.zeros((nb_samples,nb_actions))
                for a2 in range(nb_actions):
                    A2 = a2*np.ones((self.S.shape[0],1))
                    S2A2 = np.append(self.S2,A2,axis=1)
                    Q2[:,a2] = Qfunctions[-1].predict(S2A2)
                max_Q2 = np.max(Q2,axis=1)
                value = self.R + gamma*(1-self.D)*max_Q2
            Q = RandomForestRegressor()
            Q.fit(SA,value)
            Qfunctions.append(Q)
        return Qfunctions[-1]
    def greedy_action(self,s,nb_actions=int(4)):
        Qsa = []
        for a in range(nb_actions):
            sa = np.append(s,a).reshape(1, -1)
            Qsa.append(self.Qfunction.predict(sa))
        return np.argmax(Qsa)
     
    def act(self, observation, use_random=False):
        Qfunction = self.Qfunction
        if use_random:
            a = self.env.action_space.sample()

        a = self.greedy_action(observation)

        return a

    def save(self, path):
        joblib.dump(self.Qfunction, path)
        self.path = path

    def load(self):
        if self.path:
            loaded_state = joblib.load(self.path)
            self.Qfunction = loaded_state
        else:
            print(f"File not found at path: {self.path}. Skipping loading.")

