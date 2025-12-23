import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import time

register(
    id="MyTaxi-v3",
    entry_point="my_taxi:MyTaxiEnv",
    max_episode_steps=200,
)
episodes = 3000        
max_steps = 200         
alpha = 0.1     
gamma = 0.99    
EPS_START = 1.0     
EPS_END = 0.01      

env = gym.make("MyTaxi-v3")
n_actions = env.action_space.n
n_state = env.observation_space.n
print(n_state, n_actions)
Q = np.zeros((n_state, n_actions))

def epsilon_greedy(Q, epsilon, s):
    if np.random.rand() > epsilon:
        action = np.argmax(Q[s, :]).item()
    else:
        action = env.action_space.sample() 

    return action

epsilon_decay_rate = -np.log(EPS_END / EPS_START) / episodes
for episode in range(episodes):
    s, info = env.reset()
    total_reward = 0
    t = 0
    epsilon_decay_rate = 0.001
    epsilon = EPS_START * np.exp(-epsilon_decay_rate * episode)
    while t < max_steps:
        t += 1
        a = epsilon_greedy(Q, epsilon, s)
        s_, reward, terminated, truncated, info = env.step(a)
        if terminated and reward < 20: 
            reward = -100 

        if terminated or truncated:
            Q[s][a] = Q[s][a] + alpha * (reward - Q[s][a])
        else:
            Q[s][a] = Q[s][a] + alpha * (reward + gamma * Q[s_].max() - Q[s][a])
        s = s_
        total_reward += reward
       
        
        if terminated or truncated:
            env.reset()
            
    if episode % 200 == 0:
        print(f"Episode {episode}: Total Reward = {total_reward}")

print("Training Finished.")

env = gym.make("MyTaxi-v3", render_mode="human")
for i in range(5):
    s, info = env.reset()
    done = False
    total_test_reward = 0
    
    while not done:
        a = epsilon_greedy(Q, 0, s)
        
        s, reward, terminated, truncated, info = env.step(a)
        total_test_reward += reward
        if terminated or truncated:
            done = True
            time.sleep(1)

env.close()