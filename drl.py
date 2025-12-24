import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import matplotlib.pyplot as plt
import time

register(
    id="MyTaxi-v3",
    entry_point="my_taxi:MyTaxiEnv",
    max_episode_steps=200,
)
episodes = 5000      
max_steps = 200         
alpha = 0.1     
gamma = 0.99    
EPS_START = 1.0     
EPS_END = 0.01      

env = gym.make("MyTaxi-v3")
n_actions = env.action_space.n
n_state = env.observation_space.n
Q = np.zeros((n_state, n_actions))

time_step_reward = []

def plot(reward):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.plot(reward, color="royalblue", label="time step rewards")
    ax1.legend()
    
    fig.tight_layout()
    plt.title('Total Reward and Epsilon Decay - Q-Learning')
    plt.grid(axis='x', color='0.80')
    plt.show()

def epsilon_greedy(Q, epsilon, s):
    if np.random.rand() > epsilon:
        action = np.argmax(Q[s, :]).item()
    else:
        action = env.action_space.sample() 

    return action

epsilon_decay_rate = -np.log(EPS_END / EPS_START) / episodes
for episode in range(episodes + 1):
    s, info = env.reset()
    step_reward = 0
    t = 0

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

        step_reward += reward
        if terminated or truncated:
            break

    time_step_reward.append(step_reward)
            
    if episode % 1000 == 0:
        print(f"Episode {episode}: Total Reward = {step_reward}")

print("Training Finished.")
plot(time_step_reward)


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