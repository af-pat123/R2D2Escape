

import Environment
import time
import R2D2 
import random


num_episodes = 1000

env = Environment.Environment()
agent = R2D2.R2D2(env)

epsilon = 1 
epsilon_decay = 0.9995
epsilon_min = 0.01
eps_count = 0 
buffer_size = 10000
buffer = []
max_reward = 0

for e in range(num_episodes):

    st = env.reset()
    episode_reward = 0
    t = 0 

    while True:
    
        at = agent.getAction(st, epsilon)
        st1, rt, done, debug = env.step(at)
        episode_reward += rt

        timestep = [st, at, rt, st1, done]
        st = st1

        if len(buffer) > buffer_size:
            del buffer[:1]

        buffer.append(timestep)

        if t % 5 == 0:
            batch_size = min(len(buffer), 32)
            batch = random.sample(buffer, batch_size)
            agent.update(batch)
        
        epsilon = epsilon*epsilon_decay if epsilon < epsilon_min else epsilon_min
        t += 1 

        if done:
            print('Episode {} Finished with Reward {}'.format(eps_count, episode_reward))
            eps_count += 1
            break

    if episode_reward > max_reward:
        max_reward = episode_reward 
        agent.model.save('models/r2d2_dqn_{}.h5'.format(max_reward))
