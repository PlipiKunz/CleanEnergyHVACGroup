import csv
import random
import gym
import gym_hvac
import time

import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
ENV_NAME = "HVAC-v0"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.99995


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state,mode):
        if np.random.rand() < self.exploration_rate:
            size = self.action_space
            if(mode==0):
                size = 3
            return random.randrange(size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

def hvac(mode, name, limit):
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    outputFileName = 'output/' + name + 'results.csv'
    with open(outputFileName, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(['episode',
                             'step',
                             'time',
                             'air_temperature',
                             'ground_temperature',
                             'hvac_temperature',
                             'basement_temperature',
                             'main_temperature',
                             'attic_temperature',
                             'heat_added',
                             'action',
                             'reward',
                             'total_reward',
                             'terminal'])

    while run <= limit:
        state = env.reset()
        start_time = env.time
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            action = dqn_solver.act(state, mode)
            state_next, reward, terminal, info = env.step(action)
            notWrote = True
            while(notWrote):
                try:
                    with open(outputFileName, 'a', newline='') as outfile:
                        csv_writer = csv.writer(outfile)
                        csv_writer.writerow([run, step, (env.time - start_time).total_seconds()] +
                                            state_next.tolist() +
                                            [env.total_heat_added, int(action), reward, env.total_reward, terminal])
                    notWrote = False
                except(PermissionError):
                    print("trying again soon")
                    time.sleep(1)

            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate)
                      + ", score: " + str(step) + ', total reward: ' + str(env.total_reward))
                break
            dqn_solver.experience_replay()
            step += 1
        run += 1

# # if mode = 0, runs old model, if mode = 1 runs new model
# limit = 250
# mode = 0
# name = "old250"
# hvac(mode,name,limit)
#
# mode = 1
# name = "new250"
# hvac(mode,name,limit)
#
#
# limit = 500
# mode = 0
# name = "old500"
# hvac(mode,name,limit)
# mode = 1
# name = "new500"
# hvac(mode,name,limit)
#
# limit = 1000
# mode = 0
# name = "old1000"
# hvac(mode,name,limit)
# mode = 1
# name = "new1000"
# hvac(mode,name,limit)

# limit = 2000
# mode = 0
# name = "old2000"
# hvac(mode,name,limit)
# mode = 1
# name = "new2000"
# hvac(mode,name,limit)

limit = 15000
# mode = 0
# name = "old5000"
# hvac(mode,name,limit)
mode = 1
name = "new15000"
hvac(mode,name,limit)
