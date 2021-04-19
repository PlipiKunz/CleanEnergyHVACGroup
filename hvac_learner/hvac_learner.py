import csv
import random
import gym
import gym_hvac
import time

import numpy as np
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import os
ENV_NAME = "HVAC-v0"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 50

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

    # https://www.tensorflow.org/guide/keras/save_and_serialize
    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    def dump(self, name):
        self.model.save(name + ".h5")

    def load(self,name):
        # with open(name) as file:
        #     self.model = model_from_json(file.read())
        self.model = load_model(name)

def hvac(mode, name, limit, resume=False, inputFile="", learn=True, passedInExplorationRate=EXPLORATION_MAX):
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0



    outputCSVFilePath = 'output/' + name + '/csvFiles/'
    if not os.path.exists(outputCSVFilePath):
        os.makedirs(outputCSVFilePath)
    outputCSVFileName = outputCSVFilePath + name + 'results.csv'

    outputJSONFilePath = 'output/' + name + '/LearnedModels/'
    if not os.path.exists(outputJSONFilePath):
        os.makedirs(outputJSONFilePath)
    outputJSONFilePreface = outputJSONFilePath + name

    dqn_solver.exploration_rate = passedInExplorationRate

    if(resume):
            dqn_solver.load(inputFile)
            if not learn:
                dqn_solver.exploration_rate = 0

    with open(outputCSVFileName, 'w', newline='') as outfile:
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
            endOfWeatherFile = info["weather"]

            notWrote = True
            if(terminal or run==0):
                while(notWrote):
                    try:
                        with open(outputCSVFileName, 'a', newline='') as outfile:
                            csv_writer = csv.writer(outfile)
                            csv_writer.writerow([run, step, (env.time - start_time).total_seconds()] +
                                                state_next.tolist() +
                                                [env.total_heat_added, int(action), reward, env.total_reward, terminal])
                        notWrote = False
                    except(PermissionError):
                        print("trying again soon")
                        time.sleep(1)


            reward = reward if (not terminal or endOfWeatherFile) else -reward
            if(endOfWeatherFile):
                print("EOF")

            state_next = np.reshape(state_next, [1, observation_space])

            if(learn):
                dqn_solver.remember(state, action, reward, state_next, terminal)

                if(step%BATCH_SIZE==0):
                    dqn_solver.experience_replay()

            state = state_next
            if terminal:
                dqn_solver.experience_replay()
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate)
                      + ", score: " + str(step) + ', total reward: ' + str(env.total_reward))
                # if(run%5)==0:
                if(learn):
                    print("saving")
                    dqn_solver.dump(outputJSONFilePreface+ "_" + str(run))
                break
            step += 1
        run += 1

# # if mode = 0, runs old model, if mode = 1 runs new model
# for relearning from old models, dont start at max learn rate or it will go forget everything


limit = 250
mode = 0
name = "NormalOld250"
print(name)
hvac(mode, name, limit)
mode = 1
name = "NormalNew250"
print(name)
hvac(mode, name, limit)


limit = 1500
mode = 0
name = "NormalOldLong"
print(name)
hvac(mode, name, limit)
mode = 1500
name = "NormalNewLong"
print(name)
hvac(mode, name, limit)



# limit = 5000
# mode = 1
# name = "LearningModelFromExampleQuick"
# print(name)
# hvac(mode, name, limit, True, "assets/newest1000FromExample_204.h5", passedInExplorationRate=.15)

# limit = 250
# mode = 1
# name = "NonLearningModelFromGoodExample"
# print(name)
# hvac(mode, name, limit, True, "assets/Models/BestModels/LearningModelFromExampleQuick_87.h5", False)
