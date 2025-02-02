"""
HVAC system following the classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
from abc import ABC, abstractmethod
from collections import namedtuple
import csv
import configparser
import datetime
import gym
from gym import spaces, logger
from gym.utils import seeding
import math
import numpy as np
import os
import random
import re


class WeatherGenerator(object):
    def __init__(self, params, weather_filename=os.path.join(os.path.dirname(__file__),
                                                             'resources/weather.csv')):
        # WeatherGenerator.clean_weather_file(params, weather_filename)
        self.weather = self.read_weather(params, weather_filename)
        self.current_idx = None
        self.reset()

    def time(self):
        return self.weather[self.current_idx]['datetime']

    def temperature(self):
        return self.weather[self.current_idx]['temperature']

    def reset(self):
        # Start at random time
        self.current_idx = random.randint(0, len(self.weather) - 1)
        return self.weather[self.current_idx]['datetime'], self.weather[self.current_idx]['temperature']

    def set_idx_from_datetime(self, custom_datetime):
        idx = 0
        while custom_datetime > self.weather[idx]['datetime'] and idx < len(self.weather) - 1:
            idx += 1
        self.current_idx = idx

    def temp_from_datetime(self, custom_datetime):
        idx = 0
        while custom_datetime > self.weather[idx]['datetime'] and idx < len(self.weather) - 1:
            idx += 1

        old = self.current_idx;


        self.current_idx = idx
        temp = self.temperature()
        self.current_idx = old

        return temp

    def step(self, current_time):
        if self.current_idx + 2 >= len(self.weather):
            return self.weather[self.current_idx]['temperature']
        current_idx = self.current_idx
        assert(current_time >= self.weather[current_idx]['datetime'])
        while current_time > self.weather[current_idx + 1]['datetime']:
            current_idx += 1
        self.current_idx = current_idx
        return self.weather[self.current_idx]['temperature']

    @staticmethod
    def read_weather(params, weather_filename):
        weather = list()
        with open(weather_filename) as weather_file:
            for row in csv.DictReader(weather_file, skipinitialspace=True):
                time_str = row[params['datetime_col']]
                temperature_str = row[params['temperature_col']]
                weather.append(
                    {'datetime': datetime.datetime.strptime(time_str, params['datetime_format']),
                     'temperature': float(temperature_str)})
        return weather

    @staticmethod
    def clean_weather_file(params, weather_filename):
        weather = list()
        with open(weather_filename) as weather_file:
            for row in csv.DictReader(weather_file, skipinitialspace=True):
                time_str = row[params['datetime_col']]
                temperature_str = row[params['temperature_col']]
                if time_str and temperature_str:
                    try:
                        float(temperature_str)
                    except ValueError:
                        groups = re.search(r'\D*(-?\d+(?:\.\d+)?)\D*', temperature_str)
                        if groups is None:
                            continue
                        else:
                            temperature_str = groups[1]
                    temperature = float(temperature_str)
                    # Convert to celsius
                    temperature = (temperature - 32) * (5/9)
                    weather.append(
                        {'datetime': datetime.datetime.strptime(time_str, params['datetime_format']),
                         'temperature': temperature})
        with open(os.path.join(os.path.dirname(__file__), 'resources/weather_cleaned.csv'), 'w',
                  newline='') as weather_file:
            weather_csv_writer = csv.DictWriter(weather_file, ['datetime', 'temperature'])
            weather_csv_writer.writeheader()
            for row in weather:
                weather_csv_writer.writerow(row)
        return weather

class HVACEnv(gym.Env):
    """
    Description:
        A home with three rooms: A basement, an attic, and the surrounding air.
        The basement is the lowest room. It touches the earth and the main floor.
        The main floor is the middle room. It touches the basement, the attic, and the surrounding air.
        The attic is upper room. It touches the main floor and the surrounding air.

    Source:
        http://www.sharetechnote.com/html/DE_Modeling_Example_Cooling.html

    Observation:
        Type: Box(5)
        Num	Observation                 Min         Max
        0	Temperature Air             -273        Inf
        1	Temperature Ground          -273        Inf
        2	Temperature HVAC            -273        Inf
        3	Temperature Basement        0           40
        4	Temperature Main Floor      0           40
        5	Temperature Attic           0           40

    "30 is hot, 20 is pleasing, 10 is cold, 0 is freezing"
    20 Celsius (68 F) is roughly room temperature, and 30 and 10 make convenient hot/cold thresholds.

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Turn the cooler on
        1   Turn everything off
        2	Turn the heater on
        3   Fan of main room to basement
        4   Fan of main room to attic
        5   All fans on

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [10..20]

    Episode Termination:
        Temperature Basement is less than 10 or more than 30
        Temperature Main Floor is less than 10 or more than 30
        Temperature Attic is less than 10 or more than 30
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    # k - The cooling constant of the boundary
    # t - Function to get the temperature of the other side of the boundary
    # w - The weight of the boundary. Can be used to give relative size of boundary
    Boundary = namedtuple('Boundary', ['k', 't', 'w'])

    class Room(object):
        def __init__(self, boundary_list=None, hvac=None,fans=[], name=None, volume=None):
            self.name = name
            self.boundary_list = boundary_list
            self.volume = volume
            self.hvac = hvac

        def get_temp_change_eq(self,fans):
            def temp_change_eq(time, tau, current_temp, action):
                return sum([tau.seconds * boundary.k * boundary.w * (boundary.t(time) - current_temp)
                            for boundary in self.boundary_list]) \
                       + (self.hvac(action) if self.hvac is not None else 0) \
                       + sum([fan.tempChange(self.name, action, time,tau) for fan in fans])
            return temp_change_eq

    class Fan(ABC):
        @abstractmethod
        def tempChange(self, name, action, time, tau):
            pass

    class TwoRoomFan(Fan):
        def __init__(self, roomAName, roomATempGetter, roomBName, roomBTempGetter, actionNums, aVol, bVol):
            self.a = roomAName
            self.aTempGet = roomATempGetter
            self.aVol = aVol

            self.b = roomBName
            self.bTempGet = roomBTempGetter
            self.bVol = bVol

            self.actionNums = actionNums

            # the static part of the temperature change equation, the CFM is 1700,
            # the density of the air is 1.20, and the RPM is 300, the .05 is a constant for the equation to be in
            # m^3/min from
            # https://www.cuidevices.com/blog/understanding-airflow-fundamentals-for-proper-dc-fan-selection
            self.cmm = 48 #m^3/min

        def tempChange(self, name, action, time, tau):
            if(name==self.a or name==self.b) and action in self.actionNums:
                overallVol = self.aVol + self.bVol
                timeForCompleteMixSeconds = overallVol/self.cmm * 60

                timeStepRateOfTempChange = min(tau.seconds/timeForCompleteMixSeconds,1)
                endTemp = (((self.aTempGet(time)*self.aVol) + (self.bTempGet(time)*self.bVol)) / (overallVol))


                changeTemp = 0
                if(name==self.a):
                    changeTemp = endTemp - self.aTempGet(time)
                else:
                    changeTemp = endTemp - self.bTempGet(time)

                changeTemp *= timeStepRateOfTempChange
                return changeTemp
            return 0


        # TODO FIND AN ACCEPTABLE VALUE FOR THIS CONSTANT

    def get_hvac(self, action):
        if(0<=action<=2):
            heat_added = (action - 1) * self.hvac_temperature * self.tau.seconds
            self.total_heat_added += heat_added
            return heat_added
        return 0

    def get_ground_temperature(self, time):
        # Very rough estimate, but the ground temperature appears to be about 10 on average
        return self.ground_temperature

    def get_air_temperature(self, time):
        # This could be where weather data could come in.
        # For now just use 0 (or 40)
        return self.weather_generator.step(time)

    def get_air_temp_future(self, timeHours):
        futureTime = self.time + datetime.timedelta(seconds=(timeHours * 3600))
        return self.weather_generator.temp_from_datetime(futureTime)

    def __init__(self):
        cfg = configparser.ConfigParser()
        cfg.read(os.path.join(os.path.dirname(__file__), 'resources/env_config.ini'))

        # initial_state
        self.ground_temperature = float(cfg['initial_state'].get('ground_temperature', '10'))
        # Roughly 1 degree every five minutes
        self.hvac_temperature = float(cfg['initial_state'].get('hvac_temperature', '10'))

        self.weather_generator = WeatherGenerator(cfg['weather_params'],
                                                  os.path.join(os.path.dirname(__file__),
                                                               cfg['weather'].get('weather_file')))

        self.total_heat_added = 0
        self.total_reward = 0
        self.air_temperature = self.weather_generator.temperature()


        def get_temperature_world(time):
            return self.state[0]

        def get_temperature_basement(time):
            return self.state[3]

        def get_temperature_main(time):
            return self.state[4]

        def get_temperature_attic(time):
            return self.state[5]

        k_insulated_boundary = float(cfg['initial_state'].get('insulated_boundary', '0.0000694'))
        k_uninsulated_boundary = float(cfg['initial_state'].get('uninsulated_boundary', '0.0011111'))


        outsideName = "outside"
        mainName = "main"
        atticName = "attic"
        basementName="basement"


        # rooms are 10m by 10m by 10m cubes
        self.basement = HVACEnv.Room(boundary_list=[
            # Basement-Earth Boundary
            # The weight is a half cube where 5 of the 6 sides are below ground)
            HVACEnv.Boundary(k_insulated_boundary, self.get_ground_temperature, (3 / 4)),
            # Basement-Main Boundary
            # The weight is a half cube where 1 of the 6 sides is touching the main level)
            HVACEnv.Boundary(k_uninsulated_boundary, get_temperature_main, (1 / 4))
        ], name=basementName, volume=1000)
        self.main = HVACEnv.Room(boundary_list=[
            # Main-Basement Boundary
            # The weight is a cube where 1 of the 6 sides is touching the main level)
            HVACEnv.Boundary(k_uninsulated_boundary, get_temperature_basement, (1 / 4)),
            # Main-Air Boundary
            # The weight is a cube where 4 of the 6 sides are below ground)
            HVACEnv.Boundary(k_insulated_boundary, self.get_air_temperature, (1 / 2)),
            # Main-Attic Boundary
            # The weight is a cube where 1 of the 6 sides is touching the main level)
            HVACEnv.Boundary(k_uninsulated_boundary, get_temperature_attic, (1 / 4))
        ], hvac=self.get_hvac, name=mainName, volume=1000)
        self.attic = HVACEnv.Room(boundary_list=[
            # Main-Attic Boundary
            # The weight is a cube where 1 of the 6 sides is touching the main level)
            HVACEnv.Boundary(k_uninsulated_boundary, get_temperature_main, (1 / 4)),
            # Attic-Air Boundary
            # The weight is a cube where 5 of the 6 sides are below ground)
            HVACEnv.Boundary(k_insulated_boundary, self.get_air_temperature, (3 / 4))
        ], name=atticName, volume=1000)


        fanBasementToMain = HVACEnv.TwoRoomFan(roomAName=mainName, roomATempGetter=get_temperature_main, aVol= self.main.volume, roomBName=basementName,
                                               roomBTempGetter=get_temperature_basement, bVol=self.basement.volume, actionNums=[3,5])
        fanMainToAttic = HVACEnv.TwoRoomFan(roomAName=mainName, roomATempGetter=get_temperature_main, aVol= self.main.volume, roomBName=atticName,
                                            roomBTempGetter=get_temperature_attic, bVol=self.attic.volume, actionNums=[4,5])
        self.fans = [fanBasementToMain, fanMainToAttic]

        # Thresholds at which to fail the episode
        self.desired_temperature_low = 20
        self.desired_temperature_mean = 21.5
        self.desired_temperature_high = 23
        self.lower_temperature_threshold = 10
        self.upper_temperature_threshold = 33

        '''
        Action space
            Num	Action
            0	Turn the cooler on
            1   No action
            2	Turn the heater on
            3   Fan on from basement to main
            4   Fan on from main to attic
            5   all fans on
            
        '''
        self.action_space = spaces.Discrete(6)

        '''
        Observation Space
            Num	Observation                 Min         Max
            0	Temperature Air             -273        Inf
            1	Temperature Ground          -273        Inf
            2	Temperature HVAC            -273        Inf
            3	Temperature Basement        0           40
            4	Temperature Main Floor      0           40
            5	Temperature Attic           0           40
            6   Temp in 1 hour              -273        Inf
            7   Temp in 2 hours             -273        Inf
            8   Temp in 3 hours             -273        Inf
            
        '''
        low = np.array([
            -273,
            -273,
            -273,
            0,
            0,
            0,
            -273,
            -273,
            -273,

        ])
        high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            40,
            40,
            40,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
        ])


        self.step_count = 0
        # 900 * 4* 24
        self.step_limit = len(self.weather_generator.weather)-2
        self.time = self.weather_generator.time()

        # Tau is the time scale (seconds)
        self.tau = datetime.timedelta(seconds=int(cfg['params'].get('tau', '900')))

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        # Terminate upon reaching failure conditions
        self.termination = True

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Calculate reward using this continuous function
    # y = -0.8165 * sqrt(abs(x - 21.5)) + 1
    # This function was chosen/created because at room temperature (21.5 celsius) it gives a reward of +1,
    # at the thresholds of comfort (roughly 20 to 23 celsius) it returns 0,
    # and around the minimum and maximum threshold (10 and 33 celsius) it returns roughly -1.75, which isn't too extreme
    # In the range 20-23 just use reward 1.
    def calculate_temperature_reward(self, state):
        reward = 0
        for temperature in state[3:]:
            if self.desired_temperature_low <= temperature <= self.desired_temperature_high:
                reward += 1
            elif temperature < self.desired_temperature_low:
                reward += -0.8165 * math.sqrt(abs(temperature - self.desired_temperature_low)) + 1
            # Temperature > desired temperature high
            else:
                reward += -0.8165 * math.sqrt(abs(temperature - self.desired_temperature_high)) + 1
        return reward

    def calculate_action_cost(self, action):
        # HVAC rewards
        if(0<=action<=2):
            return -1 if action != 1 else 0

        # Fan costs
        if(3<=action<=4):
            return -.25
        return -.5

    # The weights 0.75 and 0.25 are arbitrary, but we probably don't want the learner to gain too much from no action
    def calculate_reward(self, state, action):
        return 0.75 * self.calculate_temperature_reward(state) + 0.25 * self.calculate_action_cost(action)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state
        air_temp, ground_temp, hvac_temp, basement_temp, main_temp, attic_temp, onehourt, twohourt, threehourt= state

        # Basement
        basement_temp_change_equation = self.basement.get_temp_change_eq(self.fans)
        new_basement_temp = basement_temp_change_equation(self.time, self.tau, basement_temp, action) + basement_temp

        # Main Room
        main_temp_change_equation = self.main.get_temp_change_eq(self.fans)
        new_main_temp = main_temp_change_equation(self.time, self.tau, main_temp, action) + main_temp

        # Attic
        attic_temp_change_equation = self.attic.get_temp_change_eq(self.fans)
        new_attic_temp = attic_temp_change_equation(self.time, self.tau, attic_temp, action) + attic_temp




        self.state = (self.get_air_temperature(self.time),
                      self.get_ground_temperature(self.time),
                      self.get_hvac(action),
                      new_basement_temp,
                      new_main_temp,
                      new_attic_temp,
                      self.get_air_temp_future(1),
                      self.get_air_temp_future(2),
                      self.get_air_temp_future(3)
                      )



        # print(t)


        # Calculate done - Separated for debugging
        done_basement_lower = new_basement_temp < self.lower_temperature_threshold
        done_basement_upper = new_basement_temp > self.upper_temperature_threshold
        done_main_lower = new_main_temp < self.lower_temperature_threshold
        done_main_upper = new_main_temp > self.upper_temperature_threshold
        done_attic_lower = new_attic_temp < self.lower_temperature_threshold
        done_attic_upper = new_attic_temp > self.upper_temperature_threshold
        done_step_count_limit = self.step_count >= self.step_limit
        if(done_step_count_limit):
            print("read whole weather file")

        done = bool(done_basement_lower or done_basement_upper
                    or done_main_lower or done_main_upper
                    or done_attic_lower or done_attic_upper or done_step_count_limit) \
               and self.termination

        if not done:
            reward = self.calculate_reward(state, action)
        elif self.steps_beyond_done is None:
            # Episode just ended!
            self.steps_beyond_done = 0
            reward = self.calculate_reward(state, action)
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
        self.step_count += 1
        self.time += self.tau
        self.total_reward += reward
        return np.array(self.state), reward, done, {"weather": done_step_count_limit}

    def reset(self):
        self.weather_generator.reset()
        self.time = self.weather_generator.time()
        self.total_heat_added = 0
        self.step_count = 0
        self.total_reward = 0
        self.state = np.concatenate((np.array([self.weather_generator.temperature(),
                                               self.get_ground_temperature(0),
                                               0]),
                                     # Note if you must change the size of the observations, change the size in below array to match total - 3
                                     self.np_random.uniform(low=10, high=30, size=(3,)),

                                     np.array([
                                         self.get_air_temp_future(1),
                                         self.get_air_temp_future(2),
                                         self.get_air_temp_future(3)
                                     ])
                                     ), axis=0)

        self.steps_beyond_done = None
        return np.array(self.state)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
