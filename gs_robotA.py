# coding:utf-8

import os
import random
import serial
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Dense,Flatten
import cmath

STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
GAMMA = 0.98  # Discount factor
EXPLORATION_STEPS = 36000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
INITIAL_REPLAY_SIZE = 300  # Number of steps to populate the replay memory before training starts
NUM_REPLAY_MEMORY = 1000000  # Number of replay memory the agent uses for training
BATCH_SIZE = 32  # Mini batch size
TARGET_UPDATE_INTERVAL = 500  # The frequency with which the target network is updated
TRAIN_INTERVAL = 1  # The agent selects 4 actions between successive updates
LEARNING_RATE = 0.0025  # Learning rate used by RMSProp
MOMENTUM = 0.95  # Momentum used by RMSProp
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
TRAIN = True
NUM_ACTIONS=7

T_STEP = 0.1

STOP_ACTION = ( NUM_ACTIONS - 1 ) / 2 # motor_speed = 0

class Agent():
    def __init__(self):
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS

        # Create replay memory
        self.replay_memory = deque()
        
        self.n = 1
        # Create q network
        self.s, self.q_values, q_network = self.build_network()
        q_network_weights = q_network.trainable_weights

        # Create target network
        self.st, self.target_q_values, target_network = self.build_network()
        target_network_weights = target_network.trainable_weights

        # Define target network update operation
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in xrange(len(target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.loss, self.grad_update = self.build_training_op(q_network_weights)

        self.sess = tf.InteractiveSession()

        self.sess.run(tf.initialize_all_variables())

        # Initialize target network
        self.sess.run(self.update_target_network)

    def build_network(self):
        model = Sequential()
        model.add(Flatten(input_shape=(4,4)))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(NUM_ACTIONS))

        s = tf.placeholder(tf.float32, [None, STATE_LENGTH,4])
        q_values = model(s)

        return s, q_values, model

    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, NUM_ACTIONS, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
        grad_update = optimizer.minimize(loss, var_list=q_network_weights)

        return a, y, loss, grad_update

    def get_action(self, state):

        if self.epsilon >= random.random() or self.n < INITIAL_REPLAY_SIZE:
            action = random.randrange(NUM_ACTIONS)
        else:
            action = np.argmax(self.q_values.eval(feed_dict={self.s: [state]}))

        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= self.epsilon_step

        return action

    def run(self, state, action, reward, observation):
        
        next_state = np.append(state[1:], [observation], axis=0)

        # Store transition in replay memory
        self.replay_memory.append((state, action, reward, next_state))
        if len(self.replay_memory) > NUM_REPLAY_MEMORY:
            self.replay_memory.popleft()

        if self.n >= INITIAL_REPLAY_SIZE:
            # Train network
            if self.n % TRAIN_INTERVAL == 0:
                self.train_network()

            # Update target network
            if self.n % TARGET_UPDATE_INTERVAL == 0:
                self.sess.run(self.update_target_network)

        self.n += 1

        return next_state

    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])

        target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: next_state_batch})
        y_batch = reward_batch + GAMMA * np.max(target_q_values_batch, axis=1)

        loss, _ = self.sess.run([self.loss, self.grad_update], feed_dict={
            self.s: state_batch,
            self.a: action_batch,
            self.y: y_batch
        })
      
        print loss 

class gs_robot_physical_env():
    def __init__(self):
        self.ser = serial.Serial('/dev/ttyUSB0', 38400)

        self.T1 = -20.5
        self.S1 = -1853.0
        self.D1 = -303.0
        self.C1 = 1678.0
        self.F1 = 48.5
        self.E1 = 1663.0

        self.thD_old = 0.0
        self.th_old = 0.0

        self.motor_cmd = 0.0
        self.motor_speed = 0.0
        
        self.error1 = 0
        self.error2 = 0
    def compute_theta(self,ax_dat,ay_dat,gz_dat):
        
        ax = ( ax_dat - self.D1 ) / self.C1
        ay = ( ay_dat - self.F1 ) / self.E1

        thD = ( gz_dat - self.T1 ) / self.S1
        
        tmpTh = self.th_old + T_STEP * ( thD + self.thD_old ) / 2.0

#       [theta = a] is equivalent to [theta = a + 2*n*Pi]. I do this way to use accel data but to avoid the problem of plural theta values.

        deltaTh = cmath.phase( complex(ax,ay)/cmath.rect(1.0,tmpTh) )
        
        th = 0.98 * tmpTh + 0.02 * ( tmpTh + deltaTh)

        self.thD_old = thD
        self.th_old = th        

        return th,thD

    def get_motor_command(self,action):
        
        self.motor_speed = float(action - STOP_ACTION)
        self.motor_cmd = self.motor_cmd + 40.0 * self.motor_speed

        if self.motor_cmd > 960.0:
           self.motor_cmd = 960.0
    
        if self.motor_cmd < 0.0:
           self.motor_cmd = 0.0

    def step(self,action,initial=False):

        if not initial:
            self.get_motor_command(action)
  
            self.ser.write(self.iii + ':' + self.timeA + ':' + str(int(self.motor_cmd)) + '\n')
    
        dat1 = self.ser.readline()

        sp1 = dat1.split(',')

        sp2 = [0] * 20
        for num in range(9):
          sp2[num] = int(sp1[num])

        i_delta = sp2[8]

        if i_delta != 1:
           self.error1 += 1

        t_delta = sp2[7]

        if t_delta > 105:
           self.error2 += 1

        ax_dat = sp2[0]
        ay_dat = sp2[1]
        az_dat = sp2[2]

        gx_dat = sp2[3]
        gy_dat = sp2[4]
        gz_dat = sp2[5]

        theta, thD = self.compute_theta(ax_dat,ay_dat,gz_dat)        
 
        observation = [theta, thD, self.motor_cmd, self.motor_speed]

        reward = theta

        self.iii = sp1[6]
        self.timeA = sp1[9] 

        return observation, reward

def initial_prepare(env):
    
    observation0, _ = env.step(STOP_ACTION,initial=True)

    observation1, _ = env.step(STOP_ACTION)

    observation2, _ = env.step(STOP_ACTION)

    observation3, _ = env.step(STOP_ACTION)

    return [observation0, observation1, observation2, observation3] 

def main():
    env = gs_robot_physical_env()
    agent = Agent()

    state = initial_prepare(env)

    for _ in range(EXPLORATION_STEPS):
 
        action = agent.get_action(state)
        observation, reward = env.step(action)
        
        print observation        
       
        state = agent.run(state, action, reward, observation)
        
if __name__ == '__main__':
    main()
