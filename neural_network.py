# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


import random
import numpy as np
from collections import deque

import cv2
import sys

sys.path.append("game/")
import wrapped_flappy_bird as game


def conv2d(x, window, stride):
    return tf.nn.conv2d(x, window, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool2d(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")



def InitiateModel():
    ConvW1 = tf.Variable(tf.random.truncated_normal([8, 8, 4, 32], stddev = 0.01))
    ConvB1 = tf.Variable(tf.constant(0.01, shape = [32]))

    ConvW2 = tf.Variable(tf.random.truncated_normal([4, 4, 32, 64], stddev = 0.01))
    ConvB2 = tf.Variable(tf.constant(0.01, shape = [64]))

    ConvW3 = tf.Variable(tf.random.truncated_normal([3, 3, 64, 64], stddev = 0.01)) 
    ConvB3 = tf.Variable(tf.constant(0.01, shape = [64]))

    FChannelW1 = tf.Variable(tf.random.truncated_normal([1600, 512], stddev = 0.01)) 
    FChannelB1 = tf.Variable(tf.constant(0.01, shape = [512]))

    FChannelW2 = tf.Variable(tf.random.truncated_normal([512, OUTPUT], stddev = 0.01)) 
    FChannelB2 = tf.Variable(tf.constant(0.01, shape = [OUTPUT]))

    sph = tf.compat.v1.placeholder("float", [None, 80, 80, 4])

    ConvH1 = tf.nn.relu(conv2d(sph, ConvW1, 4) + ConvB1)
    PoolH1 = max_pool2d(ConvH1)

    ConvH2 = tf.nn.relu(conv2d(PoolH1, ConvW2, 2) + ConvB2)
    ConvH3 = tf.nn.relu(conv2d(ConvH2, ConvW3, 1) + ConvB3)

    Flatten = tf.reshape(ConvH3, [-1, 1600])

    FChannelH = tf.nn.relu(tf.matmul(Flatten, FChannelW1) + FChannelB1)

    final_read_out = tf.matmul(FChannelH, FChannelW2) + FChannelB2

    return sph, final_read_out, FChannelH


def train(sph, readout, h_fc1, sess):

    epy = tf.placeholder("float", [None])

    act = tf.placeholder("float", [None, OUTPUT])

    out_action = tf.reduce_sum(tf.multiply(readout, act), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(epy - out_action))
    steps = tf.train.AdamOptimizer(1e-6).minimize(cost)

    initial_state = game.GameState()

    D = deque()

    do_nothing = np.zeros(OUTPUT)
    do_nothing[0] = 1
    x_t, r_0, end = initial_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    state_frame = np.stack((x_t, x_t, x_t, x_t), axis=2)

    chkpnt = checkpoint_save(sess)

    epsilon = INITIAL_EPSILON
    timestamp = 0


    while True:
        Readout_all_t = readout.eval(feed_dict={sph : [state_frame]})[0]
        A_t = np.zeros([OUTPUT])
        Agent_action_pos = 0
        if timestamp % FPS_ACTION == 0:
            if random.random() <= epsilon:
                print("*"*30, "Take Action Randomly", "*"*30)
                Agent_action_pos = random.randrange(OUTPUT)
                A_t[random.randrange(OUTPUT)] = 1
            else:
                Agent_action_pos = np.argmax(Readout_all_t)
                A_t[Agent_action_pos] = 1
        else:
            A_t[0] = 1 

        if epsilon > EPSILON and timestamp > EPOCH:
            epsilon -= (INITIAL_EPSILON - EPSILON) / RANDOMACT

        colored_frame_first, reward_frame, end = initial_state.frame_step(A_t)
        frame_first = cv2.cvtColor(cv2.resize(colored_frame_first, (80, 80)), cv2.COLOR_BGR2GRAY)
        _thres, frame_first = cv2.threshold(frame_first, 1, 255, cv2.THRESH_BINARY)
        frame_first = np.reshape(frame_first, (80, 80, 1))
        state_frame_first = np.append(frame_first, state_frame[:, :, :3], axis=2)

        D.append((state_frame, A_t, reward_frame, state_frame_first, end))

        if len(D) > REWATCH:
            D.popleft()

        if timestamp > EPOCH:
            minibatch = random.sample(D, BATCH)

            s_j_batch = [batch[0] for batch in minibatch]
            a_batch = [batch[1] for batch in minibatch]
            r_batch = [batch[2] for batch in minibatch]
            s_j1_batch = [batch[3] for batch in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {sph : s_j1_batch})
            for i in range(0, len(minibatch)):
                end = minibatch[i][4]
                if end:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + DISCOUNT * np.max(readout_j1_batch[i]))

            steps.run(feed_dict = {
                epy : y_batch,
                act : a_batch,
                sph : s_j_batch}
            )

        state_frame = state_frame_first
        timestamp += 1

        if timestamp % 10000 == 0:
            chkpnt.save(sess, 'saved_networks/' + NAME + '-dqn', global_step = t)

        state = ""
        if timestamp <= EPOCH:
            state = "observe"
        elif timestamp > EPOCH and timestamp <= EPOCH + RANDOMACT:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", timestamp, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", "Fly" if Agent_action_pos else "Fall", "/ REWARD", reward_frame, \
            "/ Q_MAX %e" % np.max(Readout_all_t))

def checkpoint_save(sess):
    chkpt = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")

    if checkpoint and checkpoint.model_checkpoint_path:
        chkpt.restore(sess, checkpoint.model_checkpoint_path)
        print("Model Loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Old weights missing")
    
    return chkpt

NAME = 'bird' 
OUTPUT = 2 
EPOCH = 100000.
RANDOMACT = 2000000. 
EPSILON = 0.0001
INITIAL_EPSILON = EPSILON
DISCOUNT = 0.99 
REWATCH = 50000 
BATCH = 32
FPS_ACTION = 1