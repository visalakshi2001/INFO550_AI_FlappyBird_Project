#!/usr/bin/env python
from __future__ import print_function

import tensorflow.compat.v1 as tf
from neural_network import InitiateModel, train


def playGame():
    sess=tf.compat.v1.InteractiveSession()
    s, readout, h_fc1 = InitiateModel()
    train(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
