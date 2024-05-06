#!/usr/bin/env python
from __future__ import print_function

import tensorflow.compat.v1 as tf
from neural_network import InitiateModel, train


def start():
    sess=tf.compat.v1.InteractiveSession()
    state, out, _ = InitiateModel()
    train(state, out, _, sess)

def main():
    start()

if __name__ == "__main__":
    main()
