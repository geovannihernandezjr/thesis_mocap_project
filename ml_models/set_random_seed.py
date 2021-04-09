def set_random_seed(seed = 1):
    import numpy as np
    import matplotlib.pyplot as plt

    import tensorflow.compat.v1 as tf
    import random as rn

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.

    np.random.seed(seed)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.

    rn.seed(seed)

    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see:
    # https://www.tensorflow.org/api_docs/python/tf/set_random_seed

    tf.random.set_random_seed(seed)
