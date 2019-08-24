

# Hyperparams
class config(object):

    # training
    epochs = 500
    batch_size = 64

    # adam
    learning_rate = 0.00001
    beta_1 = 0.9
    beta_2 = 0.98
    epsilon = 1e-8

    # focal loss
    alpha= 0.25
    gamma= 2

    # embedding
    embedding = 512

    # multi-head attention
    num_heads = 4

    # dropout
    dropout_rate = 0.3

    # early stopping tolerance epochs
    tolerance = 100
