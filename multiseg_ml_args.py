from math import ceil, log2

import torch


class BreathingAnalysisArgs:
    def __init__(self):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.batch_size = 32
        self.learn_rate = 1e-3
        self.momentum = 0.9
        self.epochs = 100
        self.gamma = 0.001 ** (1 / self.epochs)
        self.winsize = 25  # 5 seconds
        self.stepsize = 10  # 2 second steps
        self.peek_intervals = 2  # 2 = 10 seconds on either side of the current window

        # transformer related
        self.time_enc_size = ceil(
            log2((self.peek_intervals * 2 + 1) * self.winsize) * 2
        )
        self.embedding_size = 2 ** round(log2(self.time_enc_size + 2) + 2)
        self.ff_size = 64

        self.n_head = 8
        self.n_layers = 4

        # conv net related
        # first element 2 is for 2 input channels, x and y velocity
        # could also add in other channels -> x and y height, for img and motion
        # this might make it easier and possible to differentiate between sleeping posn
        self.channels = [2, 6, 10, 16, 20]
        self.padding = self.winsize
        self.conv_bottle = 128

        # bottleneck -> signal
        self.dropout = 0.1

        self.bottleneck = 8
        self.ffn_layers = [self.bottleneck, 64, 128, 64, self.winsize * 2]
