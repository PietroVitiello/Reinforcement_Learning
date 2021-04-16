import numpy as np
import torch
import collections

class ReplayBuffer():

    def __init__(self, size):
        self.buffer = collections.deque(maxlen=size)
        self.buffer.clear()

    def add(self, touple):
        self.buffer.append(touple)

    def sample(self, N):
        assert len(self.buffer) >= N

        indeces = np.random.randint(N, size=N)
        batch = [self.buffer[i] for i in indeces]
        # print(batch[1])

        return batch

    def get_len(self):
        return len(self.buffer)