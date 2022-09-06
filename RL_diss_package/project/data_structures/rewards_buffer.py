import numpy as np

class Reward_Buffer():
    def __init__(self,buffer_size):
        # self._buffer = deque(maxlen=buffer_size)
        self.BUFFER_SIZE = buffer_size

        self._buffer = np.empty(self.BUFFER_SIZE,dtype=np.float32)

        self._n = 0
        self._new_index = 0

    # adds new reward into the buffer
    def add_reward(self,r):
        self._buffer[self._new_index] = r
        self._n = np.max([self._n, self._new_index + 1])
        self._new_index = (self._new_index + 1) % self.BUFFER_SIZE

    # Normalise given reward with running average and standard deviation 
    def normalise(self,r):
        self.add_reward(r)
        mu = np.mean(self._buffer[:self._n])
        sigma = np.std(self._buffer[:self._n])
        if sigma == 0:
            normalised = (r - mu)
        else:
            normalised = (r - mu)/sigma
        return normalised

