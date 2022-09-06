from collections import namedtuple
import numpy as np

Experience = namedtuple('Experience', field_names=['state', 'action', 'ext_reward', 'int_reward', 'done', 'new_state'])

# Replay buffer which stores extrinsic and intrinisic reward separately 
class Int_Replay_Buffer():
  def __init__(self,buffer_size,batch_size,stack_size):
    assert batch_size > 0, "Batch size must be greater than zero"
    assert buffer_size > 0, "Buffer size must be greater than zero"
    assert stack_size > 0, "Stack size must be greater than zero"
    assert batch_size <= buffer_size, "Batch size must be smaller than buffer size"
    assert stack_size <= buffer_size, "Stack size must be smaller than buffer size"
    self.BATCH_SIZE = batch_size
    self.STACK_SIZE = stack_size
    self.BUFFER_SIZE = buffer_size

    self._transitions = np.empty(self.BUFFER_SIZE,dtype=Experience)

    self._exp_count = 0
    self._new_index = 0

  # adds new experiences into the buffer from the format
  # <st,at,r_e,r_i,stprime,term>
  def add_exp(self,s,a,r_e,r_i,sp,d):
    exp = Experience(s,a,r_e,r_i,d,sp)
    self._transitions[self._new_index] = exp

    self._exp_count = np.max([self._exp_count, self._new_index + 1])
    self._new_index = (self._new_index + 1) % self.BUFFER_SIZE

  # samples from the buffer and processes the sample so it is in the desired form for the agent
  def sample(self):
    indices = np.random.choice(self._exp_count, self.BATCH_SIZE, replace=False)

    states, actions, ext_rewards, int_rewards, dones, next_states = zip(*[self._transitions[idx] for idx in indices])
    return np.array(states,dtype=np.float32), np.array(actions,dtype=np.uint8), np.array(ext_rewards, dtype=np.float32), \
            np.array(int_rewards, dtype=np.float32), np.array(next_states), np.array(dones, dtype=np.bool8)