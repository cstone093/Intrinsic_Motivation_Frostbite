##########################################
# DQN super class
# Purely extrinsic
##########################################

import numpy as np
import os
import csv
from datetime import datetime 

from project.environments.environment import Environment
from project.data_structures.replay_buffer import Replay_Buffer
from project.policies.q_value_function import Q_Value_Function

class DQN():
    def __init__(self,hyp):
        self.hyp = hyp
        # Instantiate environment wrapper class
        self.env = Environment(hyp=self.hyp)
        # Get action options and the state of the processed state
        self.ACTIONS, self.STATE_SHAPE = self.env.get_actions_and_obs_shape()
        # Instantiate the state value q-function
        self.policy = Q_Value_Function(hyp=self.hyp,a_size=self.ACTIONS,s_size=self.STATE_SHAPE)
        print("Policy created")
        # Create empty replay buffer of fixed size
        self.memory = Replay_Buffer(self.hyp["BUFFER_SIZE"],self.hyp["BATCH_SIZE"],self.hyp["STACK_SIZE"])
        print("Memory created")

        self.stacked_state = None
        self.top_state = None

        # Set up logs
        base_dir = os.getcwd()
        self.log_dir = os.path.join(base_dir, "logs")
        self.state_dir = os.path.join(base_dir, "state_saves")
        self.log_path = None
        self.create_log()


        self.frame_i = 0
        self.ep_i = 0
        # Buffers for command line averages
        self.last_ep_rewards = np.zeros(self.hyp["UPDATE_EVERY_N"])
        self.last_ep_lengths = np.zeros(self.hyp["UPDATE_EVERY_N"])
        self.averaging_index = 0

    # Will train the agent's policy for the specified number of frames
    def train(self):
        # learn until max number of frames reached
        while self.frame_i < self.hyp["EPS_FRAMES_FINAL"]:
            # Episode setup
            ep_reward=0
            ep_length=0
            terminal = False
            self.stacked_state, self.top_state  = self.env.reset()
            action_i = 0
            # Start episode
            while not terminal and action_i <= self.hyp["MAX_EP_ACTIONS"]:
                # Action selection
                action = self.policy.choose_action(self.stacked_state,self.frame_i)
                # Environment interaction
                new_stacked_state, reward, terminal, _, new_top_state = self.env.step(action)

                # Log reward and increment count of frames
                ep_reward += reward
                action_i += 1
                ep_length += 1
                self.frame_i += 1

                # Store transition in buffer
                self.memory.add_exp(self.stacked_state,action,reward,new_stacked_state,terminal)
                self.top_state = new_top_state
                self.stacked_state = new_stacked_state

                # Replay from memory
                if  self.frame_i % self.hyp["REPLAY_FREQ"] == 0 and self.frame_i > self.hyp["EPS_FRAMES_INIT"]:
                    # Sample batch from memory
                    states,actions,rewards,new_states,terminals = self.memory.sample()
                    # Improve policy with sampled batch
                    self.policy.learn(states,actions,rewards,new_states,terminals)
                
                # Update target network
                if  self.frame_i % self.hyp["UPDATE_TARGET"] == 0 and self.frame_i > self.hyp["EPS_FRAMES_INIT"]:
                    self.policy.update_target_model()

            self.last_ep_rewards[self.averaging_index] = ep_reward
            self.last_ep_lengths[self.averaging_index] = ep_length
            self.averaging_index = (self.averaging_index + 1) % self.hyp["UPDATE_EVERY_N"]

            # Save gif of most recent episode
            if self.ep_i % self.hyp["RENDER_EVERY_N"] == 0:
                self.env.save_ep_gif(self.ep_i)
                self.env.save_ep_gif_processed(self.ep_i)

            # Print training status update to command line
            if self.ep_i % self.hyp["UPDATE_EVERY_N"] == 0:
                av_reward = np.mean(self.last_ep_rewards)
                av_length = np.mean(self.last_ep_lengths)

                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(f"{current_time} Ep: {self.ep_i},"
                + f" Frames: {self.frame_i},"
                + f" Av reward: {av_reward},"
                + f" Av ep length: {av_length},"
                + f" Epsilon: {self.policy.get_epsilon(self.frame_i)}")

            # Log episode in csv
            self.write_to_log(self.ep_i, self.frame_i, ep_reward, ep_length)

            # Clear episode history ready for next episode
            self.env.clear_ep_buffer()

            # Save state 
            if self.ep_i % self.hyp["SAVE_EVERY_N"] == 0:
                self.ep_i += 1
                self.log_state() 
            else:
                self.ep_i += 1
    
    # creates log for ep length and rewards for graphing
    def create_log(self):
        now = datetime.now()
        current_time = now.strftime("%d-%m-%y-%H:%M:%S")
        log_name = "./log" + "-" + str(current_time) +".csv"
        self.log_path = os.path.join(self.log_dir, log_name) 

    # writes ep length and reward in log
    def write_to_log(self, episode, frames, ep_reward, ep_frame_count):
        row = [frames, episode, ep_reward, ep_frame_count]
        with open(self.log_path, 'a',newline="\n") as out:
            csv_out = csv.writer(out, delimiter =',')
            csv_out.writerow(row)

    def log_state(self):
        # Create state save directory
        now = datetime.now()
        current_time = now.strftime("%d-%m-%y-%H:%M:%S")
        directory = os.path.join(self.state_dir, f"save-ep{self.ep_i}-{current_time}")
        os.mkdir(directory)

        self.policy.save_state(directory)