import numpy as np
import csv
import os
import random
from datetime import datetime

from project.agents.dqn_agent import DQN

# Model classes
from project.models.pixels_forward_model import FM_Pixels
from project.models.random_forward_model import FM_Random

# Memory classes
from project.data_structures.rewards_buffer import Reward_Buffer
from project.data_structures.int_replay_buffer import Int_Replay_Buffer

# Policy classes
from project.policies.int_q_value_function import Int_Q_Value_Function
from project.policies.q_value_function import Q_Value_Function

# Environment class
from project.environments.environment import Environment


class Intrinsic_DQN(DQN):
    def __init__(self,hyp,alpha=1,beta=0,i_type="pixels"):
        self.hyp = hyp
        # Instantiate environment wrapper class
        self.env = Environment(hyp=self.hyp)
        # Get action options and the state of the processed state
        self.ACTIONS, self.STATE_SHAPE = self.env.get_actions_and_obs_shape()

        self.stacked_state = None
        self.top_state = None

        # Set up log files
        base_dir = os.getcwd()
        self.log_dir = os.path.join(base_dir, "logs")
        now = datetime.now()
        current_time = now.strftime("%d-%m-%y-%H:%M:%S")
        log_name = "./error" + "-" + str(current_time) +".csv"
        self.pred_log_path = os.path.join(self.log_dir, log_name) 
        self.state_dir = os.path.join(base_dir, "state_saves")
        self.log_path = None
        self.create_log()


        self.frame_i = 0
        self.ep_i = 0

        # Buffers for command line averages
        self.last_ep_rewards = np.zeros(self.hyp["UPDATE_EVERY_N"])
        self.last_ep_lengths = np.zeros(self.hyp["UPDATE_EVERY_N"])
        self.averaging_index = 0
        
        self.alpha = alpha
        self.beta = beta
        self.intrinsic_type = i_type

        assert self.alpha > 0 or self.beta > 0, "Must use either intrinsic or extrinsic reward function"

        # Uses extrinsic reward - purely or mix
        if self.alpha > 0:
            self.extr_buffer = Reward_Buffer(self.hyp["REWARD_BUFFER_SIZE"])
            self.training_mode = "extr"

        # Uses intrinsic reward - purely or mix
        if self.beta > 0:
            self.intr_buffer = Reward_Buffer(self.hyp["REWARD_BUFFER_SIZE"])
            self.training_mode = "intr"
            if self.intrinsic_type == "pixels":
                self.forward_model = FM_Pixels(hyp,self.ACTIONS,self.STATE_SHAPE)
            elif self.intrinsic_type == "random":
                self.forward_model = FM_Random(hyp,self.ACTIONS,self.STATE_SHAPE,feature_count=512) 
            else: raise(NotImplementedError)

        # Mix of extrinsic and intrinsic
        if self.alpha > 0 and self.beta > 0:
            self.policy = Int_Q_Value_Function(hyp=self.hyp,a_size=self.ACTIONS,s_size=self.STATE_SHAPE)
            self.training_mode = "mix"
        else: 
            self.policy = Q_Value_Function(hyp=self.hyp,a_size=self.ACTIONS,s_size=self.STATE_SHAPE)

        self.memory = Int_Replay_Buffer(self.hyp["BUFFER_SIZE"],self.hyp["BATCH_SIZE"],self.hyp["STACK_SIZE"])

        print(f"Set up agent. Training mode is {self.training_mode} with {self.intrinsic_type} if using intrinsic motivation")
        print(f"Alpha is {self.alpha} and Beta is {self.beta}")

    def train(self):
        # Step through environment to gather mean and variance of rewards for normalisation
        print("FILLING BUFFER")
        for _ in range (self.hyp["REWARD_BUFFER_SIZE"]//100):
            # Reset environment
            buff_stacked_state, _  = self.env.reset()
            # Generate a random action
            action = np.random.randint(self.ACTIONS)
            # Step environment with random action
            buff_new_stacked_state, extr_reward, terminal, _, _ = self.env.step(action)
            intr_reward = 0
            if self.training_mode != "extr":
                if self.intrinsic_type == "pixels" or self.intrinsic_type == "random":
                    # Generate the (un-normalised) intrinsic reward 
                    intr_reward = self.forward_model.get_error(buff_stacked_state, action, buff_new_stacked_state, training=False)
                else: raise(NotImplementedError)
                self.intr_buffer.add_reward(intr_reward)
            if self.training_mode != "intr":
                self.extr_buffer.add_reward(extr_reward)

            buff_stacked_state = buff_new_stacked_state

        print("Beginning Training")
        # Train until frame goal met
        while self.frame_i < self.hyp["EPS_FRAMES_FINAL"]:
            ep_reward=0
            ep_length=0
            terminal = False
            # Reset the environment
            self.stacked_state, self.top_state  = self.env.reset()
            self.env.clear_ep_buffer()

            # For each episode
            while not terminal and ep_length <= self.hyp["MAX_EP_ACTIONS"]:
                if self.training_mode == "mix":
                    action = self.policy.choose_action(self.stacked_state,self.frame_i,self.alpha,self.beta)
                else:
                    action = self.policy.choose_action(self.stacked_state,self.frame_i)
                
                # Step through environment with chosen action
                new_stacked_state, extr_reward, terminal, _, new_top_state = self.env.step(action)
                ep_reward += extr_reward

                intr_reward = 0

                if  self.training_mode != "extr":
                    # Calculate prediction error
                    if self.intrinsic_type == "pixels" or self.intrinsic_type == "random":
                        error = self.forward_model.get_error(self.stacked_state,action,new_stacked_state)
                    else: raise(NotImplementedError)
                    # Store intrinsic reward in buffer
                    self.intr_buffer.add_reward(error)
                    # Normalise intrinsic reward
                    intr_reward = self.intr_buffer.normalise(error)
                    if random.uniform(0,1) < 0.002:
                        self.write_error(self.frame_i,intr_reward,error)
                if self.training_mode != "intr":
                    self.extr_buffer.add_reward(extr_reward)
                    extr_reward = self.extr_buffer.normalise(extr_reward)

                # Store transition in replay buffer
                self.memory.add_exp(self.stacked_state,action,extr_reward,intr_reward,new_stacked_state,terminal)
                
                self.top_state = new_top_state
                self.stacked_state = new_stacked_state

                self.frame_i += 1
                ep_length += 1
                
                if  self.frame_i % self.hyp["REPLAY_FREQ"] == 0 and self.frame_i > self.hyp["EPS_FRAMES_INIT"]:
                    # Replay from memory
                    states,actions,ext_rewards,int_rewards,new_states,terminals = self.memory.sample()
                    # Train policy (and model if intrinsic motivation used)
                    if self.training_mode == "mix": 
                        self.policy.learn(states,actions,ext_rewards,int_rewards,new_states,terminals)
                        self.forward_model.learn(states,actions,new_states)
                    elif self.training_mode == "extr":
                        self.policy.learn(states,actions,ext_rewards,new_states,terminals)
                    elif self.training_mode == "intr": 
                        self.policy.learn(states,actions,int_rewards,new_states,terminals)
                        self.forward_model.learn(states,actions,new_states)
                    else:
                        raise(NotImplementedError)
                
                # update target
                if  self.frame_i % self.hyp["UPDATE_TARGET"] == 0 and self.frame_i > self.hyp["EPS_FRAMES_INIT"]:
                    self.policy.update_target_model()

            self.last_ep_rewards[self.averaging_index] = ep_reward
            self.last_ep_lengths[self.averaging_index] = ep_length

            self.averaging_index = (self.averaging_index + 1) % self.hyp["UPDATE_EVERY_N"]

            if self.ep_i % self.hyp["RENDER_EVERY_N"] == 0:
                self.env.save_ep_gif(self.ep_i)
                self.env.save_ep_gif_processed(self.ep_i)

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

            self.write_to_log(self.ep_i, self.frame_i, ep_reward, ep_length)


            if self.ep_i % self.hyp["SAVE_EVERY_N"] == 0:
                self.ep_i += 1
                self.log_state() 
            else:
                self.ep_i += 1
        print("SAVING FINAL GIFs")        
        self.env.save_ep_gif(self.ep_i-1)
        self.env.save_ep_gif_processed(self.ep_i-1)

        # writes ep length and reward in log
    def write_error(self, frames, reward, prediction_error):
        row = [frames, reward, prediction_error]
        with open(self.pred_log_path, 'a',newline="\n") as out:
            csv_out = csv.writer(out, delimiter =',')
            csv_out.writerow(row)