
import numpy as np
from datetime import datetime 
import os
import csv
from project.environments.environment import Environment
import random

class Random_Agent():
    def __init__(self,hyp):
        self.hyp = hyp
        self.env = Environment(hyp=self.hyp)
        print("Environent created")

        base_dir = os.getcwd()
        self.log_dir = os.path.join(base_dir, "logs")
        self.log_path = None
        self.create_log()

        self.frame_i = 0
        self.ep_i = 0
        self.last_ep_rewards = np.zeros(self.hyp["UPDATE_EVERY_N"])
        self.last_ep_lengths = np.zeros(self.hyp["UPDATE_EVERY_N"])
        self.averaging_index = 0

    def train(self):
        # learn until max number of frames reached
        while self.frame_i < self.hyp["EPS_FRAMES_FINAL"]:
            ep_reward=0
            ep_length=0
            terminal = False
            _, _  = self.env.reset()
            action_i = 0
            while not terminal and action_i <= self.hyp["MAX_EP_ACTIONS"]:
                # RANDOM action
                action = random.choice(range(0,18))

                _, reward, terminal, _, _ = self.env.step(action)

                ep_reward += reward
                action_i += 1
                ep_length += 1

                self.frame_i += 1

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
                + f" Av ep length: {av_length},")

            self.write_to_log(self.ep_i, self.frame_i, ep_reward, ep_length)

            self.env.clear_ep_buffer()

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
