import csv
import os
import matplotlib.pyplot as plt
import time
import sys
import numpy as np

def read_rewards(csv_dir):
    #  check that there is arg
    f = open(csv_dir)
    results = csv.reader(f, delimiter=",")

    # s = sum(1 for row in results)
    frames = []
    episodes = []
    rewards = []
    frame_count = []

    for row in results:
        frames.append(int(row[0]))
        episodes.append(int(row[1]))
        rewards.append(float(row[2]))
        frame_count.append(int(row[3]))
    return frames, episodes, rewards, frame_count

def running_average(x,y):
    window = 5
    average_y = []

    for ind in range(len(y)):
        if ind<window:
            average_y.append(np.mean(y[:ind+1]))
        else:
            average_y.append(np.mean(y[ind-window:ind+1]))

    return average_y


def plot_results(csv_dir,log_dir):
    fs, eps, rs, fcs = read_rewards(csv_dir)
    # Plot results
    plt.scatter(eps,rs,s=1)
    ra = running_average(eps,rs)
    plt.plot(eps,ra,c="b")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward against Number of Episodes")
    plt.savefig(log_dir + "/ep_rewards" + str(time.time())[:10] + ".PNG")
    print("Generated episode v reward graph")

    plt.cla()

    plt.scatter(eps,fcs,s=1)
    ra = running_average(eps,fcs)
    plt.plot(eps,ra,c="b")
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length against Number of Episodes")
    plt.savefig(log_dir + "/ep_ep_lengths" + str(time.time())[:10] + ".PNG")
    print("Generated episode v episode length graph")

    plt.cla()

    plt.scatter(fs,fcs,s=1)
    ra = running_average(fs,fcs)
    plt.plot(fs,ra,c="b")
    plt.xlabel("Frames")
    plt.ylabel("Episode Length")
    plt.title("Reward against Number of Frames")
    plt.savefig(log_dir + "/frane_ep_lengths" + str(time.time())[:10] + ".PNG")
    print("Generated episode v episode length graph")

    plt.cla()

    plt.scatter(fs,rs,s=1)
    ra = running_average(fs,rs)
    plt.plot(fs,ra,c="b")
    plt.xlabel("Frames")
    plt.ylabel("Rewards")
    plt.title("Reward against Number of Frames")
    plt.savefig(log_dir + "/frame_rewards" + str(time.time())[:10] + ".PNG")
    print("Generated frame v reward graph")


# get pwd for gif store
base_dir = os.getcwd()
log_dir = os.path.join(base_dir, "logs")
csv_dir = os.path.join(base_dir, sys.argv[1])
plot_results(csv_dir,log_dir)