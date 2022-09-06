import getopt
from project.agents.intrinsic_dqn_agent import Intrinsic_DQN
from project.hyperparameters.frostbite_run import frostbite
import os
import sys

#Defaults

if len(sys.argv) >= 1:
    try:
        options, arguments = getopt.getopt(
            sys.argv[1:],                      # Arguments
            'i:e:m:',                            # Short option definitions
            ["intratio=", "extratio=","intmethod="]) # Long option definitions
    except getopt.GetoptError as err:
        print(err)  # will print something like "option -a not recognized"
        sys.exit(2)
    save_dir = None
    alpha = 0
    beta = 0
    i_type = "pixels"
    for o, a in options:
        if o in ("-e", "--extratio"):
            print(a)
            alpha = float(a)
        elif o in ("-i", "--intratio"):
            print(a)
            beta = float(a)
        elif o in ("-m", "--intmethod"):
            i_type = str(a)
        else:
            assert False, "unhandled option"

print("-------------------------------------------------------------")
print(f"Starting DQN Agent with {alpha*100}% extrinsic reward {beta*100}% intrinsic reward using {i_type} method for generating intrinsic reward")
print("-------------------------------------------------------------")

agent = Intrinsic_DQN(hyp=frostbite,alpha=alpha,beta=beta,i_type=i_type)

agent.train()

