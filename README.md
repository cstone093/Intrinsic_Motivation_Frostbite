# Intrinsic Motivation

### Pre-requisites
The DQN training algorithm has been run using Python 3.8. The following packages are required:

- gym[atari,accept-rom-license] 
- tensorflow
- keras
- numpy
- PIL
- imageio
- matplotlib

### Training
To train the agent run the following command in terminal:
```bash
python scripts/intrinsic_extrinsic_runner.py -e <extrinsic reward ratio> -i <intrinsic reward ratio> -m <state encoding>
```
We recommend reward ratios in the range [0,1].

The following folders are used to record training progress:
- /gif: gifs of training.
- /logs: training episodic reward and episodic length logs. State prediction images if using pixel-based intrinsic motivation.
- /state_saves: the model weights

### Evaluation
To plot the graphs for a training csv log:
```bash
python scripts/training_grapher <csv-log-file>
```
