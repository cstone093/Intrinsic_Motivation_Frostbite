# Intrinsic Motivation in Atari 2600 Frostbite

### Pre-requisites

The DQN training algorithm has been run using Python 3.8. The following packages are required:

- gym[atari,accept-rom-license] 
- tensorflow
- keras
- numpy
- PIL
- imageio
- matplotlib

### Installing Package

The code is stored in a package. So, to install this execute the following command within `\RL_diss_package`:

```bash
pip install .
```

### Training

To train the agent run the following command in terminal:

```bash
python scripts/intrinsic_extrinsic_runner.py [-e <extr_ratio>] [-i <intr_ratio>] [-m {"random","pixels"}]
```
We recommend reward ratios in the range [0,1].

The following folders are used to record training progress:

- `/gif`: gifs of training.
- `/logs`: training episodic reward and episodic length logs. State prediction images if using pixel-based intrinsic motivation.
- `/state_saves`: the model weights

### Evaluation

To plot the graphs for a training csv log:
```bash
python scripts/training_grapher LOG_FILE
```
## Training Videos
### Mixed reward agent using pixels-based forward model
![](https://github.com/cstone093/Intrinsic_Motivation_Frostbite/blob/main/GIF/MIX-PIX.gif)
