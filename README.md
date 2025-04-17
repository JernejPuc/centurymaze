# Century Maze: Large-Scale Multi-Robot Navigation

<p align="center">
<img src="data/img_viewer_00.png" width="80%">
</p>

Century Maze is a virtual environment for multi-agent object-goal navigation,
characterised by a large explorable area with many visual embodied agents.
It is designed to challenge cooperative multi-robot systems on a scale of more than 100 agents
by emphasising the role of long-distance communication in the overall system performance.

The task requires each of the 4-wheeled robotic agents in a group
to reach the target specifically assigned to it
(a spherical object distinguished by its unique colour)
in a limited time frame.
As many agents explore the environment simultaneously,
locating and sharing the locations of the targets
should reduce the search times of the other agents
and help them all reach their targets in time.


## Reinforcement Learning

This repository includes a multi-agent reinforcement learning solution (currently under review)
that is able to overcome the difficulties in the scale and complexity of Century Maze:
- The base algorithm is a custom implementation of Proximal Policy Optimization (PPO, MAPPO),
augmented with advantage decomposition (QPLEX, ACPPO).
- Differentiable Inter-Agent Belief Learning (DIABL)
is a self-supervised auxiliary task that facilitates the development of communication
by deterministically associating goal-oriented beliefs with remote observers
through a differentiable communication channel.
- Informative Event Replay (InfER) complements DIABL with a dedicated buffer
to retain experience conducive to communication in the learning process.

A decentralized policy was trained in 9 parallel environments of 112 agents each,
for a total of 1,008 agents in simultaneous simulation.
The result is presented by the following example in two perspectives:

<p align="center">
<!-- <video controls src="data/vid_00.mp4" width="48%"></video> -->
<!-- <video controls src="data/vid_01.mp4" width="48%"></video> -->
<video controls src="https://github.com/user-attachments/assets/6e4d5bce-b434-4f12-ae1c-d535791b575b" width="48%"></video>
<video controls src="https://github.com/user-attachments/assets/6e43a527-964d-4219-bf50-0168da71e4fa" width="48%"></video>
</p>


## Installation

Start by downloading or cloning this repository.

Setup will check for the following packages:
- `numpy`: vectorised processing,
- `numba`: accelerating standard Python,
- `scipy`: rotation, triangulation, and clustering,
- `opencv-python-headless`: saving images and video recording,
- `torch`: main processing and AI integration with CUDA graphs,
- `isaacgym` (stand-alone preview): physics and rendering,
- [`discit`](https://github.com/JernejPuc/discit): accelerated reinforcement learning.

If you already have Python 3.8 on your system,
most of these packages should have their dependencies handled normally during setup,
while Isaac Gym should be downloaded manually beforehand
from the [NVIDIA archive](https://developer.nvidia.com/isaac-gym).
You may also want to follow the [PyTorch instructions](https://pytorch.org/get-started/locally/)
and ensure that the installation correctly targets your CUDA device.

Finally, install Century Maze in editable/development mode from the base directory with:
```
pip install -e .
```


## Running

Besides AI training and evaluation, an interface is provided
to experience and debug the environment with manual controls.
The different modes can be run through the runner with:
```
python runner.py --spec mode --args "additional args for specified mode"
```

See the code of [`runner`](runner.py) and [`session`](src/mazebots/session.py) files
for all argument options.


## Citation

If you use or reference Century Maze, DIABL, or InfER in your work,
please cite the following paper (under review):
```
@article{puc2025centurymaze,
 title={Century Maze: ...},
 author={Puc, Jernej and ...},
 year={2025}}
```
