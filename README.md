# Century Maze: Large-Scale Multi-Robot Navigation

<p align="center">
<img src="data/img_viewer_00.png" width="48%">
<img src="data/img_viewer_01.png" width="48%">
</p>

[Century Maze](https://ieeexplore.ieee.org/document/11456614)
is a virtual environment for multi-agent object-goal navigation,
characterised by a large explorable area with many visual embodied agents.
It is designed to push the scalability and efficiency of cooperative multi-robot systems
on a scale of over 100 robots by only permitting them to achieve
a high level of performance when they communicate efficiently.

The underlying task requires each of the 4-wheeled robotic agents in a group
to explore an unknown environment and reach the target specifically assigned to it
(an object distinguished by its unique colour) in a limited time frame.
As many agents explore the environment simultaneously,
locating and sharing information on discovered objects
should reduce the search times of the agents assigned to them
and help everyone in the group in turn.

We evaluate our methods on three variants of Century Maze:
1. The hand-designed prototype maze without textures or I/O imperfection
emphasises visual clarity and limits the effect of confounding variables.
2. Procedurally generated mazes with realistic textures,
multiple scale configurations, layout randomisation, varying physical
and lighting conditions, noised interactions, and imperfect communication
present robustness challenges closer to practical settings.
3. Procedurally generated mazes with high textural diversity,
based on a larger set of augmented textures, require a more robust visual model
that could function in both prior variants, as well as transfer to real-world conditions.


## Reinforcement Learning

This repository includes
[a multi-agent reinforcement learning solution](https://ieeexplore.ieee.org/document/11456614)
that is able to overcome the difficulties in the scale and complexity of Century Maze:
- The base algorithm is a custom implementation of Proximal Policy Optimization (PPO, MAPPO),
augmented with advantage decomposition (QPLEX, ACPPO).
- Differentiable Inter-Agent Belief Learning (DIABL)
is a supervised auxiliary task that facilitates the development of communication
by deterministically associating goal-oriented beliefs with remote observers
through a differentiable communication channel.
- Informative Event Replay (InfER) complements DIABL with a dedicated buffer
to retain experience conducive to communication in the learning process.

A decentralized policy was trained in 9 parallel environments of 112 agents each,
for a total of 1,008 agents in simultaneous simulation.
The result is presented by the following examples in multiple perspectives:

<p align="center">
<!-- <video controls src="data/vid_00.mp4" width="48%"></video> -->
<!-- <video controls src="data/vid_01.mp4" width="48%"></video> -->
<video controls src="https://github.com/user-attachments/assets/6e4d5bce-b434-4f12-ae1c-d535791b575b" width="48%"></video>
<video controls src="https://github.com/user-attachments/assets/6e43a527-964d-4219-bf50-0168da71e4fa" width="48%"></video>
</p>

<p align="center">
<!-- <video controls src="data/vid_02.mp4" width="48%"></video> -->
<!-- <video controls src="data/vid_03.mp4" width="48%"></video> -->
<video controls src="https://github.com/user-attachments/assets/ae2b903b-5722-43fc-bffc-c194413d7716" width="48%"></video>
<video controls src="https://github.com/user-attachments/assets/d59ba695-5bac-4006-9dd0-190e84fa2379" width="48%"></video>
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

See the code of [`runner`](runner.py) and [`session`](src/mazebotsgen/session.py) files
for all argument options.


## Citation

If you use or reference Century Maze, DIABL, or InfER in your work,
please cite [the following paper](https://ieeexplore.ieee.org/document/11456614):
```
@article{puc2026,
 author={Puc, Jernej and Škulj, Gašper and Pleterski, Jan and Podržaj, Primož and Vrabič, Rok},
 journal={IEEE Transactions on Robotics}, 
 title={Interagent Beliefs for Learning to Communicate in Large-Scale Multirobot Visual Object Search}, 
 year={2026},
 volume={42},
 pages={1729-1748},
 doi={10.1109/TRO.2026.3677944},
 ISSN={1941-0468}}
```
