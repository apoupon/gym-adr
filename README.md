# To do before release
- register to Gym
- run experiments with Stable-Baselines agents and debug with Weights & Biases
- add rendering (render previous ep in the reset function and loop on it until next reset is reached ?)
- write readme
- comment and clean the whole repo (i.e. maybe we don't need all these files in space_physics/)


# gym-adr
A gym implementation of the environment used for "AI-Driven Risk-Aware Scheduling for Active Debris Removal Missions", a paper published at SPAICE 2024, a European Space Agency conference.

## Installation

## Quick start

[!WARNING]
Rendering the environment requires running the same environment sequentially with different configurations. In the first configuration, the agent uses removal steps to acquire a list of debris to deorbit. In the second configuration, the environment runs with time steps to smoothly render the orbital transfer vehicle's movement between the debris on the deorbited list. This second configuration is time-consuming because the agent is not actively training, many small actions are simulated, and rendering consumes additional computational resources. As a result, frequent rendering during training is not recommended.

## Description

## Contribute

## Citations
```bibtex
@article{2024activedebrisremoval,
    title   = {AI-Driven Risk-Aware Scheduling for Active Debris Removal Missions},
    author  = {Antoine Poupon, Hugo de Rohan Willner, Pierre Nikitits and Adam Abdin},
    year    = {2024},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url  = {https://arxiv.org/abs/2409.17012}
}
```
