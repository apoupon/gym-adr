# gym-adr
A gym implementation of the environment used for "AI-Driven Risk-Aware Scheduling for Active Debris Removal Missions", a paper published at SPAICE 2024, a European Space Agency conference.

## Installation

Create a virtual environment with Python 3.10 and activate it, e.g. with [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html):
```bash
conda create -y -n active_debris_removal python=3.10 && conda activate active_debris_removal
```

Install gym-adr:
```bash
pip install gym-adr
```

To run the example script, install gym-adr with the additional dependencies:
```bash
pip install gym-adr[example]
```

## Quick start

> [!WARNING]
> Rendering the environment requires running the same environment sequentially with different configurations. In the first configuration, the agent uses removal steps to acquire a list of debris to deorbit. In the second configuration, the environment runs with time steps to smoothly render the orbital transfer vehicle's movement between the debris on the deorbited list. This second configuration is time-consuming because the agent is not actively training, many small actions are simulated, and rendering consumes additional computational resources. As a result, frequent rendering during training is not recommended.

## Description

### Description

Active Debris Removal environment.

The goal of the agent is to deorbit as many debris as possible given constraints on fuel
and time. The agent is an Orbital Transfer Vehicle (OTV).

### Action Space

The action space is discrete and consists of n values, n being the number of debris in
orbit around the Earth.

### Observation Space

The observation space is a (5+2n)-dimensional vector representing the state of the agent:
- removal_step: refer to the number of debris deorbited by the OTV
- number_debris_left: refer to the number of debris still in orbits around the Earth
- current_removing_debris: refer to the current target debris
- dv_left: refer to the current amount of fuel available to the OTV
- dt_left: refer to the current amount of time available to the OTV
- binary_flag_debris_1, ..., binary_flag_debris_n: refer to the state of
    debris (0 is in orbit, 1 is already deorbited)
- priority_score_debris_1, ..., priority_score_debris_n: refer to the priority
    score of debris (1 is not prioritary, 10 is prioritary = with a high chance of collision)

### Rewards

The reward is 1 when the OTV deorbit an non-prioritary debris, 10 when it deorbit a prioritary
debris, 0 if it doesn't deorbit any debris (no more fuel/time or debris already deorbited).

### Starting State

The agent starts at the position of a random debris. This debris is considered deorbited for the
rest of the episode.

### Episode Termination

The episode terminates when the OTV run out of fuel (or time) or when it chose as target debris
a debris that has already been deorbited.

### Arguments

* `total_n_debris`: (int) The number of total debris in orbit around the Earth. Default is `10`.
* `dv_max_per_mission`: (int) The total amount of fuel available at the start of the mission.
    Default is `5`.
* `dt_max_per_mission`: (int) The initial duration of the mission. Default is `100`.
* `random_first_debris`: (bool) The debris chosen to initialize the position of the OTV.
    Default is `True`.
* `first_debris`: (int) If `random_first_debris` is set to `False`, the debris chosen to initialize
    the position of the OTV. Default is `None`.

### Version History

* v0: Original version

## Contribute

Contributions are welcome and greatly appreciated! 🚀

### Follow our style

```bash
# install pre-commit hooks
pre-commit install

# apply style and linter checks on staged files
pre-commit run --all-files
```

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
