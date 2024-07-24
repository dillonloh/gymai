<h1 align="center">GymAI</h1>

<div align="center">

![STATUS](https://img.shields.io/badge/status-active-brightgreen?style=for-the-badge)

</div>

---
<div align="center">

![exercise](https://github.com/user-attachments/assets/3eb33a6f-f3ae-4c13-ac9f-082eee01671a)

</div>

---
## Table of Contents
* [About](#about)
* [Getting Started](#getting_started)
* [Usage](#usage)
* [Authors](#authors)

## About <a name="about"></a>

Most powerlifting competitions these days, where competitors get a fixed number of attempts to squat/bench/deadlift as heavy as possible, still employ the use of human judges. As the requirements for a valid lift are typically well-defined, the margin of error for judging them tends to be an extremely thin line. For instance, a good squat is defined as one that goes "below parallel (hip crease below top of knee)" ([*1](https://www.tucsonstrength.com/powerlifting-meet-rules/)), which means that even a 1mm error in height perception can have a huge impact on a competition's results. When we compound this with the nature of human biases and errors (think parallax errors), we invite the possibility of contentious results and controversy. This leads to athletes attempting to 'overcompensate' — such as by squatting deeper than they are required to — in order to ensure favourable judging, which deviates them from their main goal: to lift as heavy as bloody possible.

As such, this project is an attempt at providing an unbiased and predictable way of scoring a lift. Similar to the VAR system in football, I expect that the initial direction of this project would be to serve as a 'consultant' for the main human judges, rather than being the only judge, especially when dealing with controversial decisions.

## Getting Started <a name="getting_started"></a>

Documentation for this code is still under construction. However, feel free to try running the `main.py` file for a demonstration of the work's current progress.

### Prerequisites

The following must be installed on the host machine:

- [Python 3.9](https://github.com/pyenv/pyenv) or above

### Installation

To get started with this project, clone the repository to your local machine and install the required dependencies.

```bash
git clone https://github.com/dillonloh/gymai.git
cd gymai
pip install -r requirements.txt
```

## Usage <a name="usage"></a>

To try it out, simply place a video of you performing a squat movement into the videos folder, indicate the path to it in `main.py`, and run.

## Authors <a name="authors"></a>

- [Dillon Loh](https://github.com/dillonloh)
