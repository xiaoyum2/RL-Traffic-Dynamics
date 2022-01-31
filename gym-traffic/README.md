# gym-traffic
This is the self-defied gym env for running stable-baselines3 algorithms. The env is defined in the folder named "envs". Model including traffic dynamics is included in the directery of ```gym-traffic/gym_traffic/envs```

## Case now considering
![This is an image](/simple_case.png)

## Setup
In this project, you will need to have ```gym``` and ```stable-baselines3``` installed. This repo is usable on Linux machine.

## Register env
After defining functions needed for a gym class, register using

```
pip install -e gym-basic
```

Refer to https://towardsdatascience.com/beginners-guide-to-custom-environments-in-openai-s-gym-989371673952 for details.


## Run the RL with custom env
run ```python3 main.py``` in this folder for trainging and validating
