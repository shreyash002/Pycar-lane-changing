## PyCar DQN Agent
![alt text](https://github.com/shreyash002/Pycar-lane-changing/blob/master/screenshot.jpeg "PyCar Game Environment")

Autonomous lane changing is a critical task in assisting human drivers and to achieve autonomous driving cars. Due to the sequential nature of decision making, the task forms an infinite horizon Markov Decision Process (MDP). Thus naturally lending itself to the framework of Reinforcement Learning. However, the problem becomes challenging owing to high-dimensionality of the state space, complex and dynamic environments, making traditional tabular methods which store each action-state values impractical due to computational constraints. Recent advances in Deep Learning have proved the effectiveness of neural networks as a universal function approximator.

We propose a modified version of the DQN network to estimate action-values given a state to achieve autonomous lane changing.

### Dependencies
```
pygame
pytorch
tqdm
PIL
numpy 
```

### To play the game
```
python pycar_env.py
```

### To run the DQN Agent
```
python DQNAgent.py
```
