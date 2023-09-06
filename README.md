# Attention Learning Method
## Introduction
Reinforcement Learning (RL) has gained significant traction across various domains, such as robotics, telecommunications, gaming, and more, yielding remarkable results. In many applications, the presence of multiple RL agents in an environment necessitates multi-agent reinforcement learning (MARL) approaches. Notably, algorithms like SAC, MADDPG, and others effectively manage interactions and communication among these agents, leading to impressive achievements in solving diverse problems. One such multi-agent reinforcement learning algorithm, which we explore in this research, is the QMIX algorithm, to be detailed further.

In numerous multi-agent environments, we have access to information beyond what individual agents possess, typically including the entire state space of each agent. Leveraging this knowledge not only enables achieving the primary learning objectives but also enhances the quality and performance of the learning process. By timely and judiciously manipulating the environment, we can influence the actions and states of the agents to attain higher learning quality and efficiency. Realizing these manipulations necessitates effectively utilizing information from each agent and the environment.

A pivotal component we can influence in partially observable environments is the observation function of each agent. By aligning the learning of actions and observations over time and correlating them with the rewards received, the algorithm can determine the optimal level of observation required to achieve higher rewards. Thus, it provides us with the means to modify the observations of each agent effectively.

In this project, we demonstrate that incorporating an additional agent with the objective of enhancing learning quality and efficiency is feasible. This added agent, akin to a coach or commander, learns various positions of individual agents and adapts to environmental changes in real-time. We also compare centralized and decentralized methods.

To test our results and methodologies, we employ a popular MARL environment named "Starcraft II," which features a series of mini-maps, highly suited for evaluating multi-agent reinforcement learning algorithms.

## Proposed Method
(note: this is just a very brief summary)

Many multi-agent reinforcement learning methods control the agents in a manner akin to coaching. This means that they train the agents by taking into account the observations of all the agents. One of these algorithms is QMIX, a centralized multi-agent reinforcement learning algorithm. QMIX and other MARL algorithms assume that the size of the agents' observations is fixed. In this project, we introduce a new method for modifying the agents' observations to enhance the speed and efficiency of learning. One influential factor in these observations is the agents' attention. One of the outcomes of attention is the agents' field of view (FOV).

In our proposed method, we introduce an agent named the Captain. This Captain learns during the training process how to adjust the field of view in the environment at the right time to improve the speed and quality of learning. Changing the field of view incurs a cost for the Captain. We define the Captain's reward as follows:

<p align="center">
<img width="289" alt="image" src="https://github.com/bizhii2000/Attention-Learning/assets/109950718/32aee674-b4f2-453e-94f8-a1dc9c8b8804">

In which r represents the shared reward among the agents. The second part of the reward is defined in a way that if the Captain increases the agents' FOV, they receive a negative reward, and if the Captain reduces the agents' FOV, they receive a positive reward. α is also used to balance the two rewards.

The inputs to the Captain's network are the cumulative observations of individual agents concatenated together. The Captain, upon receiving observations from the agents, selects an appropriate field of view for each of the agents. Then, the agents, based on their new observations, find the optimal action using the QMIX algorithm and receive corresponding rewards. Using these rewards, the learning parameters of both the agents (QMIX) and the Captain are updated. This cycle continues until the end of the game.

<p align="center">
<img width="605" alt="image" src="https://github.com/bizhii2000/Attention-Learning/assets/109950718/9e0c4c7e-8276-4a13-9977-31a956574767">
