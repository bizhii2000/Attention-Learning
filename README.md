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

## Results
To implement the Captain, we use methods such as QMIX and IQL. In SMAC, the default field of view is set to 9. The actions defined for a Captain are as follows:
- Increase field of view to 12
- Keep field of view constant at 9
- Decrease field of view to 6

Please note that the shoot range for individuals is a constant value of 6.

For the implementation of our method, we place the map in the 5m model. We run the game three times for each algorithm, each time for 5x10^5 steps. We halt training every 25x10^3 steps and run the game 100 times. We calculate the win rate in these 100 games and then continue training. We ensure the exploration-exploitation balance by controlling ε, which changes from 1 to 0.05 after 10^5 steps.

For evaluating our method, we consider two criteria. The first criterion is the win rate, which represents the percentage of victories in the games played. The second criterion examines the actions selected by the Captain. This criterion shows us whether our method was able to achieve a high win rate with an appropriate sight range or not.

<img width="307" alt="image" src="https://github.com/bizhii2000/Attention-Learning/assets/109950718/803b9e88-60ae-49d5-a21f-52d423163b24">

As we can observe, when the map is in the 5m mode, agents achieve a significantly lower win rate when their sight range is set to 6 compared to other configurations. The remaining configurations have approximately similar win rates. Therefore, to investigate each case, we need to examine the sight range at which these win rates were achieved. A lower field of view indicates that the Captain has learned to adjust the field of view at the right time.

<img width="279" alt="image" src="https://github.com/bizhii2000/Attention-Learning/assets/109950718/8df632bc-75d5-46f2-a19a-bf658aaa8dac">

<img width="265" alt="image" src="https://github.com/bizhii2000/Attention-Learning/assets/109950718/4d672224-6ebb-4ce9-9674-8cd46738db57">

<img width="277" alt="image" src="https://github.com/bizhii2000/Attention-Learning/assets/109950718/2aad8028-775f-483b-8610-d832e15eac51">

In the above charts, we can observe the selected sight ranges for each of the agents. As we can see, IQL couldn't achieve the desired results like QMIX and IQL with parameter sharing. In these two cases, the Captain(s) managed to perform well with a sight range very close to 6, which is similar to the performance with a sight range of 12. This is a very positive outcome.

