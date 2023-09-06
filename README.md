# Attention Learning Method
Reinforcement Learning (RL) has gained significant traction across various domains, such as robotics, telecommunications, gaming, and more, yielding remarkable results. In many applications, the presence of multiple RL agents in an environment necessitates multi-agent reinforcement learning (MARL) approaches. Notably, algorithms like SAC, MADDPG, and others effectively manage interactions and communication among these agents, leading to impressive achievements in solving diverse problems. One such multi-agent reinforcement learning algorithm, which we explore in this research, is the QMIX algorithm, to be detailed further.

In numerous multi-agent environments, we have access to information beyond what individual agents possess, typically including the entire state space of each agent. Leveraging this knowledge not only enables achieving the primary learning objectives but also enhances the quality and performance of the learning process. By timely and judiciously manipulating the environment, we can influence the actions and states of the agents to attain higher learning quality and efficiency. Realizing these manipulations necessitates effectively utilizing information from each agent and the environment.

A pivotal component we can influence in partially observable environments is the observation function of each agent. By aligning the learning of actions and observations over time and correlating them with the rewards received, the algorithm can determine the optimal level of observation required to achieve higher rewards. Thus, it provides us with the means to modify the observations of each agent effectively.

In this project, we demonstrate that incorporating an additional agent with the objective of enhancing learning quality and efficiency is feasible. This added agent, akin to a coach or commander, learns various positions of individual agents and adapts to environmental changes in real-time. We also compare centralized and decentralized methods.

To test our results and methodologies, we employ a popular MARL environment named "Starcraft II," which features a series of mini-maps, highly suited for evaluating multi-agent reinforcement learning algorithms.

## Proposed Method
