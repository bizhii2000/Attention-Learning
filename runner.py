from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from replay_buffer import ReplayBuffer
from qmix_smac import QMIX_SMAC
from normalization import Normalization
import sys
import matplotlib.pyplot as plt
import continuous_ppo
import datetime
import random
from os import replace
from absl import logging
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
logging.set_verbosity(logging.DEBUG)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

class Runner_SMAC_QMIX:
    def __init__(self, args, number, seed):
        self.args = args
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.distribution_config = {
                        "n_units": self.args.N_agents,
                        "team_gen": {
                            "dist_type": "weighted_teams",
                            "unit_types": ["marine", "marauder", "medivac"],
                            "exception_unit_types": ["medivac"],
                            "weights": [1, 0, 0],
                            "observe": True,
                        },
                        "start_positions": {
                            "dist_type": "surrounded_and_reflect",
                            "p": 0.5,
                            "map_x": 32,
                            "map_y": 32,
                        },
                        "n_enemies": self.args.N_agents,
                    }
        self.env = StarCraftCapabilityEnvWrapper(
                    capability_config=self.distribution_config,
                    map_name="10gen_terran",
                    debug=False,
                    conic_fov=False,
                    obs_own_pos=True,
                    use_unit_ranges=False,
                    min_attack_range=2,
                    seed = self.seed)
        
        self.env_info = self.env.get_env_info()
        self.args.N = self.env_info["n_agents"]  # The number of agents
        self.args.obs_dim = self.env_info["obs_shape"]  # The dimensions of an agent's observation space
        self.args.state_dim = self.env_info["state_shape"]  # The dimensions of global state space
        self.args.action_dim = self.env_info["n_actions"]  # The dimensions of an agent's action space
        self.args.episode_limit = self.env_info["episode_limit"]  # Maximum number of steps per episode
        print("number of agents={}".format(self.args.N))
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        # Create N agents
        self.agent_n = QMIX_SMAC(self.args)
        self.replay_buffer = ReplayBuffer(self.args)
        self.initial_visions = np.ones(self.args.N)*self.args.sight_range
        self.env.change_unit_capability_range(self.initial_visions)
        # Create a tensorboard
        
        
        self.epsilon = self.args.epsilon  # Initialize the epsilon
        self.win_rates = []  # Record the win rates
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=1)

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, _, episode_steps = self.run_episode_smac(evaluate=False)  # Run an episode
            self.total_steps += episode_steps

            if self.replay_buffer.current_size >= self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Training

        self.evaluate_policy()
        self.env.close()
        # Save the win rates
        current_time = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.datetime.now())
        np.save('./data_train/{}_env_{}m_sight_{}_seed_{}_{}.npy'.format(self.args.algorithm, self.args.N, self.args.sight_range,
                                                                    self.seed, current_time), np.array(self.win_rates))

    def evaluate_policy(self, ):
        win_times = 0
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            win_tag, episode_reward, _ = self.run_episode_smac(evaluate=True)
            if win_tag:
                win_times += 1
            evaluate_reward += episode_reward

        win_rate = win_times / self.args.evaluate_times
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.win_rates.append(win_rate)
        print("total_steps:{} \t win_rate:{} \t evaluate_reward:{}".format(self.total_steps, win_rate, evaluate_reward))
        

    def run_episode_smac(self, evaluate=False):
        win_tag = False
        episode_reward = 0
        self.env.reset()
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            self.agent_n.eval_Q_net.rnn_hidden = None
        last_onehot_a_n = np.zeros((self.args.N, self.args.action_dim))  # Last actions of N agents(one-hot)
        for episode_step in range(self.args.episode_limit):
            obs_n = self.env.get_obs()  # obs_n.shape=(N,obs_dim)
            s = self.env.get_state()  # s.shape=(state_dim,)
            avail_a_n = self.env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)
            epsilon = 0 if evaluate else self.epsilon
            a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, avail_a_n, epsilon)
            last_onehot_a_n = np.eye(self.args.action_dim)[a_n]  # Convert actions to one-hot vectors
            r, done, info = self.env.step(a_n)  # Take a step
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            episode_reward += r

            if not evaluate:
                if self.args.use_reward_norm:
                    r = self.reward_norm(r)
                """"
                    When dead or win or reaching the episode_limit, done will be Ture, we need to distinguish them;
                    dw means dead or win,there is no next state s';
                    but when reaching the max_episode_steps,there is a next state s' actually.
                """
                if done and episode_step + 1 != self.args.episode_limit:
                    dw = True
                else:
                    dw = False

                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, r, dw)
                # Decay the epsilon
                self.epsilon = self.epsilon - self.args.epsilon_decay if self.epsilon - self.args.epsilon_decay > self.args.epsilon_min else self.args.epsilon_min

            if done:
                break

        if not evaluate:
            # An episode is over, store obs_n, s and avail_a_n in the last step
            obs_n = self.env.get_obs()
            s = self.env.get_state()
            avail_a_n = self.env.get_avail_actions()
            self.replay_buffer.store_last_step(episode_step + 1, obs_n, s, avail_a_n)

        return win_tag, episode_reward, episode_step + 1


class Runner_SMAC_QMIX_IQL:
    def __init__(self, args, number, seed):
        self.args = args
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.distribution_config = {
                        "n_units": self.args.N_agents,
                        "team_gen": {
                            "dist_type": "weighted_teams",
                            "unit_types": ["marine", "marauder", "medivac"],
                            "exception_unit_types": ["medivac"],
                            "weights": [1, 0, 0],
                            "observe": True,
                        },
                        "start_positions": {
                            "dist_type": "surrounded_and_reflect",
                            "p": 0.5,
                            "map_x": 32,
                            "map_y": 32,
                        },
                        "n_enemies": self.args.N_agents,
                    }
        self.env = StarCraftCapabilityEnvWrapper(
                    capability_config=self.distribution_config,
                    map_name="10gen_terran",
                    debug=False,
                    conic_fov=False,
                    obs_own_pos=True,
                    use_unit_ranges=False,
                    min_attack_range=2,
                    seed = self.seed)
        self.env_info = self.env.get_env_info()
        self.args.N = self.env_info["n_agents"]  # The number of agents
        self.args.obs_dim = self.env_info["obs_shape"]  # The dimensions of an agent's observation space
        self.args.state_dim = self.env_info["state_shape"]  # The dimensions of global state space
        self.captain_action_dim = 3  # The dimensions of an captain's action space
        self.args.action_dim = self.env_info["n_actions"]  # The dimensions of an agent's action space
        self.args.episode_limit = self.env_info["episode_limit"]  # Maximum number of steps per episode
        print("number of agents={}".format(self.args.N))
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        # Create N agents
        self.agent_n = QMIX_SMAC(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        self.list_of_agents_sight_networks = []

        # self.Captain = captain.Agent(100, state_size=self.args.obs_dim, action_size=3 ,eps_start=1.0,
        #                               eps_end=0.05, eps_decay=0.998, BUFFER_SIZE = int(1e5),
        #                                 BATCH_SIZE = 64, GAMMA = 0.99, TAU = 1e-3, LR = 5e-4, UPDATE_EVERY = 4)

        for i in range(self.args.N):
            self.list_of_agents_sight_networks.append(captain.Agent(i, state_size=self.args.obs_dim, action_size=3 ,eps_start=1.0,
                                      eps_end=0.05, eps_decay=0.998, BUFFER_SIZE = int(1e5),
                                        BATCH_SIZE = 64, GAMMA = 0.99, TAU = 1e-3, LR = 5e-4, UPDATE_EVERY = 4))

        # Create a tensorboard
        
        
        self.epsilon = self.args.epsilon  # Initialize the epsilon
        self.kill_sight_reward_ratio = self.args.kill_sight_reward_ratio
        self.win_rates = []  # Record the win rates
        self.total_steps = 0
        self.captain_action_history = []
        self.reward_list = []
        self.episode_reward_list = []
        self.captain_reward_list = []
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=1)

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, _, episode_steps = self.run_episode_smac(evaluate=False)  # Run an episode


            self.total_steps += episode_steps

            if self.replay_buffer.current_size >= self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Training

        self.evaluate_policy()
        self.env.close()
        
        self.captain_action_history = np.array(self.captain_action_history, dtype = np.float16)
        self.reward_list = np.array(self.reward_list)
        self.episode_reward_list = np.array(self.episode_reward_list)
        self.captain_reward_list = np.array(self.captain_reward_list)
        # Save the win rates
        current_time = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.datetime.now())
        np.save('./data_train/{}_env_{}m_captain_IQL_seed_{}_{}.npy'.format(self.args.algorithm, self.args.N,
                                                                    self.seed, current_time), np.array(self.win_rates))
        np.save('./data_train/{}_env_{}m_captain_IQL_seed_{}_{}captain_action_history.npy'.format(self.args.algorithm, self.args.N,
                                                                    self.seed, current_time), np.array(self.captain_action_history))

    def evaluate_policy(self, ):
        win_times = 0
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            win_tag, episode_reward, _ = self.run_episode_smac(evaluate=True)
            if win_tag:
                win_times += 1
            evaluate_reward += episode_reward

        win_rate = win_times / self.args.evaluate_times
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.win_rates.append(win_rate)
        print("total_steps:{} \t win_rate:{} \t evaluate_reward:{}".format(self.total_steps, win_rate, evaluate_reward))
        

    def run_episode_smac(self, evaluate=False):
        win_tag = False
        episode_reward = 0
        self.env.reset()
        self.shadow_healths = np.zeros(self.args.N)
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            self.agent_n.eval_Q_net.rnn_hidden = None
        last_onehot_a_n = np.zeros((self.args.N, self.args.action_dim))  # Last actions of N agents(one-hot)
        i = 0
        for episode_step in range(self.args.episode_limit):
            i += 1
            values = []
            obs_n = self.env.get_obs()  # obs_n.shape=(N,obs_dim)
            for n in range(self.args.N):
                values.append(self.list_of_agents_sight_networks[n].act(obs_n[n]))
            
            self.share_parameters()

            self.captain_action_history.append(values)
            self.env.change_unit_capability_range(values)       # Captian changes sights and now we have new obs
            obs_n = self.env.get_obs()  # obs_n.shape=(N,obs_dim)
            s = self.env.get_state()  # s.shape=(state_dim,)
            avail_a_n = self.env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)
            epsilon = 0 if evaluate else self.epsilon
            a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, avail_a_n, epsilon)
            last_onehot_a_n = np.eye(self.args.action_dim)[a_n]  # Convert actions to one-hot vectors
            r, done, info = self.env.step(a_n)  # Take a step
            self.reward_list.append(r)
            ########################### captain ###########################
            # self.reduce_health_by_sight()
            for j in range(self.args.N): #set dead agents sight to almost 9
                if  obs_n[j][0] == 0.0:
                    values[j] = 9.0625
            self.captain_action_history.append(values) 
            new_obs_n = self.env.get_obs()                   
            captain_reward = r + (9**2*self.args.N-(self.env.visions**2).sum())/self.kill_sight_reward_ratio/self.args.N  # captain reward 
            self.captain_reward_list.append(captain_reward)
            
            ########################### captain ###########################
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            episode_reward += r

            if not evaluate:
                if self.args.use_reward_norm:
                    r = self.reward_norm(r)
                """"
                    When dead or win or reaching the episode_limit, done will be Ture, we need to distinguish them;
                    dw means dead or win,there is no next state s';
                    but when reaching the max_episode_steps,there is a next state s' actually.
                """
                if done and episode_step + 1 != self.args.episode_limit:
                    dw = True
                else:
                    dw = False

                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, r, dw)
                # Decay the epsilon
                self.epsilon = self.epsilon - self.args.epsilon_decay if self.epsilon - self.args.epsilon_decay > self.args.epsilon_min else self.args.epsilon_min
                values = [int((value-9))/3+1 for value in values]
                for n1 in range(self.args.N):
                    self.list_of_agents_sight_networks[n1].step(obs_n[n1], values[n1], captain_reward, new_obs_n[n1], done)       # Captain experience replay

            if done:
                break
        for n2 in range(self.args.N):
            self.list_of_agents_sight_networks[n2].decay_epsilon()
        if not evaluate:
            # An episode is over, store obs_n, s and avail_a_n in the last step
            obs_n = self.env.get_obs()
            s = self.env.get_state()
            avail_a_n = self.env.get_avail_actions()
            self.replay_buffer.store_last_step(episode_step + 1, obs_n, s, avail_a_n)
        self.episode_reward_list.append(episode_reward)
        return win_tag, episode_reward, episode_step + 1
    

    def calculate_mean_parameters(self):
        mean = 0
        for i in range(self.args.N):
            mean += 1/(self.args.N)*self.list_of_agents_sight_networks[i].get_weights()
        return mean
    
    def share_parameters(self):
        for i in range(self.args.N):
            self.list_of_agents_sight_networks[i].change_weights(self.calculate_mean_parameters())

    def remove_dead_from_list(self, start_from_last_indexes = None):
        if start_from_last_indexes is None:
            captain_action_history = []
            captain_list_len = len(self.captain_action_history)
            for i in range(self.args.N):
                agent_i = []
                for j in range(captain_list_len):
                    if self.captain_action_history[j][i] != 9.0625:
                        agent_i.append(self.captain_action_history[j][i])
                captain_action_history.append(agent_i.copy())
        else:       # when you run an episode and not a long run
            captain_action_history = []
            for i in range(self.args.N):
                agent_i = []
                for j in range(start_from_last_indexes):
                    if self.captain_action_history[j-start_from_last_indexes][i] != 9.0625:
                        agent_i.append(self.captain_action_history[j][i])
                captain_action_history.append(agent_i.copy())
        return captain_action_history
        
    def save_model(self):
         torch.save(self.agent_n.eval_Q_net.state_dict(), "./model/model_{}_{}m_total_steps_{}_ratio_{}.pth".format( 
                                    self.args.algorithm, self.args.N, self.args.max_train_steps, self.args.kill_sight_reward_ratio))
         
         torch.save(self.captain.eval_Q_net.state_dict(), "./model/captain_model_{}_{}m_total_steps_{}_ratio_{}.pth".format( 
                                    self.args.algorithm, self.args.N, self.args.max_train_steps, self.args.kill_sight_reward_ratio))
    
    def load_model(self):
        self.agent_n.eval_Q_net.load_state_dict(torch.load("./model/model_{}_{}m_total_steps_{}_ratio_{}.pth".format( 
                                    self.args.algorithm, self.args.N, self.args.max_train_steps, self.args.kill_sight_reward_ratio)))
        
        self.captain.eval_Q_net.load_state_dict(torch.load("./model/captain_model_{}_{}m_total_steps_{}_ratio_{}.pth".format(
                                    self.args.algorithm, self.args.N, self.args.max_train_steps, self.args.kill_sight_reward_ratio)))
    
    def plot(self, average = 5000):
        plt.plot(self.win_rates)
        plt.title('Win Rate')
        plt.figure()
        plt.plot(moving_average(self.reward_list, average))
        plt.title('reward list')
        plt.figure()
        plt.plot(moving_average(self.episode_reward_list, average//7+1))
        plt.title('episode_reward listward list')
        plt.figure()
        plt.plot(moving_average(self.captain_reward_list, average))
        plt.title('captain reward list')
        plt.figure()
        captain_action_history = self.remove_dead_from_list()
        for i in range(self.args.N):
            plt.plot(moving_average(captain_action_history[i], average), label = 'agent' + str(i));
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left');
        plt.title('captain action list')


class Runner_SMAC_QMIX_QMIX:
    def __init__(self, args, number, seed):
        self.args = args
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.distribution_config = {
                        "n_units": self.args.N_agents,
                        "team_gen": {
                            "dist_type": "weighted_teams",
                            "unit_types": ["marine", "marauder", "medivac"],
                            "exception_unit_types": ["medivac"],
                            "weights": [1, 0, 0],
                            "observe": True,
                        },
                        "start_positions": {
                            "dist_type": "surrounded_and_reflect",
                            "p": 0.5,
                            "map_x": 32,
                            "map_y": 32,
                        },
                        "n_enemies": self.args.N_agents,
                    }
        self.env = StarCraftCapabilityEnvWrapper(
                    capability_config=self.distribution_config,
                    map_name="10gen_terran",
                    debug=False,
                    conic_fov=False,
                    obs_own_pos=True,
                    use_unit_ranges=False,
                    min_attack_range=2,
                    seed = self.seed)
        self.env_info = self.env.get_env_info()
        self.args.N = self.env_info["n_agents"]  # The number of agents
        self.args.obs_dim = self.env_info["obs_shape"]  # The dimensions of an agent's observation space
        self.args.state_dim = self.env_info["state_shape"]  # The dimensions of global state space
        self.captain_action_dim = 3  # The dimensions of an captain's action space
        self.args.action_dim = self.env_info["n_actions"]  # The dimensions of an agent's action space
        self.args.episode_limit = self.env_info["episode_limit"]  # Maximum number of steps per episode
        print("number of agents={}".format(self.args.N))
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        # Create N agents
        self.agent_n = QMIX_SMAC(self.args)
        self.replay_buffer = ReplayBuffer(self.args)
        # Create Captain
        self.captain = QMIX_SMAC(self.args, captain= True)
        self.captain.action_dim = self.captain_action_dim
        self.captain.action_dim = self.captain_action_dim
        self.captain_replay_buffer = ReplayBuffer(self.args, captain= True)
        self.captain_avail_a_n = np.ones((self.args.N, self.captain_action_dim))
        self.shadow_healths = np.zeros(self.args.N)
        if self.args.load_from_file:
            self.load_model()

        # Create a tensorboard
        
        
        self.epsilon = self.args.epsilon  # Initialize the epsilon
        self.kill_sight_reward_ratio = self.args.kill_sight_reward_ratio
        self.win_rates = []  # Record the win rates
        self.total_steps = 0
        self.captain_action_history = []
        self.reward_list = []
        self.episode_reward_list = []
        self.captain_reward_list = []
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=1)

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, _, episode_steps = self.run_episode_smac(evaluate=False)  # Run an episode


            self.total_steps += episode_steps

            if self.replay_buffer.current_size >= self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Training
                self.captain.train(self.captain_replay_buffer, self.total_steps)

        self.evaluate_policy()
        
        ##### ostad version #####
        # values = np.ones(self.args.N, dtype=float)*9
        # self.env.change_unit_capability_range(values)
        # self.evaluate_policy(no_captain=True)
        ##### ostad version #####
        
        self.env.close()
        
        self.captain_action_history_np = np.array(self.captain_action_history, dtype = np.float16)
        self.reward_list_np = np.array(self.reward_list)
        self.episode_reward_list_np = np.array(self.episode_reward_list)
        self.captain_reward_list_np = np.array(self.captain_reward_list)
        # Save the win rates
        current_time = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.datetime.now())
        np.save('./data_train/{}_env_{}m_captain_QMIX_seed_{}_{}.npy'.format(self.args.algorithm, self.args.N,
                                                                    self.seed, current_time), np.array(self.win_rates))
        np.save('./data_train/{}_env_{}m_captain_QMIX_seed_{}_{}captain_action_history.npy'.format(self.args.algorithm, self.args.N,
                                                                    self.seed, current_time), np.array(self.captain_action_history_np))
        self.save_model()

    def evaluate_policy(self, no_captain = False):
        win_times = 0
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            win_tag, episode_reward, _ = self.run_episode_smac(evaluate=True, no_captain = no_captain)
            if win_tag:
                win_times += 1
            evaluate_reward += episode_reward

        win_rate = win_times / self.args.evaluate_times
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.win_rates.append(win_rate)
        print("total_steps:{} \t win_rate:{} \t evaluate_reward:{}".format(self.total_steps, win_rate, evaluate_reward))
         
    def run_episode_smac(self, evaluate=False, extrenal_test = False, no_captain = False):
        win_tag = False
        episode_reward = 0
        self.env.reset()
        self.shadow_healths = np.zeros(self.args.N)
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            self.agent_n.eval_Q_net.rnn_hidden = None
        last_onehot_a_n = np.zeros((self.args.N, self.args.action_dim))  # Last actions of N agents(one-hot)
        captain_last_onehot_a_n = np.zeros((self.args.N, self.captain_action_dim))  # Last actions of N agents(one-hot)
        captian_avail_a_n = self.captain_avail_a_n  # Last actions of N agents(one-hot)
        epsilon = 0 if evaluate else self.epsilon
        i = 0
        for episode_step in range(self.args.episode_limit):
            i += 1
            obs_n = self.env.get_obs()  # obs_n.shape=(N,obs_dim)
            one_hot_sights = self.captain.choose_action(obs_n, captain_last_onehot_a_n, captian_avail_a_n, epsilon)
            values = (np.array(one_hot_sights, dtype=np.float16)-1)*3+9
            if not no_captain:
                self.env.change_unit_capability_range(values)       # Captian changes sights and own we have new obs
            obs_n = self.env.get_obs()  # obs_n.shape=(N,obs_dim)
            s = self.env.get_state()  # s.shape=(state_dim,)
            avail_a_n = self.env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)
            a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, avail_a_n, epsilon)
            last_onehot_a_n = np.eye(self.args.action_dim)[a_n]  # Convert actions to one-hot vectors
            r, done, info = self.env.step(a_n)  # Take a step
            self.reward_list.append(r)
            ########################### captain ###########################
            # self.reduce_health_by_sight()
            for j in range(self.args.N): #set dead agents sight to almost 9
                if  obs_n[j][0] == 0.0:
                    values[j] = 9.0625
            self.captain_action_history.append(values)    
            captain_reward = r + (9**2*self.args.N-(self.env.visions**2).sum())/self.kill_sight_reward_ratio/self.args.N  # captain reward 
            self.captain_reward_list.append(captain_reward)
            
            ########################### captain ###########################
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            episode_reward += r

            if not evaluate:
                if self.args.use_reward_norm:
                    r = self.reward_norm(r)
                """"
                    When dead or win or reaching the episode_limit, done will be Ture, we need to distinguish them;
                    dw means dead or win,there is no next state s';
                    but when reaching the max_episode_steps,there is a next state s' actually.
                """
                if done and episode_step + 1 != self.args.episode_limit:
                    dw = True
                else:
                    dw = False

                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, r, dw)
                self.captain_replay_buffer.store_transition(episode_step, obs_n, s, self.captain_avail_a_n, captain_last_onehot_a_n,
                                                             one_hot_sights, captain_reward, dw)
                # Decay the epsilon
                self.epsilon = self.epsilon - self.args.epsilon_decay if self.epsilon - self.args.epsilon_decay > self.args.epsilon_min else self.args.epsilon_min

            if done:
                break
        
        if not evaluate:
            # An episode is over, store obs_n, s and avail_a_n in the last step
            obs_n = self.env.get_obs()
            s = self.env.get_state()
            avail_a_n = self.env.get_avail_actions()
            self.replay_buffer.store_last_step(episode_step + 1, obs_n, s, avail_a_n)
            self.captain_replay_buffer.store_last_step(episode_step + 1, obs_n, s, self.captain_avail_a_n)
        self.episode_reward_list.append(episode_reward)
        
        if extrenal_test:
            return win_tag, self.captain_action_history[-i:], self.reward_list[-i:], i
        
        return win_tag, episode_reward, episode_step + 1
 
    def reduce_health_by_sight(self):
        to_cemetery = []
        state_dict = self.env.get_state_dict()['allies']
        for al_id, al_unit in self.env.env.agents.items():
            vision = self.env.visions[al_id]
            self.shadow_healths[al_id] -= (9**2 - vision**2)/5000
            self.shadow_healths[al_id] = max(0, self.shadow_healths[al_id])
            self.shadow_healths[al_id] = min(1, self.shadow_healths[al_id])
            
            if self.shadow_healths[al_id]>= state_dict[al_id][0]:
                if state_dict[al_id][0] > 0:
                    to_cemetery.append(al_unit.tag)
        if len(to_cemetery):
            self.env._kill_units(to_cemetery)
        
            
 
    def remove_dead_from_list(self, start_from_last_indexes = None):
        if start_from_last_indexes is None:
            captain_action_history = []
            captain_list_len = len(self.captain_action_history)
            for i in range(self.args.N):
                agent_i = []
                for j in range(captain_list_len):
                    if self.captain_action_history[j][i] != 9.0625:
                        agent_i.append(self.captain_action_history[j][i])
                captain_action_history.append(agent_i.copy())
        else:       # when you run an episode and not a long run
            captain_action_history = []
            for i in range(self.args.N):
                agent_i = []
                for j in range(start_from_last_indexes):
                    if self.captain_action_history[j-start_from_last_indexes][i] != 9.0625:
                        agent_i.append(self.captain_action_history[j][i])
                captain_action_history.append(agent_i.copy())
        return captain_action_history
        
    def save_model(self):
         torch.save(self.agent_n.eval_Q_net.state_dict(), "./model/model_{}_{}m_total_steps_{}_ratio_{}.pth".format( 
                                    self.args.algorithm, self.args.N, self.args.max_train_steps, self.args.kill_sight_reward_ratio))
         
         torch.save(self.captain.eval_Q_net.state_dict(), "./model/captain_model_{}_{}m_total_steps_{}_ratio_{}.pth".format( 
                                    self.args.algorithm, self.args.N, self.args.max_train_steps, self.args.kill_sight_reward_ratio))
    
    def load_model(self):
        self.agent_n.eval_Q_net.load_state_dict(torch.load("./model/model_{}_{}m_total_steps_{}_ratio_{}.pth".format( 
                                    self.args.algorithm, self.args.N, self.args.max_train_steps, self.args.kill_sight_reward_ratio)))
        
        self.captain.eval_Q_net.load_state_dict(torch.load("./model/captain_model_{}_{}m_total_steps_{}_ratio_{}.pth".format(
                                    self.args.algorithm, self.args.N, self.args.max_train_steps, self.args.kill_sight_reward_ratio)))
    
    def plot(self, average = 5000):
        plt.plot(self.win_rates)
        plt.title('Win Rate')
        plt.figure()
        plt.plot(moving_average(self.reward_list, average))
        plt.title('reward list')
        plt.figure()
        plt.plot(moving_average(self.episode_reward_list, average//7+1))
        plt.title('episode_reward listward list')
        plt.figure()
        plt.plot(moving_average(self.captain_reward_list, average))
        plt.title('captain reward list')
        plt.figure()
        captain_action_history = self.remove_dead_from_list()
        for i in range(self.args.N):
            plt.plot(moving_average(captain_action_history[i], average), label = 'agent' + str(i));
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left');
        plt.title('captain action list')


class Runner_SMAC_QMIX_Continuous_PPO:
    def __init__(self, args, number, seed):
        self.args = args
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.distribution_config = {
                        "n_units": self.args.N_agents,
                        "team_gen": {
                            "dist_type": "weighted_teams",
                            "unit_types": ["marine", "marauder", "medivac"],
                            "exception_unit_types": ["medivac"],
                            "weights": [1, 0, 0],
                            "observe": True,
                        },
                        "start_positions": {
                            "dist_type": "surrounded_and_reflect",
                            "p": 0.5,
                            "map_x": 32,
                            "map_y": 32,
                        },
                        "n_enemies": self.args.N_agents,
                    }
        self.env = StarCraftCapabilityEnvWrapper(
                    capability_config=self.distribution_config,
                    map_name="10gen_terran",
                    debug=False,
                    conic_fov=False,
                    obs_own_pos=True,
                    use_unit_ranges=False,
                    min_attack_range=2,
                    seed = self.seed)
        self.env_info = self.env.get_env_info()
        self.args.N = self.env_info["n_agents"]  # The number of agents
        self.args.obs_dim = self.env_info["obs_shape"]  # The dimensions of an agent's observation space
        self.args.state_dim = self.env_info["state_shape"]  # The dimensions of global state space
        self.captain_action_dim = 3  # The dimensions of an captain's action space
        self.args.action_dim = self.env_info["n_actions"]  # The dimensions of an agent's action space
        self.args.episode_limit = self.env_info["episode_limit"]  # Maximum number of steps per episode
        self.action_std_decay_freq = 10_000
        self.action_std_decay_rate = 0.1
        self.captain_update_timestep = 1000
        print("number of agents={}".format(self.args.N))
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        # Create N agents
        self.agent_n = QMIX_SMAC(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        self.list_of_agents_sight_networks = []


        for i in range(self.args.N):
            self.list_of_agents_sight_networks.append(continuous_ppo.PPO(state_dim=self.args.obs_dim, action_dim=1 ,lr_actor=5e-4,
                                      lr_critic = 5e-4, gamma=0.99, K_epochs = 40, eps_clip = 0.2, has_continuous_action_space = True,
                                        action_std_init = 3.0 , action_std_decay_rate = self.action_std_decay_rate))

        # Create a tensorboard
        
        
        self.epsilon = self.args.epsilon  # Initialize the epsilon
        self.kill_sight_reward_ratio = self.args.kill_sight_reward_ratio
        self.win_rates = []  # Record the win rates
        self.total_steps = 0
        self.captain_action_history = []
        self.reward_list = []
        self.episode_reward_list = []
        self.captain_reward_list = []
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=1)

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, _, episode_steps = self.run_episode_smac(evaluate=False)  # Run an episode


            
            if self.replay_buffer.current_size >= self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Training

        self.evaluate_policy()
        self.env.close()
        
        self.captain_action_history = np.array(self.captain_action_history, dtype = np.float16)
        self.reward_list = np.array(self.reward_list)
        self.episode_reward_list = np.array(self.episode_reward_list)
        self.captain_reward_list = np.array(self.captain_reward_list)
        # Save the win rates
        current_time = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.datetime.now())
        np.save('./data_train/{}_env_{}m_PPO_seed_{}_{}.npy'.format(self.args.algorithm, self.args.N,
                                                                    self.seed, current_time), np.array(self.win_rates))
        np.save('./data_train/{}_env_{}m_PPO_seed_{}_{}captain_action_history.npy'.format(self.args.algorithm, self.args.N,
                                                                    self.seed, current_time), np.array(self.captain_action_history))

    def evaluate_policy(self, ):
        win_times = 0
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            win_tag, episode_reward, _ = self.run_episode_smac(evaluate=True)
            if win_tag:
                win_times += 1
            evaluate_reward += episode_reward

        win_rate = win_times / self.args.evaluate_times
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.win_rates.append(win_rate)
        print("total_steps:{} \t win_rate:{} \t evaluate_reward:{}".format(self.total_steps, win_rate, evaluate_reward))
        
    def run_episode_smac(self, evaluate=False):
        win_tag = False
        episode_reward = 0
        self.env.reset()
        self.shadow_healths = np.zeros(self.args.N)
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            self.agent_n.eval_Q_net.rnn_hidden = None
        last_onehot_a_n = np.zeros((self.args.N, self.args.action_dim))  # Last actions of N agents(one-hot)
        i = 0
        for episode_step in range(self.args.episode_limit):
            self.total_steps += 1
            i += 1
            values = []
            obs_n = self.env.get_obs()  # obs_n.shape=(N,obs_dim)
            for n in range(self.args.N):
                values.append(self.list_of_agents_sight_networks[n].select_action(obs_n[n]))
            

            self.captain_action_history.append(values)
            self.env.change_unit_capability_range(values)       # Captain changes sights and now we have new obs
            obs_n = self.env.get_obs()  # obs_n.shape=(N,obs_dim)
            s = self.env.get_state()  # s.shape=(state_dim,)
            avail_a_n = self.env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)
            epsilon = 0 if evaluate else self.epsilon
            a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, avail_a_n, epsilon)
            last_onehot_a_n = np.eye(self.args.action_dim)[a_n]  # Convert actions to one-hot vectors
            r, done, info = self.env.step(a_n)  # Take a step
            self.reward_list.append(r)
            ########################### captain ###########################
            # self.reduce_health_by_sight()
            for j in range(self.args.N): #set dead agents sight to almost 9
                if  obs_n[j][0] == 0.0:
                    values[j] = 9.0625
            self.captain_action_history.append(values)       
            captain_reward = r + (9**2*self.args.N-(self.env.visions**2).sum())/self.kill_sight_reward_ratio/self.args.N  # captain reward 
            self.captain_reward_list.append(captain_reward)
            for agent_ID in range(self.args.N):
                self.list_of_agents_sight_networks[agent_ID].add_to_buffer(captain_reward, done)       # agent sight network experience replay
            ########################### captain ###########################
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            episode_reward += r

            if not evaluate:
                if self.args.use_reward_norm:
                    r = self.reward_norm(r)
                """"
                    When dead or win or reaching the episode_limit, done will be True, we need to distinguish them;
                    dw means dead or win,there is no next state s';
                    but when reaching the max_episode_steps,there is a next state s' actually.
                """
                if done and episode_step + 1 != self.args.episode_limit:
                    dw = True
                else:
                    dw = False

                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, r, dw)
                # Decay the epsilon
                self.epsilon = self.epsilon - self.args.epsilon_decay if self.epsilon - self.args.epsilon_decay > self.args.epsilon_min else self.args.epsilon_min
                values = [int((value-9))/3+1 for value in values]
                for n1 in range(self.args.N):
                                        
                    if self.total_steps % self.captain_update_timestep == 0:
                        self.list_of_agents_sight_networks[n1].update()

                    # if continuous action space; then decay action std of ouput action distribution
                    if self.total_steps % self.action_std_decay_freq == 0:
                        self.list_of_agents_sight_networks[n1].decay_action_std()


            if done:
                break
        if not evaluate:
            # An episode is over, store obs_n, s and avail_a_n in the last step
            obs_n = self.env.get_obs()
            s = self.env.get_state()
            avail_a_n = self.env.get_avail_actions()
            self.replay_buffer.store_last_step(episode_step + 1, obs_n, s, avail_a_n)
        self.episode_reward_list.append(episode_reward)
        return win_tag, episode_reward, episode_step + 1
    
    def remove_dead_from_list(self, start_from_last_indexes = None):
        if start_from_last_indexes is None:
            captain_action_history = []
            captain_list_len = len(self.captain_action_history)
            for i in range(self.args.N):
                agent_i = []
                for j in range(captain_list_len):
                    if self.captain_action_history[j][i] != 9.0625:
                        agent_i.append(self.captain_action_history[j][i])
                captain_action_history.append(agent_i.copy())
        else:       # when you run an episode and not a long run
            captain_action_history = []
            for i in range(self.args.N):
                agent_i = []
                for j in range(start_from_last_indexes):
                    if self.captain_action_history[j-start_from_last_indexes][i] != 9.0625:
                        agent_i.append(self.captain_action_history[j][i])
                captain_action_history.append(agent_i.copy())
        return captain_action_history
        
    def save_model(self):
         torch.save(self.agent_n.eval_Q_net.state_dict(), "./model/model_{}_{}m_total_steps_{}_ratio_{}.pth".format( 
                                    self.args.algorithm, self.args.N, self.args.max_train_steps, self.args.kill_sight_reward_ratio))
         
         torch.save(self.captain.eval_Q_net.state_dict(), "./model/captain_model_{}_{}m_total_steps_{}_ratio_{}.pth".format( 
                                    self.args.algorithm, self.args.N, self.args.max_train_steps, self.args.kill_sight_reward_ratio))
    
    def load_model(self):
        self.agent_n.eval_Q_net.load_state_dict(torch.load("./model/model_{}_{}m_total_steps_{}_ratio_{}.pth".format( 
                                    self.args.algorithm, self.args.N, self.args.max_train_steps, self.args.kill_sight_reward_ratio)))
        
        self.captain.eval_Q_net.load_state_dict(torch.load("./model/captain_model_{}_{}m_total_steps_{}_ratio_{}.pth".format(
                                    self.args.algorithm, self.args.N, self.args.max_train_steps, self.args.kill_sight_reward_ratio)))
    
    def plot(self, average = 5000):
        plt.plot(self.win_rates)
        plt.title('Win Rate')
        plt.figure()
        plt.plot(moving_average(self.reward_list, average))
        plt.title('reward list')
        plt.figure()
        plt.plot(moving_average(self.episode_reward_list, average//7+1))
        plt.title('episode_reward listward list')
        plt.figure()
        plt.plot(moving_average(self.captain_reward_list, average))
        plt.title('captain reward list')
        plt.figure()
        captain_action_history = self.remove_dead_from_list()
        for i in range(self.args.N):
            plt.plot(moving_average(captain_action_history[i], average), label = 'agent' + str(i));
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left');
        plt.title('captain action list')
