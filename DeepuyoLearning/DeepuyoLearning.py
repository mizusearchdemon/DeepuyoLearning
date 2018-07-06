#!/usr/bin/python

import os
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import numpy as np
import cupy

# Set up the logger to print info messages for understandability.
import logging
import sys
gym.undo_logger_setup()  # Turn off gym's default logger settings
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

import puyoenv

def test():
    field0 = np.array(
        [
            [3,2,2,1,3,1,1,3,4,1,3,0,0,0],
            [3,2,3,1,3,4,2,1,2,4,4,3,0,0],
            [3,4,3,1,3,4,2,1,0,0,0,0,0,0],
            [2,3,4,2,4,2,1,2,1,0,0,0,0,0],
            [2,3,2,4,2,1,4,1,3,1,0,0,0,0],
            [2,3,4,4,2,1,4,4,3,1,1,0,0,0],
        ],
        dtype=np.uint8,
    )
    pe = puyoenv.PuyoEnv(field = field0)
    h = pe.get_reachable_height(current_line=2, current_height=8)
    
    field = np.array(
        [
            [3,2,2,1,3,1,1,3,4,1,3,3,2,6],
            [3,2,3,1,3,4,2,1,2,4,4,3,4,6],
            [3,4,3,1,3,4,2,1,2,4,2,3,0,6],
            [2,3,4,2,4,2,1,2,1,2,1,0,0,6],
            [2,3,2,4,2,1,4,1,3,1,3,0,0,6],
            [2,3,4,4,2,1,4,4,3,1,3,4,2,6],
        ],
        dtype=np.uint8,
    )
    pe = puyoenv.PuyoEnv(field = field)
    print(pe.field_to_url())

    rensa, point = pe.exec_rensa()  # 18連鎖, 159180点
    h = pe.get_reachable_height
    print(rensa, point)

    url = "http://www.inosendo.com/puyo/rensim/??700067600555554765447664554776647445664774447664776555667444676557555677777555"
    pe = puyoenv.PuyoEnv()
    pe.url_to_field(url)
    print(pe.field)

#test()
#exit()

env = puyoenv.PuyoEnv()
# reset は環境をリセットして現在の観測を返す
obs = env.reset()
action = 0
# step は環境にアクションを送り，4つの値（次の観測，報酬，エピソード終端かどうか，追加情報）を返す
obs, r, done, info = env.step(action)

class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=50):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_hidden_channels)
            self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l2 = L.Linear(n_hidden_channels, n_actions)

    def __call__(self, x, test=False):
        """
        Args:
            x (ndarray or chainer.Variable): An observation
            test (bool): a flag indicating whether it is in test mode
        """
        h = F.tanh(self.l0(x))
        h = F.tanh(self.l1(h))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))

obs_size = env.observation_space.shape[1]
n_actions = env.action_space.n
q_func = QFunction(obs_size, n_actions)
q_func.to_gpu()

# Use Adam to optimize q_func. eps=1e-2 is for stability.
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)

# Set the discount factor that discounts future rewards.
gamma = 0.95

# Use epsilon-greedy for exploration
explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space.sample)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

# Since observations from CartPole-v0 is numpy.float64 while
# Chainer only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
phi = lambda x: x.astype(np.float32, copy=False)

# Now create an agent that will interact with the environment.
agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500, update_interval=1, gpu=0,
    target_update_interval=100, phi=phi)

def learn(agent, env, dir):
    if (dir is not None):
        agent.load(sys.argv[1])

    chainerrl.experiments.train_agent_with_evaluation(
        agent, env,
        steps=10000,          # Train the agent for 2000 steps
        eval_n_runs=100,      # 10 episodes are sampled for each evaluation
        max_episode_len=2000, # Maximum length of each episodes
        eval_interval=1000,   # Evaluate the agent after every 1000 steps
        outdir='result')      # Save everything to 'result' directory
    
def play(agent, env, dir = None, seed = None):
    if (dir is not None):
        agent.load(sys.argv[1])

    #print(env.observation_space)
    chainerrl.misc.draw_computational_graph(
        [q_func(cupy.zeros_like(env.observation_space, dtype=np.float32))],
        os.path.join('.\model'))

    max_r = 0
    max_seed = None
    max_actions = None
    max_obs = None
    for i in range(1000):
        #print({'i': i})
        action_list = []
        obs = env.reset()
        if seed is not None:
            obs = env.reset(seed)

        url = env.field_to_url()
        for step in range(100):
            action = agent.act(obs)
            #print({'action': action})
            action_list.append(action)

            obs, r, done, info = env.step(action)
            #print({"step": step, "r": r, "url": env.field_to_url()})
            if r > max_r:
                max_r = r
                max_actions = action_list
                max_obs = obs
                max_url = url
                max_seed = env.seed
                print({"max_r": max_r, "seed": max_seed})
                print(url)
                print({"actions": env.as2lr(max_actions)})
                print({"tsumo": env.tsumo_list[0:len(max_actions)]})
            url = env.field_to_url()
            if done:
                break
        agent.stop_episode()
        if seed is not None:
            break

dir = None
if len(sys.argv) > 1:
    dir = sys.argv[1]

#learn(agent, env, dir)
#play(agent, env, dir)
play(agent, env, dir, seed=2491495141)
