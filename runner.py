import numpy as np
from common.rollout import RolloutWorker
from agent.agent import Agents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
from tqdm import tqdm
from smac.env import StarCraft2Env


class Runner:
    def __init__(self, env, args):
        self.env = env
        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        self.buffer = ReplayBuffer(args)
        self.args = args
        self.epsilon = args.epsilon

        # 用来在一个稀疏奖赏的环境上评估算法的好坏，胜利为1，失败为-1，其他普通的一步为0
        self.env_evaluate = StarCraft2Env(map_name=args.map,
                                          step_mul=args.step_mul,
                                          difficulty=args.difficulty,
                                          game_version=args.game_version,
                                          seed=args.seed,
                                          replay_dir=args.replay_dir,
                                          reward_sparse=True,
                                          reward_scale=False)
        self.evaluateWorker = RolloutWorker(self.env_evaluate, self.agents, args)

    def run(self):
        plt.figure()
        plt.axis([0, self.args.n_epoch, 0, 100])
        win_rates = []
        episode_rewards = []
        train_steps = 0
        for epoch in tqdm(range(self.args.n_epoch)):
            # print('Train epoch {} start'.format(epoch))
            self.epsilon = self.epsilon - 0.0001125 if self.epsilon > 0.05 else self.epsilon
            episodes = []
            # 收集self.args.n_episodes个episodes
            for episode_idx in range(self.args.n_episodes):
                episode, _ = self.rolloutWorker.generate_episode(self.epsilon)
                episodes.append(episode)
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            self.buffer.store_episode(episode_batch)
            if self.buffer.current_size > 100:
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(self.args.batch_size)
                    self.agents.train(mini_batch, train_steps)
                    train_steps += 1
                win_rate, episode_reward = self.evaluate()
                # print('win_rate is ', win_rate)
                win_rates.append(win_rate)
                episode_rewards.append(episode_reward)
                # 可视化
                if epoch % 100 == 0:
                    plt.cla()
                    plt.subplot(2, 1, 1)
                    plt.plot(range(len(win_rates)), win_rates)
                    plt.xlabel('epoch')
                    plt.ylabel('win_rate')

                    plt.subplot(2, 1, 2)
                    plt.plot(range(len(episode_rewards)), episode_rewards)
                    plt.xlabel('epoch')
                    plt.ylabel('episode_rewards')

                    plt.savefig(self.args.result_dir + '/plt.png', format='png')
                    np.save(self.args.result_dir + '/win_rates', win_rates)
                    np.save(self.args.result_dir + '/episode_rewards', episode_rewards)

        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(win_rates)), win_rates)
        plt.xlabel('epoch')
        plt.ylabel('win_rate')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(episode_rewards)), episode_rewards)
        plt.xlabel('epoch')
        plt.ylabel('episode_rewards')

        plt.savefig(self.args.result_dir + '/plt.png', format='png')
        np.save(self.args.result_dir + '/win_rates', win_rates)
        np.save(self.args.result_dir + '/episode_rewards', episode_rewards)

    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward = self.rolloutWorker.generate_episode(0)
            episode_rewards += episode_reward
            if episode_reward > self.args.threshold:
                win_number += 1
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    def evaluate_sparse(self):
        win_number = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward = self.evaluateWorker.generate_episode(0)
            result = 'win' if episode_reward > 0 else 'defeat'
            print('Epoch {}: test reslut is {}'.format(epoch, result))
            if episode_reward > 0:
                win_number += 1
        return win_number / self.args.evaluate_epoch








