import numpy as np


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon

    def generate_episode(self, episode_num=None, evaluate=False):
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False
        step = 0
        episode_reward = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)  # 初始化hidden_state
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        while not terminated:
            # time.sleep(0.2)
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                if self.args.map == '3m':   # 3m的后5个动作一直为0，因为它表示攻击另外5个agent
                    avail_action += [0, 0, 0, 0, 0]
                # 输入当前agent上一个时刻的动作
                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id, avail_action, epsilon, evaluate)
                # 生成对应动作的0 1向量
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            reward, terminated, _ = self.env.step(actions)
            if step == self.episode_limit - 1:
                terminated = True

            o.append(obs)
            s.append(state)
            # 和环境交互的actions需要是一个list，里面就装着代表每个agent动作的整数
            # buffer里存的action，每个agent的动作都需要是一个1维向量
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            # if terminated:
            #     time.sleep(1)
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # 处理最后一个obs
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # 最后一个obs需要单独计算一下avail_action，到时候需要计算target_q
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            if self.args.map == '3m':  # 3m的后5个动作一直为0，因为它表示攻击另外5个agent
                avail_action += [0, 0, 0, 0, 0]
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # 返回的episode必须长度都是self.episode_limit，所以不够的话进行填充
        for i in range(step, self.episode_limit):  # 没有的字段用0填充，并且padded为1
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])
        avail_u_next = np.array(avail_u_next.copy())
        '''
        (o[n], u[n], r[n], o_next[n], avail_u[n], u_onehot[n])组成第n条经验，各项维度都为(episode数，transition数，n_agents, 自己的具体维度)
         因为avail_u表示当前经验的obs可执行的动作，但是计算target_q的时候，需要obs_net及其可执行动作，
        '''
        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
            # print('Epsilon is ', self.epsilon)
        return episode, episode_reward
        # 因为buffer里存的是四维的，这里得到的episode只有三维，即transition、agent、shape三个维度，
        # 还差一个episode维度，所以给它加一维