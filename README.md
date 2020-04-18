# StarCraft

This is a pytorch implementation of the multi-agent reinforcement learning algorithms, including [QMIX](https://arxiv.org/abs/1803.11485), [VDN](https://arxiv.org/abs/1706.05296), [COMA](https://arxiv.org/abs/1705.08926), [QTRAN](https://arxiv.org/abs/1905.05408)(both **QTRAN-base** and **QTRAN-alt**), [CommNet](https://arxiv.org/abs/1605.07736), [DyMA-CL](https://arxiv.org/abs/1909.02790?context=cs.MA), and [G2ANet](https://arxiv.org/abs/1911.10715), which are the state of art MARL algorithms. In addition, because CommNet and G2ANet need a external training algorithm, you can combine them with COMA, we also provide **Central-V** and **REINFORCE** for them to training. We trained these algorithms on [SMAC](https://github.com/oxwhirl/smac), the decentralised micromanagement scenario of [StarCraft II](https://en.wikipedia.org/wiki/StarCraft_II:_Wings_of_Liberty).

## Corresponding Papers

- [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296)
- [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)
- [Learning Multiagent Communication with Backpropagation](https://arxiv.org/abs/1605.07736)
- [From Few to More: Large-scale Dynamic Multiagent Curriculum Learning](https://arxiv.org/abs/1909.02790?context=cs.MA)
- [Multi-Agent Game Abstraction via Graph Attention Neural Network](https://arxiv.org/abs/1911.10715)

## Requirements

- python
- torch
- [SMAC](https://github.com/oxwhirl/smac)
- [pysc2](https://github.com/deepmind/pysc2)

## Acknowledgement

+ [SMAC](https://github.com/oxwhirl/smac)
+ [pymarl](https://github.com/oxwhirl/pymarl)


## TODO List

- [x] Add CUDA option
- [x] DyMA-CL
- [x] G2ANet
- [ ] MAVEN
- [ ] VBC
- [ ] Other SOTA MARL algorithms
- [ ] Update results on other maps

## Quick Start

```shell
$ python main.py --map=3m --alg=qmix
```

Directly run the `main.py`, then the algorithm will start **training** on map `3m`， and the difficulty is **7(VeryHard)**. **Note** CommNet and G2ANet need a external training algorithm, so the name of the algorithm is like `reinforce+commnet` or `central_v+g2anet`, all the algorithms we provide are written on  `./common/arguments.py`.

If you just want to use this project for demonstration, you should set `--learn=False --load_model=True`. **Note** we trained these models on GPU, if you don't have CUDA, you need to train the model first.

The running of DyMA-CL is independent from others because it requires different environment settings, so we put it on another project. For more details, please read [DyMA-CL documentation](https://github.com/starry-sky6688/DyMA-CL/blob/master).

## Result

We independently train these algorithms for 8 times and take the mean of the 8 independent results. We train these algorithms for 20000 episodes and evaluate them for every 100 episodes. All of the results are saved in  `./result`.
Result on other maps are still in training, we will update them later.

### 1. Mean Win Rate of 8 Independent Runs on `3m --difficulty=7(VeryHard)`
<div align=center><img width = '600' height ='300' src ="https://github.com/starry-sky6688/StarCraft/blob/master/result/overview.png"/></div>