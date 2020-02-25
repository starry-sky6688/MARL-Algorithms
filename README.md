# StarCraft

This is a pytorch implementation of the multi-agent reinforcement learning algorithms, including [QMIX](https://arxiv.org/abs/1803.11485), [VDN](https://arxiv.org/abs/1706.05296), [COMA](https://arxiv.org/abs/1705.08926), [QTRAN](https://arxiv.org/abs/1905.05408)(both QTRAN-base and QTRAN-alt), [CommNet](https://arxiv.org/abs/1605.07736), [DyMA-CL](https://arxiv.org/abs/1909.02790?context=cs.MA), and [G2ANet](https://arxiv.org/abs/1911.10715), which are the state of art MARL algorithms. In addition, because CommNet and G2ANet need a external training algorithm, you can combine them with COMA, we also provide `__central-v__` and `__reinforce__` for them to training. We trained these algorithms on [SMAC](https://github.com/oxwhirl/smac), the decentralised micromanagement scenario of [StarCraft II](https://en.wikipedia.org/wiki/StarCraft_II:_Wings_of_Liberty).

##

- [x] Add CUDA option

## Corresponding Papers

- [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296)
- [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)
- [Learning Multiagent Communication with Backpropagation](https://arxiv.org/abs/1605.07736)
- [From Few to More: Large-scale Dynamic Multiagent Curriculum Learning](https://arxiv.org/abs/1909.02790?context=cs.MA)
- [Multi-Agent Game Abstraction via Graph Attention Neural Network](https://arxiv.org/abs/1911.10715)

## Requirements

- python=3.6.5
- torch=1.2.0
- [SMAC](https://github.com/oxwhirl/smac)
- [pysc2](https://github.com/deepmind/pysc2)

## Acknowledgement

+ [SMAC](https://github.com/oxwhirl/smac)
+ [pymarl](https://github.com/oxwhirl/pymarl)

## Quick Start

```shell
$ python main.py --map=3m --alg=qmix
```

Directly run the main.py, then the algorithm will start training on map '3m'. Note CommNet and G2ANet need a external training algorithm, so the name of the algorithm is like `reinforce+commnet` or `central_v+g2anet`, all the algorithms we provide are written on  `./common/arguments`.  The running of DyMA-CL is independent from others beacuse it requires different environment settings, you should open it as a new project, for more details, please read [DyMA-CL documentation](dyma/README.md).

## Result

We independently train these algorithms for 8 times and take the mean of the 8 independent results. In order to make the curves smoother, we also take the mean of every five points in the horizontal direction. In each independent training process, we train these algorithms for 5000 epochs and evaluate them for every 5 epochs. Furthermore, as show in figure 2, we compare the best result we think in the 8 independent results. All of the results are saved in  `./result`.

### 1. Mean Win Rate of 8 Independent Runs on '3m'
<div align=center><img width = '600' height ='300' src ="https://github.com/starry-sky6688/StarCraft/blob/master/result/overview.png"/></div>

### 2. Best Win Rate of 8 Independent Runs on '3m'
<div align=center><img width = '600' height ='300' src ="https://github.com/starry-sky6688/StarCraft/blob/master/result/best/best.png"/></div>
