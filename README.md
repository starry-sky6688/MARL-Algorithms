# StarCraft

This is a pytorch implementation of the multi-agent reinforcement learning algorithms, including [QMIX](https://arxiv.org/abs/1803.11485), [VDN](https://arxiv.org/abs/1706.05296), [COMA](https://arxiv.org/abs/1705.08926), and [QTRAN](https://arxiv.org/abs/1905.05408)(both QTRAN-base and QTRAN-alt), which are the state of art MARL algorithms. In addition, we implemented [CommNet](https://arxiv.org/abs/1605.07736) and combined it with coma, which we called CommNet_COMA. We trained these algorithms on [SMAC](https://github.com/oxwhirl/smac), the decentralised micromanagement scenario of [StarCraft II](https://en.wikipedia.org/wiki/StarCraft_II:_Wings_of_Liberty).

## Corresponding Papers

- [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296)
- [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)
- [Learning Multiagent Communication with Backpropagation](https://arxiv.org/abs/1605.07736)

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
$ python main.py --evaluate_epoch=100 --map=3m --alg=qmix
```

Directly run the main.py, then the algorithm will be tested on map '3m' for 100 episodes, using the pretrained model.

## Result

We independently train these algorithms for 8 times and take the mean of the 8 independent results. In order to make the curves smoother, we also take the mean of every five points in the horizontal direction. In each independent training process, we train these algorithms for 5000 epochs and evaluate them for every 5 epochs.
From the figure 1 we can see that our results is not the same as in the papers, maybe there are some small bugs, we are pleasure that you pull request to improve this project.
Furthermore, as show in figure 2, we compare the best performance in the 8 independent results of these algorithms.

### 1. Mean Win Rate of 8 Independent Runs on '3m'
<div align=center><img width = '500' height ='250' src ="https://github.com/starry-sky6688/StarCraft/blob/master/result/overview.png"/></div>

### 2. Best Win Rate of 8 Independent Runs on '3m'
<div align=center><img width = '500' height ='250' src ="https://github.com/starry-sky6688/StarCraft/blob/master/result/best/best.png"/></div>
