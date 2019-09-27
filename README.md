# StarCraft
This is a pytorch implementation of the multi-agent reinforcement learning algorithms, [QMIX](https://arxiv.org/abs/1803.11485),[VDN](https://arxiv.org/abs/1706.05296) and [COMA](https://arxiv.org/abs/1705.08926), which are the state of art MARL algorithms. We trained these algorithms on [SMAC](https://github.com/oxwhirl/smac), the decentralised micromanagement scenario of [StarCraft II](https://en.wikipedia.org/wiki/StarCraft_II:_Wings_of_Liberty).

## Corresponding Papers

- QMIX: [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- VDN: [Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296)
- COMA: [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)

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
Although QMIX, VDN and COMA are the state of art multi-agent algorithms, they are unstable sometimes. If you want the same results as in the papers, you need to independently run several times(more than 10) and take the median or mean of them.

### 1. Win Rate of QMIX in Two Independent Runs on '3m'
<div align=center><img width = '500' height ='400' src ="https://github.com/starry-sky6688/StarCraft/blob/master/model/qmix/3m/compare.png"/></div>

### 2. Win Rate of VDN in Two Independent Runs on '3m'
<div align=center><img width = '500' height ='400' src ="https://github.com/starry-sky6688/StarCraft/blob/master/model/vdn/3m/compare.png"/></div>

### 3. Win Rate of COMA in a Run on '3m'
<div align=center><img width = '500' height ='200' src ="https://github.com/starry-sky6688/StarCraft/blob/master/model/coma/3m/plt.png"/></div>
