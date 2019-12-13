# StarCraft
This is a pytorch implementation of the multi-agent reinforcement learning algorithms, including [QMIX](https://arxiv.org/abs/1803.11485), [VDN](https://arxiv.org/abs/1706.05296), [COMA](https://arxiv.org/abs/1705.08926), and [QTRAN](https://arxiv.org/abs/1905.05408), which are the state of art MARL algorithms. We trained these algorithms on [SMAC](https://github.com/oxwhirl/smac), the decentralised micromanagement scenario of [StarCraft II](https://en.wikipedia.org/wiki/StarCraft_II:_Wings_of_Liberty).

## Corresponding Papers

- [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296)
- [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)

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
Although QMIX, VDN, COMA and QTRAN are the state of art multi-agent algorithms, they are unstable sometimes. If you want the same results as in the papers, you need to independently run several times(more than 10) and take the median or mean of them.

### 1. Win Rate of QMIX in Two Independent Runs on '3m'
<div align=center><img width = '500' height ='400' src ="https://github.com/starry-sky6688/StarCraft/blob/master/model/qmix/3m/compare.png"/></div>

### 2. Win Rate of VDN in Two Independent Runs on '3m'
<div align=center><img width = '500' height ='400' src ="https://github.com/starry-sky6688/StarCraft/blob/master/model/vdn/3m/compare.png"/></div>

### 3. Win Rate of COMA in One Run on '8m'
<div align=center><img width = '500' height ='200' src ="https://github.com/starry-sky6688/StarCraft/blob/master/model/coma/8m/plt.png"/></div>


### 4. Win Rate of QTRAN-base in One Run on '3m'
<div align=center><img width = '500' height ='400' src ="https://github.com/starry-sky6688/StarCraft/blob/master/model/qtran_base/3m/plt.png"/></div>
