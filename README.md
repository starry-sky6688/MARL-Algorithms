# StarCraft
This is a pytorch implementation of the multi-agent algrithms, [QMIX](https://arxiv.org/abs/1803.11485) and [VDN](https://arxiv.org/abs/1706.05296), both of which are the state of art multi-agent algrithms.We trained these algrithms on [SMAC](https://github.com/oxwhirl/smac), which is the decentralised micromanagement scenario of [StarCraft II](https://en.wikipedia.org/wiki/StarCraft_II:_Wings_of_Liberty)

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
$ python main.py --evaluate_epoch=100
```

Directly run the main.py, then the two algrithms will be tested for 100 episodes seperately, using the pretrained model.

## Result
Although qmix and vdn are the state of art multi-agent algrithms, they are unstable sometimes, you need to independently run several times to get better performence.

### 1. Win Rate of QMIX in Two Independent Runs
<div align=center><img width = '500' height ='400' src ="https://github.com/starry-sky6688/StarCraft/blob/master/model/qmix/compare.png"/></div>

### 2. Win Rate of VDN in Two Independent Runs
<div align=center><img width = '500' height ='400' src ="https://github.com/starry-sky6688/StarCraft/blob/master/model/vdn/compare.png"/></div>
