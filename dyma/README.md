# DyMA

This is a pytorch implementation of the multi-agent reinforcement learning algorithm, [DyMA-CL](https://arxiv.org/abs/1909.02790?context=cs.MA)

## Note

### Action Space

Beacuse  previous StarCraft II settings enable an agent to attack one of its enemies by choosing one of id numbers, 
the action spaces of different maps are not the same, so in the original paper, the author design the attack
action is to choose one of the grid units by dividing the battlefield into several grids.

We don't want to waste time modifying environment code, so we just run the algorithm on map '8m' and '3m', and for '3m', 
we pad its action space with 5 zeros to make sure it has the same action space dimension as '8m', but we forbid agents to choose these 5 actions.

### Why from '8m' to '3m'

Because we use zero-padding, there is a problem for transferring model from '3m' to '8m'. In '3m', the last 5 actions have never
be chosen, so their Q-value is far from true value, when we use the model of '3m' on '8m', we find the performance of transferring is negative.
Therefore, we transfer the model from '8m' to '3m' to avoid this problem. 
We first train the model on '8m' for a while, and then transfer the model to '3m', and we compare the result between transferring and not transferring.



## Quick Start

```shell
$ python main.py
```

Directly run the main.py, then the algorithm will start training on map '3m', using the pretrained model on map '8m'. If you
want to train from scratch, you should use it to train on '8m', then move the model of '8m' to `./model/dyma/3m/` and rename the model.

## Result

We independently train DyMa for 8 times and take the mean of the 8 independent results.
### 1. Mean Win Rate of 8 Independent Runs on '3m'
<div align=center><img width = '600' height ='300' src ="https://github.com/starry-sky6688/StarCraft/blob/master/dyma/result/compare.png"/></div>