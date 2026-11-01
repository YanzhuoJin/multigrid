# Third-party code and scientific references

## Supplied MultiGrid environment and starter project

The supplied project is based on Natasha Jaques' MultiGrid starter repository:
https://github.com/natashamjaques/multigrid . Its MIT notice is retained in
`LICENSE` (Copyright 2024 Natasha Jaques). Several individual environment files
carry Google Research Authors' Apache-2.0 notices; these notices are preserved
in those files. MiniGrid/Gym, NumPy, PyTorch, Pillow, Matplotlib and imageio
remain external dependencies under their respective upstream licenses.

## Supplied PPO implementation and completed entry points

The supplied earlier PPO script follows the CleanRL implementation and links
its documentation. It is preserved in `legacy/ppo_original.py`. The active
`ppo.py` is a thin single-agent MultiGrid entry point into the same tested core
used by `main.py`. The completed trainer in `multiagent_metacontroller.py` and
`ippo_agent.py` adapts the supplied CleanRL-style clipped-PPO/GAE components to
independent multi-agent learning. Each agent has its own policy, critic,
optimizer, and rollout storage. The single-agent debugging experiment uses
this same core with one agent; it is not a separately implemented algorithm.

- Source: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
- Documentation: https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
- Upstream license: https://github.com/vwxyzjn/cleanrl/blob/master/LICENSE

CleanRL MIT notice (applicable to the supplied/adapted PPO implementation):

Copyright (c) 2019 CleanRL developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Algorithm references

- Schulman et al. (2017), *Proximal Policy Optimization Algorithms*.
  https://arxiv.org/abs/1707.06347
- Schulman et al. (2015), *High-Dimensional Continuous Control Using Generalized
  Advantage Estimation*. https://arxiv.org/abs/1506.02438
- de Witt et al. (2020), *Is Independent Learning All You Need in the StarCraft
  Multi-Agent Challenge?* https://arxiv.org/abs/2011.09533

The StarCraft paper motivates independent PPO as a multi-agent baseline;
its results are not evidence of performance in this MultiGrid task. Reported
MultiGrid results must come from the included local training/evaluation logs.
