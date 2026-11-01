# SocialRL 作业使用说明

代码、训练、评估、学习曲线和视频已经生成。现成模型可以直接使用，无需重新训练。

## 已完成的实验

- 主实验：三个完全独立的 PPO 智能体，固定 15×15 MultiGrid 环境。
- 实际完成 100,000 个完整联合回合，共 10,000,000 次环境交互。
- 训练用时约 13.29 分钟，训练种子为 1。
- 对每种策略使用 200 个未用于训练的评估种子；地图仍然是训练时的固定地图。
- 训练策略平均团队回报为 8.8624，随机策略为 0.0573。
- 三个智能体的评估成功率分别为 99.0%、99.5%、98.0%。成功指一回合内至少到达目标一次。

## 优先查看这些文件

1. `REPORT.md`：英文实验报告，包含方法、参数、结果、局限和引用。
2. `results/ippo_seed1/figures/learning_curve.png`：完整 100,000 回合学习曲线。
3. `results/ippo_seed1/evaluation/trajectory_stochastic_seed1000000.mp4`：训练策略的视频，包含全局地图、各智能体视野、动作和累计回报。
4. `results/ippo_seed1/evaluation/summary.json`：完整评估数值。
5. `results/ippo_seed1/checkpoints/final.pt`：训练好的三个模型和训练状态。
6. `results/ippo_seed1/submission_audit.json`：提交材料一致性检查结果。

视频使用事先固定的第一个评估回合，没有挑选最高分回合。报告中的置信区间反映同一个模型在不同评估回合中的变化，不代表多次独立训练的误差。

## 在这台 Mac 上运行

终端中输入：

```bash
cd /Users/yanzhuojin/Desktop/socialrl/multigrid
source ../.venv-socialrl/bin/activate
```

重新评估已有模型：

```bash
python evaluate.py --checkpoint results/ippo_seed1/checkpoints/final.pt --episodes 200 --seed 1000000 --output results/ippo_seed1/evaluation
```

完整重新训练时使用新目录，避免覆盖现有结果：

```bash
python main.py --total-episodes 100000 --num-envs 64 --run-dir results/ippo_new --wandb-mode offline
```

中断后的恢复命令见 `README.md`。复制到其他电脑时，按 README 创建新的 Python 环境。

## W&B 账号上传

训练日志已经上传并独立核验。[查看 W&B 在线训练曲线](https://forge.coreweave.com/wandb/311yanzhuo-jin299-medical-college-of-wisconsin/socialrl-ippo/runs/nksxtadt)。云端 1,000 个曲线点覆盖到第 100,000 回合，与本地日志对应的 100 回合平均回报一致。验证记录保存在 `results/ippo_seed1/wandb_sync_status.json`。

需要重新登录或同步其他离线实验时：

```bash
python scripts/sync_wandb.py
```

它会调用 W&B 的官方登录流程，然后同步保存的实验。所需 API key 属于 **W&B 账户**，用于实验日志上传；不能使用 OpenAI API key。训练本身不调用 OpenAI API。不要把 API key 放入作业代码、报告或提交压缩包。

同步脚本会将电脑已有的系统代理设置传给 W&B 后台进程，解决 macOS 上浏览器能连接、后台进程却超时的情况；不会修改系统网络设置。

## 备份和提交

原项目源代码与已有日志备份在上一级目录 `backups/original_project_before_completion.zip`，原 Windows 虚拟环境也保留。原单智能体脚本另存于 `legacy/ppo_original.py`。

`socialrl_submission.zip` 是本次代码与实验结果的提交包，位于桌面 `socialrl` 目录。GitHub fork 已发布：[查看完整仓库](https://github.com/YanzhuoJin/multigrid)，默认分支为 `socialrl-complete`。代码、模型、报告和视频已上传并核对 Git 内容哈希，验证记录见 `GITHUB_SUBMISSION.json`。

原上游仓库的 Git 历史保留；本地 `upstream` 指向原仓库，`origin` 指向你的 fork。
