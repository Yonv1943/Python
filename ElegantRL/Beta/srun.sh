Evolution Strategies 演化策略

ES算法的内容很少，可概括成：
theta 是策略网络的所有参数
现在，用 theta + gaussian noise 在参数空间做小扰动，让它能表示一堆相似策略的集合

我从这个集合中抽样得到足够多个新的策略，然后放在env中评估它的分数。

新的策略中分数有高有低，我让策略的集合朝着能抽样得到“高分的策略”的方向优化。也就是让 theta 更靠近最优策略的参数。

论文内容很少，基本都在 2.1 Scaling and parallelizing ES
论文是：[Evolution Strategies as a Scalable Alternative to Reinforcement Learning 2017.09.](https://arxiv.org/pdf/1703.03864.pdf)

如果你嫌论文啰嗦，那么可以看网上这个人写的（他写的其他内容也不错）：
https://zhuanlan.zhihu.com/p/44629892


在论文中出现的关键词 reparameterization（重chong参数化）也许会让你疑惑，在SAC算法的这篇文章中有比较详细的讲解。 
[Soft Actor-Critic Algorithms and Applications 2019.01](https://arxiv.org/pdf/1812.05905.pdf)

看论文的公式（9），reparameterization


你也许能理解到：
- SAC PPO 等随机策略算法（stochastic policy），都是在动作空间action space 上，做重参数化 reparameterization
- Noisy DQN算法，就是在输出层上，离散动作的概率空间上，做重参数化
- ES算法，就是在整个策略参数空间上，做重参数化 




# http://hpc.pku.edu.cn/_book/guide/slurm/salloc.html?q=
salloc -p C032M0128G -N2 --ntasks-per-node=12 -q low -t 2:00:00
# salloc 申请成功后会返回申请到的节点和作业ID等信息，假设申请到的是a8u03n[05-06]节点，作业ID为1078858
# 这里申请两个节点，每个节点12个进程，每个进程一个核心

# 根据需求导入MPI环境
module load intel/2018.0

# 根据以下命令生成MPI需要的machine file
# srun hostname -s | sort -n > slurm.hosts
# https://blog.csdn.net/wjn922/article/details/103125435/
srun --partition=XXX
     --mpi=pmi2
     --gres=gpu:8
     -n1
     --ntasks-per-node=1
     --job-name=TEST
     --kill-on-bad-exit=1
     python3 beta2.py 


# 跳转到申请到的头节点(第一个节点)，如a8u03n05
ssh a8u03n05

# 导入计算环境并执行程序
module load intel/2018.0
mpirun -np 24 -machinefile slurm.hosts hostname

# 结束后退出或者结束任务
scancel 1078858