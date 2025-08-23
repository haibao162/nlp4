import numpy as np
import matplotlib.pyplot as plt
import torch

# 假设汽车从出发到达目的地预测要60分钟，开了10分钟后，预测还需要45分钟，则10+45认为是实际值，即时奖励是10，初始状态价值是60，下一状态的价值是45分钟
# 60 和55的差值就是损失函数，而TD时序差分就是奖励r_k + 折扣因子 * V_s+1 - V_s

class BernoulliBandit:
    """ 伯努利多臂老虎机,输入K表示拉杆个数 """
    def __init__(self, K):
        self.probs = np.random.uniform(size=K) # 随机生成K个0～1的数,作为拉动每根拉杆的获奖
        self.best_idx = np.argmax(self.probs) # 获奖概率最大的拉杆
        self.best_prob = self.probs[self.best_idx] # 最大的获奖概率
        self.K = K
    
    def step(self, k):
        # 当玩家选择了k号拉杆后,根据拉动该老虎机的k号拉杆获得奖励的概率返回1（获奖）或0（未
        # 获奖）
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


np.random.seed(1)  # 设定随机种子,使实验具有可重复性
K = 10
bandit_10_arm = BernoulliBandit(K)
# print(bandit_10_arm.probs, 'bandit_10_arm')

# random_numbers = torch.rand(10)
# print(random_numbers, 'random_numbers')

print("随机生成了一个%d臂伯努利老虎机" % K)
print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" %
      (bandit_10_arm.best_idx, bandit_10_arm.best_prob))

class Solver:
    """ 多臂老虎机算法基本框架 """
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K) # 每根拉杆的尝试次数
        self.regret = 0
        self.actions = [] # 维护一个列表,记录每一步的动作
        self.regrets = [] # 维护一个列表,记录每一步的累积懊悔
    
    def update_regret(self, k):
        # 计算累积懊悔并保存,k为本次动作选择的拉杆的编号
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        # 返回当前动作选择哪一根拉杆,由每个具体的策略实现
        raise NotImplementedError
    
    def run(self, num_steps):
        # 运行一定次数,num_steps为总运行次数
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)