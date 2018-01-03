# -*- coding: utf-8 -*-

import numpy as np


class HMM(object):
    """
    一阶隐马尔科夫模型
    A: 状态转移概率矩阵
    B: 生成（发射）概率矩阵
    PI: 初始状态概率
    """
    def __init__(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi

    """
    前向算法（求解问题1）：在给定模型mu(A,B,pi)和观察序列O=O_1,...,O_T 的情况下，怎样快速计算p(O|mu) ?
    (1)初始化:alpha_1(i)=pi_i b_i(O_1),1≤i ≤ N
    (2)计算alpha_t+1(j)=[sum_N alpha_t(i) a_ij] b_j(O_t+1),1≤t≤T−1 i=1
    (3)结束，输出 P(O|mu)=sum_N alpha_T(i)
    """
    def _forward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)
        F = np.zeros((N, T))
        F[:, 0] = self.pi * self.B[:, obs_seq[0]]

        for t in range(1, T):
            for n in range(N):
                F[n, t] = np.dot(F[:, t - 1], (self.A[:, n])) * self.B[n, obs_seq[t]]
        return F

    """
    后向算法（求解问题1）：在给定模型mu(A,B,pi)和观察序列O=O_1,...,O_T 的情况下，怎样快速计算p(O|mu) ?
    (1)初始化:beta_T(i)=1,1≤i≤N j=1
    (2)计算beta_t(i) = sum_N a_ij b_j(O_t+1)β_t+1(j) , T − 1 ≥ t ≥ 1, 1 ≤ i ≤ N
    (3)结束，输出 P(O|mu) = sum_N pi_i beta_1(i)    
    """
    def _backward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)
        X = np.zeros((N, T))
        X[:, -1:] = 1

        for t in reversed(range(T - 1)):
            for n in range(N):
                X[n, t] = np.sum(X[:, t + 1] * self.A[n, :] * self.B[:, obs_seq[t + 1]])
        return X

    # 输出 P(O|mu)
    def observation_prob(self, obs_seq):
        """ P( entire observation sequence | A, B, pi ) """
        return np.sum(self._forward(obs_seq)[:, -1])

    # 输出最优的状态路径
    def state_path(self, obs_seq):
        V, prev = self.viterbi(obs_seq)

        # Build state path with greatest probability
        last_state = np.argmax(V[:, -1])
        path = list(self.build_viterbi_path(prev, last_state))

        return V[last_state, -1], reversed(path)

    """
    viterbi算法（求解问题2）:在给定模型mu(A,B,pi)和观察序列O=O_1,...,O_T 的情况下，如何选择在一定意义
                           下“最优”的状态序 Q = q_1,...,q_T，使得该状态序列“最好地解释”观察序列.
    (1) 初始化：delta_1(i)=pi_i bi(O_1),1≤i ≤ N                       
    (2) 递归计算 delta_t+1(i)=[max_j delta_t(j) a_ji] b_i(O_t+1)
    """
    def viterbi(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)
        prev = np.zeros((T - 1, N), dtype=int)

        # DP matrix containing max likelihood of state at a given time
        V = np.zeros((N, T))
        V[:, 0] = self.pi * self.B[:, obs_seq[0]]

        for t in range(1, T):
            for n in range(N):
                seq_probs = V[:, t - 1] * self.A[:, n] * self.B[n, obs_seq[t]]
                prev[t - 1, n] = np.argmax(seq_probs)
                V[n, t] = np.max(seq_probs)

        return V, prev

    # 回溯得到最优路径
    def build_viterbi_path(self, prev, last_state):
        """Returns a state path ending in last_state in reverse order."""
        T = len(prev)
        yield (last_state)
        for i in range(T - 1, -1, -1):
            yield (prev[i, last_state])
            last_state = prev[i, last_state]

    """
    baum welch 算法（问题3）：训练模型参数
    用EM算法更新参数
    """
    def baum_welch_train(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)

        forw = self._forward(obs_seq)
        back = self._backward(obs_seq)

        # P(entire observation sequence | A, B, pi )
        obs_prob = np.sum(forw[:, -1])
        if obs_prob <= 0:
            raise ValueError("P(O | lambda) = 0. Cannot optimize!")

        xi = np.zeros((T - 1, N, N))
        for t in range(xi.shape[0]):
            xi[t, :, :] = self.A * forw[:, [t]] * self.B[:, obs_seq[t + 1]] * back[:, t + 1] / obs_prob

        gamma = forw * back / obs_prob

        # Gamma sum excluding last column
        gamma_sum_A = np.sum(gamma[:, :-1], axis=1, keepdims=True)
        # Vector of binary values indicating whether a row in gamma_sum is 0.
        # If a gamma_sum row is 0, save old rows on update
        rows_to_keep_A = (gamma_sum_A == 0)
        # Convert all 0s to 1s to avoid division by zero
        gamma_sum_A[gamma_sum_A == 0] = 1.
        next_A = np.sum(xi, axis=0) / gamma_sum_A

        gamma_sum_B = np.sum(gamma, axis=1, keepdims=True)
        rows_to_keep_B = (gamma_sum_B == 0)
        gamma_sum_B[gamma_sum_B == 0] = 1.

        # 克罗奈克(Kronecker)函数
        obs_mat = np.zeros((T, self.B.shape[1]))
        obs_mat[range(T), obs_seq] = 1
        next_B = np.dot(gamma, obs_mat) / gamma_sum_B

        # Update model
        self.A = self.A * rows_to_keep_A + next_A
        self.B = self.B * rows_to_keep_B + next_B
        self.pi = gamma[:, 0] / np.sum(gamma[:, 0])
