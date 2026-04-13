import numpy as np

class DiscreteHMM:
    def __init__(self, num_states: int):
        self.num_states  = num_states # N
        self.num_obs = None # M
        self.seq_len = None
        self.alpha = None
        self.beta = None
        self.A = None
        self.B = None
        self.pi = None

    def forward(self, obs):
        alpha = np.zeros((self.num_states, self.seq_len))
        scales = np.zeros(self.seq_len)

        alpha[:, 0] = self.pi * self.B[:, obs[0]]
        scales[0] = np.sum(alpha[:, 0]) + 1e-12
        alpha[:, 0] /= scales[0]

        for t in range(1, self.seq_len):
            alpha[:, t] = alpha[:, t-1] @ self.A * self.B[:, obs[t]]
            scales[t] = np.sum(alpha[:, t]) + 1e-12
            alpha[:, t] /= scales[t]

        total_log_prob = np.sum(np.log(scales))
        return alpha, total_log_prob, scales

    def backward(self, obs, scales):
        beta = np.zeros((self.num_states, self.seq_len))

        beta[:, -1] = 1.0
        beta[:, -1] /= scales[-1]

        for t in range(self.seq_len-2, -1, -1):
            beta[:, t] = self.A @ (self.B[:, obs[t+1]] * beta[:, t+1])
            beta[:, t] /= scales[t]

        return beta

    def compute_expected_counts(self, obs, alpha, beta):
        xi = np.zeros((self.num_states, self.num_states, self.seq_len))

        gamma = alpha * beta
        gamma /= gamma.sum(axis=0, keepdims=True)

        for t in range(self.seq_len - 1):
            numerator = (alpha[:, t].reshape(-1, 1) * self.A * self.B[:, obs[t+1]] * beta[:, t+1])
            xi[:, :, t] = numerator / np.sum(numerator)

        return gamma, xi

    def decode(self, obs):
        seq_len = len(obs)
        delta = np.zeros((self.num_states, seq_len))
        psi = np.zeros((self.num_states, seq_len))

        log_A = np.log(self.A)
        log_B = np.log(self.B)
        log_pi = np.log(self.pi)

        for i in range(self.num_states):
            delta[i, 0] = log_pi[i] + log_B[i, obs[0]]

        for t in range(1, seq_len):
            for j in range(self.num_states):
                prob = delta[:, t-1] + log_A[:, j]
                delta[j, t] = np.max(prob) + log_B[j, obs[t]]
                psi[j, t] = np.argmax(prob)

        states = np.zeros(seq_len, dtype=np.uint8)
        states[seq_len-1] = np.argmax(delta[:, seq_len-1])

        for t in range(seq_len-2, -1, -1):
            states[t] = psi[states[t+1], t+1]

        return states

    def fit(self, obs, num_iter):
        obs = np.array(obs)
        self.num_obs = len(np.unique(obs))
        self.seq_len = len(obs)

        self.A = np.random.rand(self.num_states, self.num_states) + 1e-12
        self.A /= self.A.sum(axis=1, keepdims=True)

        self.B = np.random.rand(self.num_states, self.num_obs) + 1e-12
        self.B /= self.B.sum(axis=1, keepdims=True)

        self.pi = np.random.rand(self.num_states)
        self.pi /= self.pi.sum()

        for iter in range(num_iter):
            alpha, total_log_prob, scales= self.forward(obs)
            beta = self.backward(obs, scales)
            gamma, xi = self.compute_expected_counts(obs, alpha, beta)

            self.pi = gamma[:, 0]
            self.A = xi.sum(axis=2) / gamma[:, :-1].sum(axis=1).reshape(-1, 1)

            for k in range(self.num_obs):
                self.B[:, k] = gamma[:, obs==k].sum(axis=1) / gamma.sum(axis=1)
