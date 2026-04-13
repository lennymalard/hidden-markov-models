import numpy as np
from src.utils.math import logsumexp

# TODO Add scoring
# TODO Add a method to continue iterations until convergence

class ContinuousHMM:
    def __init__(self, num_states: int):
        self.num_states  = num_states # N
        self.num_obs = None # M
        self.num_feats = None # k
        self.seq_len = None
        self.alpha = None
        self.beta = None
        self.A = None
        self.B = None
        self.mu = None
        self.sigma  = None
        self.pi = None

    def gaussian_pdf(self, x, mu, sigma):
        diff = x - mu
        L = np.linalg.cholesky(sigma)
        y = np.linalg.solve(L, diff)
        mahalanobis = y.T @ y
        log_det = 2 * np.sum(np.log(np.diag(L)))
        return -0.5 * (self.num_feats * np.log(2 * np.pi) + log_det + mahalanobis)

    def compute_emissions(self, obs):
        L = np.linalg.cholesky(self.sigma)
        diff = obs[:, np.newaxis, :] - self.mu[np.newaxis, :, :]
        y = np.linalg.solve(L, diff[..., np.newaxis])
        y = np.squeeze(y, axis=-1)
        mahalanobis = np.sum(y**2, axis=-1)
        log_det = 2 * np.sum(np.log(np.diagonal(L, axis1=-2, axis2=-1)), axis=-1)
        const = self.num_feats * np.log(2 * np.pi)
        B = -0.5 * (const + log_det[np.newaxis, :] + mahalanobis)
        return B.T

    def forward(self, B):
        log_A = np.log(self.A + 1e-12)
        log_pi = np.log(self.pi + 1e-12)

        alpha = np.zeros((self.num_states, self.seq_len))

        alpha[:, 0] = log_pi + B[:, 0]

        for t in range(1, self.seq_len):
            alpha[:, t] = logsumexp(alpha[:, t-1][:, np.newaxis] + log_A, axis=0) + B[:, t]

        total_prob = logsumexp(alpha[:, -1])
        return alpha, total_prob

    def backward(self, B):
        log_A = np.log(self.A + 1e-12)
        beta = np.zeros((self.num_states, self.seq_len))

        beta[:, -1] = 0.0

        for t in range(self.seq_len - 2, -1, -1):
            beta[:, t] = logsumexp(log_A + (B[:, t+1] + beta[:, t+1]), axis=1)

        return beta

    def compute_expected_counts(self, B, alpha, beta, total_prob):
        log_A = np.log(self.A + 1e-12)
        xi = np.zeros((self.num_states, self.num_states, self.seq_len-1))

        gamma = alpha + beta - total_prob

        for t in range(self.seq_len - 1):
            xi[:, :, t] = (alpha[:, t][:, np.newaxis] + log_A + B[:, t+1] + beta[:, t+1]) - total_prob

        return np.exp(gamma), np.exp(xi)

    def decode(self, obs):
        obs = np.array(obs)

        self.seq_len = len(obs)
        B = self.compute_emissions(obs)

        log_A = np.log(self.A + 1e-12)
        log_pi = np.log(self.pi + 1e-12)

        delta = np.zeros((self.num_states, self.seq_len))
        psi = np.zeros((self.num_states, self.seq_len), dtype=int)

        delta[:, 0] = log_pi + B[:, 0]

        for t in range(1, self.seq_len):
            delta[:, t] = np.max(delta[:, t-1][:, np.newaxis] + log_A, axis=0) + B[:, t]
            psi[:, t] = np.argmax(delta[:, t-1][:, np.newaxis] + log_A, axis=0)

        states = np.zeros(self.seq_len, dtype=int)
        states[-1] = np.argmax(delta[:, -1])

        for t in range(self.seq_len - 2, -1, -1):
            states[t] = psi[states[t+1], t+1]

        return states

    def fit(self, obs, num_iter, init_params=True):
        obs = np.array(obs)
        self.seq_len, self.num_feats = obs.shape

        if init_params:
            self.A = np.random.rand(self.num_states, self.num_states)
            self.A /= self.A.sum(axis=1, keepdims=True)
            self.pi = np.random.rand(self.num_states)
            self.pi /= self.pi.sum()
            # Initialize mu with distinct observations to break symmetry
            indices = np.random.choice(self.seq_len, self.num_states, replace=False)
            self.mu = obs[indices].astype(float)

            diff_global = obs - np.mean(obs, axis=0)
            global_sigma = (diff_global.T @ diff_global) / self.seq_len + np.eye(self.num_feats) * 1e-6
            self.sigma = np.array([global_sigma.copy() for _ in range(self.num_states)])

        for iter in range(num_iter):

            B = self.compute_emissions(obs)

            alpha, total_prob = self.forward(B)
            beta = self.backward(B)
            gamma, xi = self.compute_expected_counts(B, alpha, beta, total_prob)

            self.A = xi.sum(axis=2) / (gamma[:, :-1].sum(axis=1)[:, np.newaxis] + 1e-12)
            self.pi = gamma[:, 0] / (np.sum(gamma[:, 0]) + 1e-12)

            for i in range(self.num_states):
                gamma_i = gamma[i, :]
                gamma_i_sum = np.sum(gamma_i) + 1e-12
                self.mu[i] = (gamma_i @ obs) / gamma_i_sum

                diff = obs - self.mu[i]
                self.sigma[i] = (diff.T @ (gamma_i[:, np.newaxis] * diff)) / gamma_i_sum
                self.sigma[i] += np.eye(self.num_feats) * 1e-6
