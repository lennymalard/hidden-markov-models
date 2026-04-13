import unittest
import numpy as np
import warnings
from src.models.ContinuousHMM import ContinuousHMM
from src.utils.math import logsumexp

class TestContinuousHMM(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.num_states = 3
        self.num_feats = 1
        self.hmm = ContinuousHMM(self.num_states)
        self.hmm.num_feats = self.num_feats

        self.hmm.pi = np.array([1.0, 0.0, 0.0])
        self.hmm.A = np.array([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
        self.hmm.mu = np.array([[0.0], [10.0], [20.0]])
        self.hmm.sigma = np.array([[[1.0]], [[1.0]], [[1.0]]])

        self.obs = np.array([[0.1], [10.2], [19.8], [0.2], [9.9], [20.1]])
        self.hmm.seq_len = len(self.obs)

    def test_large_scale_numerical_stability(self):
        seq_len = 1000
        self.hmm.seq_len = seq_len
        obs_seq = np.random.normal(10, 5, size=(seq_len, 1))

        B = self.hmm.compute_emissions(obs_seq)
        alpha, total_log_prob = self.hmm.forward(B)

        self.assertFalse(np.isnan(total_log_prob))
        self.assertFalse(np.isinf(total_log_prob))

        beta = self.hmm.backward(B)
        self.assertFalse(np.isnan(beta).any())

    def test_forward_backward_consistency(self):
        B = self.hmm.compute_emissions(self.obs)
        alpha, total_log_prob_alpha = self.hmm.forward(B)
        beta = self.hmm.backward(B)

        for t in range(self.hmm.seq_len):
            prob_at_t = logsumexp(alpha[:, t] + beta[:, t])
            self.assertAlmostEqual(prob_at_t, total_log_prob_alpha, places=10)

    def test_gamma_normalization(self):
        B = self.hmm.compute_emissions(self.obs)
        alpha, total_prob = self.hmm.forward(B)
        beta = self.hmm.backward(B)
        gamma, _ = self.hmm.compute_expected_counts(B, alpha, beta, total_prob)

        np.testing.assert_allclose(np.sum(gamma, axis=0), np.ones(self.hmm.seq_len), atol=1e-12)

    def test_xi_normalization(self):
        B = self.hmm.compute_emissions(self.obs)
        alpha, total_prob = self.hmm.forward(B)
        beta = self.hmm.backward(B)
        _, xi = self.hmm.compute_expected_counts(B, alpha, beta, total_prob)

        for t in range(self.hmm.seq_len - 1):
            self.assertAlmostEqual(np.sum(xi[:, :, t]), 1.0, places=12)

    def test_viterbi_deterministic_cycle(self):
        self.hmm.pi = np.array([1.0, 0.0, 0.0])
        self.hmm.A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        self.hmm.mu = np.array([[0.0], [100.0], [200.0]])
        self.hmm.sigma = np.array([[[0.01]], [[0.01]], [[0.01]]])

        obs_seq = np.array([[0.0], [100.0], [200.0], [0.0], [100.0], [200.0], [0.0], [100.0], [200.0]])
        self.hmm.seq_len = len(obs_seq)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predicted_states = self.hmm.decode(obs_seq)

        expected_path = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        np.testing.assert_array_equal(predicted_states, expected_path)

    def test_baum_welch_monotonic_convergence(self):
        obs_seq = np.vstack([np.random.normal(0, 1, 50).reshape(-1, 1), np.random.normal(10, 1, 50).reshape(-1, 1)])
        self.hmm.seq_len = len(obs_seq)

        previous_log_prob = -np.inf

        for iteration in range(5):
            B = self.hmm.compute_emissions(obs_seq)
            alpha, current_log_prob = self.hmm.forward(B)

            self.assertGreaterEqual(current_log_prob, previous_log_prob - 1e-7)
            previous_log_prob = current_log_prob

            beta = self.hmm.backward(B)
            gamma, xi = self.hmm.compute_expected_counts(B, alpha, beta, current_log_prob)

            self.hmm.pi = gamma[:, 0]
            self.hmm.A = np.sum(xi, axis=2) / np.sum(gamma[:, :-1], axis=1, keepdims=True)

    def test_zero_probability_handling_in_decode(self):
        self.hmm.pi = np.array([1.0, 0.0, 0.0])
        self.hmm.A = np.array([[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        obs_seq = np.array([[0.0], [10.0], [10.0], [10.0]])
        self.hmm.seq_len = len(obs_seq)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                states = self.hmm.decode(obs_seq)
                self.assertEqual(len(states), len(obs_seq))
            except Exception as e:
                self.fail(e)

if __name__ == '__main__':
    unittest.main()