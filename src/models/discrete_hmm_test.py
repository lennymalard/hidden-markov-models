import unittest
import numpy as np
import warnings
from src.models.DiscreteHMM import DiscreteHMM

class TestDiscreteHMM(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.num_states = 3
        self.num_obs_types = 5
        self.hmm = DiscreteHMM(self.num_states)
        self.hmm.num_obs = self.num_obs_types

        self.hmm.pi = np.array([1.0, 0.0, 0.0])
        self.hmm.A = np.array([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
        self.hmm.B = np.array([
            [0.8, 0.2, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.2, 0.8]
        ])

        self.obs = np.array([0, 2, 4, 0, 2, 4])
        self.hmm.seq_len = len(self.obs)

    def test_large_scale_numerical_stability(self):
        seq_len = 10000
        self.hmm.seq_len = seq_len
        obs_seq = np.random.randint(0, self.num_obs_types, size=seq_len)

        alpha, total_log_prob, scales = self.hmm.forward(obs_seq)

        self.assertFalse(np.isnan(total_log_prob))
        self.assertFalse(np.isinf(total_log_prob))

        for t in range(seq_len):
            self.assertAlmostEqual(np.sum(alpha[:, t]), 1.0, places=7)

        beta = self.hmm.backward(obs_seq, scales)
        self.assertFalse(np.isnan(beta).any())

    def test_forward_backward_consistency(self):
        alpha, total_log_prob, scales = self.hmm.forward(self.obs)
        beta = self.hmm.backward(self.obs, scales)

        for t in range(self.hmm.seq_len):
            prob_at_t = np.sum(alpha[:, t] * beta[:, t])
            self.assertGreater(prob_at_t, 0.0)

    def test_gamma_normalization(self):
        alpha, total_prob, scales = self.hmm.forward(self.obs)
        beta = self.hmm.backward(self.obs, scales)
        gamma, _ = self.hmm.compute_expected_counts(self.obs, alpha, beta)

        np.testing.assert_allclose(np.sum(gamma, axis=0), np.ones(self.hmm.seq_len), atol=1e-12)

    def test_xi_normalization(self):
        alpha, total_prob, scales = self.hmm.forward(self.obs)
        beta = self.hmm.backward(self.obs, scales)
        _, xi = self.hmm.compute_expected_counts(self.obs, alpha, beta)

        for t in range(self.hmm.seq_len - 1):
            self.assertAlmostEqual(np.sum(xi[:, :, t]), 1.0, places=12)

    def test_viterbi_deterministic_cycle(self):
        self.hmm.pi = np.array([1.0, 0.0, 0.0])
        self.hmm.A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        self.hmm.B = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0]
        ])

        obs_seq = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        self.hmm.seq_len = len(obs_seq)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predicted_states = self.hmm.decode(obs_seq)

        np.testing.assert_array_equal(predicted_states, obs_seq)

    def test_baum_welch_monotonic_convergence(self):
        obs_seq = np.random.randint(0, 5, size=500)
        self.hmm.seq_len = 500

        self.hmm.pi = np.random.dirichlet(np.ones(self.num_states))
        self.hmm.A = np.random.dirichlet(np.ones(self.num_states), size=self.num_states)
        self.hmm.B = np.random.dirichlet(np.ones(self.num_obs_types), size=self.num_states)

        previous_log_prob = -np.inf

        for iteration in range(15):
            alpha, current_log_prob, scales = self.hmm.forward(obs_seq)

            self.assertGreaterEqual(current_log_prob, previous_log_prob - 1e-7)
            previous_log_prob = current_log_prob

            beta = self.hmm.backward(obs_seq, scales)
            gamma, xi = self.hmm.compute_expected_counts(obs_seq, alpha, beta)

            for i in range(self.num_states):
                self.hmm.pi[i] = gamma[i, 0]
                for j in range(self.num_states):
                    self.hmm.A[i, j] = xi[i, j, :].sum() / gamma[i, :-1].sum()
                for k in range(self.num_obs_types):
                    self.hmm.B[i, k] = gamma[i, obs_seq == k].sum() / gamma[i, :].sum()

    def test_zero_probability_handling_in_decode(self):
        self.hmm.pi = np.array([1.0, 0.0, 0.0])
        self.hmm.A = np.array([
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        self.hmm.B = np.array([
            [0.8, 0.2, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0]
        ])

        obs_seq = np.array([0, 1, 1, 1])
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