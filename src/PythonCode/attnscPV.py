'''
Sum check that simulates sending back and forth
'''
# %%
import numpy as np
import random
from math import log2
# %%
# ---------------------------
# 1) Configuration: Prime Field
# ---------------------------
P = 2**61 - 1
# %%
# ---------------------------
# 2) Hypercube Enumerator
# ---------------------------
def binary_vectors_lsb(b):
    """Yield each binary vector of length b, LSB-first."""
    for i in range(2**b):
        yield [(i >> k) & 1 for k in range(b)]
# %%
# ---------------------------
# 3) Multilinear Extension
# ---------------------------
def multilinear_extension_int(M_int):
    N, D = M_int.shape
    b_p = int(log2(N))
    b_q = int(log2(D))
    row_bits = list(binary_vectors_lsb(b_p))
    col_bits = list(binary_vectors_lsb(b_q))

    def mle(s, t):
        total = 0
        for i, ri in enumerate(row_bits):
            Li = 1
            for k in range(b_p):
                term = ((1 - s[k]) * (1 - ri[k]) + s[k] * ri[k]) % P
                Li = (Li * term) % P
            for j, cj in enumerate(col_bits):
                Lj = 1
                for m in range(b_q):
                    term = ((1 - t[m]) * (1 - cj[m]) + t[m] * cj[m]) % P
                    Lj = (Lj * term) % P
                total = (total + M_int[i, j] * Li * Lj) % P
        return total

    return mle
# %%
# ---------------------------
# 4) Build Attention Polynomial
# ---------------------------
def build_attention_F_int(X_int, Wq_int, Wk_int, i, j):
    _, e = X_int.shape
    _, d = Wq_int.shape
    b_p = int(log2(e))
    b_q = int(log2(d))

    mle_Xi = multilinear_extension_int(X_int[i:i+1, :].T)
    mle_Wq = multilinear_extension_int(Wq_int)
    mle_Xj = multilinear_extension_int(X_int[j:j+1, :d])
    mle_Wk = multilinear_extension_int(Wk_int[:d, :])

    num_bits = b_p + 2 * b_q

    def F(z_bits):
        s = z_bits[0:b_p]
        t = z_bits[b_p:b_p + b_q]
        u = z_bits[b_p + b_q:]
        return (mle_Xi(s, []) * mle_Wq(s, t) * mle_Xj([], t) * mle_Wk(t, u)) % P

    return num_bits, F
# %%
# ---------------------------
# 5) Prover Class
# ---------------------------# 
class Prover:
    def __init__(self, X_int, Wq_int, Wk_int, i, j):
        self.X = X_int
        self.Wq = Wq_int
        self.Wk = Wk_int
        self.i = i
        self.j = j
        
        # 3A) Compute direct raw attention score S_direct = Σ_{p,q,h} X_i,p Wq[p,q] X_j,q Wk[q,h] mod P
        self.S_direct = 0
        e = X_int.shape[1]
        d = Wq_int.shape[1]
        for p in range(e):
            for q in range(d):
                for h in range(d):
                    self.S_direct = (
                        self.S_direct
                        + (self.X[i,p] * self.Wq[p,q] % P)
                        * (self.X[j,q] * self.Wk[q,h] % P)
                    ) % P

        # 3B) Build the sum-check polynomial F(z) and record its dimension m
        self.num_bits, self.F = build_attention_F_int(X_int, Wq_int, Wk_int, i, j)

    def initial_claim(self):
        """Prover sends the raw attention score to the verifier."""
        return self.S_direct

    def send_partial_sums(self, challenges, r):
        """
        In round r, prover computes h_r(0) and h_r(1):
        h_r(u) = Σ_{tail ∈ {0,1}^{m-r-1}} F(challenges + [u] + tail)
        """
        def h(u):
            total = 0
            for tail in binary_vectors_lsb(self.num_bits - (r + 1)):
                z = challenges + [u] + tail
                total = (total + self.F(z)) % P
            return total

        return h(0), h(1)

    def compute_claim_at(self, challenges, r, u):
        """Helper to compute h_r(u) for new challenge u."""
        total = 0
        for tail in binary_vectors_lsb(self.num_bits - (r + 1)):
            z = challenges + [u] + tail
            total = (total + self.F(z)) % P
        return total

    def final_evaluation(self, challenges):
        """Prover sends F(challenges) at the very end."""
        return self.F(challenges) % P
# %%
# ---------------------------
# 4) Simulate Interaction
# ---------------------------
def simulate_prover_verifier(X_int, Wq_int, Wk_int, i, j):
    random.seed(0)
    prover = Prover(X_int, Wq_int, Wk_int, i, j)

    # Verifier receives initial claim
    claim = prover.initial_claim()
    print("Verifier receives initial claim S_direct =", claim)

    challenges = []
    # m rounds of sum-check
    for r in range(prover.num_bits):
        # Prover sends h(0), h(1)
        h0, h1 = prover.send_partial_sums(challenges, r)
        print(f"Round {r} • Prover → Verifier: h(0)={h0}, h(1)={h1}")

        # Verifier checks h0+h1 == claim
        assert (h0 + h1) % P == claim, f"Round {r} mismatch"

        # Verifier chooses challenge and sends to prover
        r_r = random.randrange(P)
        print(f"Round {r} • Verifier → Prover: challenge r_r={r_r}")

        # Prover computes new claim = h(r_r)
        new_claim = prover.compute_claim_at(challenges, r, r_r)
        challenges.append(r_r)
        claim = new_claim
        print(f"Round {r} • Updated claim = {claim}")

    # Final step: Verifier asks for F(challenges)
    final_val = prover.final_evaluation(challenges)
    print("Verifier receives final F(challenges) =", final_val)

    # Final consistency check
    assert final_val == claim, "Final check mismatch"
    print("Verifier accepts: attention score verified")
# %%
# ---------------------------
# 5) Demo
# ---------------------------
if __name__ == "__main__":
    # Toy data as before
    X = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]],dtype=np.int64)
    Wq = np.array([[1,2],[3,4],[5,6],[7,8]],dtype=np.int64)
    Wk = np.array([[2,1],[4,3],[6,5],[8,7]],dtype=np.int64)

    X_int, Wq_int, Wk_int = X % P, Wq % P, Wk % P
    i, j = 1, 2

    print("▶ Simulating prover-verifier sum-check:")
    simulate_prover_verifier(X_int, Wq_int, Wk_int, i, j)