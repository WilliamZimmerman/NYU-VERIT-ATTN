#%%
import numpy as np
import random
from math import log2
random.seed(42)    
np.random.seed(42)
P = 2**61 - 1
print(P)
# Binary Vector for indices
# %%
def binary_vectors_lsb(b):
    """
    Yield each binary vector of length b, where
    vec[0] is the least-significant bit and vec[b-1] the most.
    """
    for i in range(2**b):
        # (i >> 0) & 1  is the LSB, (i >> (b-1)) & 1 is the MSB
        yield [(i >> k) & 1 for k in range(b)]
# %%
def multilinear_extension(M):
    
    N, D = M.shape
    b_p = int(np.log2(N))
    b_q = int(np.log2(D))
    assert 2**b_p == N and 2**b_q == D, "Matrix dims must be powers of 2"

    row_bits = list(binary_vectors_lsb(b_p))
    col_bits = list(binary_vectors_lsb(b_q))

    def mle(s, t):
        total = 0.0
        for i, ri in enumerate(row_bits):
            Li = np.prod([(1 - s[k])*(1 - ri[k]) + s[k]*ri[k] for k in range(b_p)]) if b_p > 0 else 1.0
            for j, cj in enumerate(col_bits):
                Lj = np.prod([(1 - t[m])*(1 - cj[m]) + t[m]*cj[m] for m in range(b_q)]) if b_q > 0 else 1.0
                total += M[i, j] * Li * Lj
        return total

    return mle
# %%
def multilinear_extension_int(M_int):

    N, D = M_int.shape
    b_p = int(log2(N))
    b_q = int(log2(D))
    assert 2**b_p == N and 2**b_q == D, "Matrix dims must be powers of two"

    # Precompute bit‐decompositions of all row and column indices
    row_bits = list(binary_vectors_lsb(b_p))
    col_bits = list(binary_vectors_lsb(b_q))

    def mle(s, t):
        total = 0
        # Loop over all entries
        for i, ri in enumerate(row_bits):
            # Compute L_i(s) in F_P
            Li = 1
            for k in range(b_p):
                term = ((1 - s[k]) * (1 - ri[k]) + s[k] * ri[k]) % P
                Li = (Li * term) % P

            for j, cj in enumerate(col_bits):
                # Compute L_j(t) in F_P
                Lj = 1
                for m in range(b_q):
                    term = ((1 - t[m]) * (1 - cj[m]) + t[m] * cj[m]) % P
                    Lj = (Lj * term) % P

                # Accumulate M_int[i,j] * Li * Lj
                total = (total + M_int[i, j] * Li * Lj) % P

        return total

    return mle

# %%
def build_attention_F(X_int, Wq_int, Wk_int, i, j):
    """
    Build the triple‐sum polynomial F for
        S_{i,j} = Σ_{p,q,h} X_int[i,p]*Wq_int[p,q]*X_int[j,q]*Wk_int[q,h]  (mod P).
    X_int: (n × e), Wq_int: (e × d), Wk_int: (e × d), all ints mod P.
    i, j: token positions in [0,n).
    Returns:
      num_bits = b_p + 2*b_q,
      F(z_bits): function on {0,1}^num_bits → F_P
    """
    # Dimensions
    _, e = X_int.shape
    _, d = Wq_int.shape
    b_p = int(log2(e))
    b_q = int(log2(d))

    # Build MLEs for each slice
    # 1) X_i row: shape (e × 1) → row‐bits b_p, col‐bits 0
    mle_Xi = multilinear_extension_int(X_int[i:i+1, :].T)
    # 2) Wq: shape (e × d) → row‐bits b_p, col‐bits b_q
    mle_Wq = multilinear_extension_int(Wq_int)
    # 3) X_j restricted to first d columns: shape (1 × d) → row‐bits 0, col‐bits b_q
    mle_Xj = multilinear_extension_int(X_int[j:j+1, :d])
    # 4) Wk restricted to first d rows: shape (d × d) → row‐bits b_q, col‐bits b_q
    mle_Wk = multilinear_extension_int(Wk_int[:d, :])

    num_bits = b_p + 2*b_q

    def F(z_bits):
        # Split z into (s, t, u)
        s = z_bits[0 : b_p]
        t = z_bits[b_p : b_p + b_q]
        u = z_bits[b_p + b_q : ]
        # Evaluate the four MLEs and multiply (mod P)
        return (
            mle_Xi(s, [])     # uses p-index bits
            * mle_Wq(s, t)    # uses p,q-index bits
            * mle_Xj([], t)   # uses     q-index bits
            * mle_Wk(t, u)    # uses q,h-index bits
        ) % P

    return num_bits, F
# %%
def sumcheck(f, m, cheat_round = -1):
    # 1) INITIAL CLAIM: exact sum over 2^m points, mod P
    claim = 0
    for z in binary_vectors_lsb(m):
        claim = (claim + f(z)) % P

    challenges = []

    # 2) m rounds: peel off one bit each time
    for r in range(m):
        # Build the univariate “partial‐sum” polynomial h_r(u)
        def h(u):
            total = 0
            # Sum over all remaining m−(r+1) bits (the “tails”)
            for tail in binary_vectors_lsb(m - (r+1)):
                z = challenges + [u] + tail
                total = (total + f(z)) % P
            return total

        # Prover “sends” h(0) and h(1)
        h0 = h(0)
        h1 = h(1)

        if r == cheat_round:
            h0 = (h0 + 1) % P

        # Verifier checks
        if (h0 + h1) % P != claim:
            print(f"❌ Cheating detected at round {r}: "
                  f"reported h0+h1={(h0+h1)%P}, expected {claim}")
            return False
        # Verifier samples a fresh random challenge in F_P
        r_r = random.randrange(P)

        new_claim = h(r_r) % P

        challenges.append(r_r)

        claim = new_claim

    # 3) FINAL CHECK: one‐point evaluation
    final = f(challenges) % P
    if final != claim:
        print(f"Final check failed: f(r)={final} vs claim={claim}")
        return False

    return True
# %%
def test_binary_vectors():
    # For b=3, expect vectors from 000 to 111 LSB-first:
    expected = [
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [1,1,0],
        [0,0,1],
        [1,0,1],
        [0,1,1],
        [1,1,1]
    ]
    result = list(binary_vectors_lsb(3))
    assert result == expected, f"binary_vectors_lsb failed: {result} != {expected}"

def test_mle():
    # Small 4x4 integer matrix mod P
    M = np.array([[1,2,3,4],
                  [5,6,7,8],
                  [9,10,11,12],
                  [13,14,15,16]], dtype=np.int64) % P
    mle = multilinear_extension(M)
    b1 = 2; b2 = 2
    for i in range(4):
        s = [(i >> k) & 1 for k in range(b1)]
        for j in range(4):
            t = [(j >> k) & 1 for k in range(b2)]
            assert mle(s, t) == M[i, j], f"MLE mismatch at ({i},{j}): {mle(s,t)} != {M[i,j]}"


def test_build_attention_F():
    import numpy as np

    # ─── Toy data ─────────────────────────────────────────
    # Small token‐embeddings X (n×e), and weight matrices Wq, Wk (e×d)
    X = np.array([
        [1,  2,  3,  4],
        [5,  6,  7,  8],
        [9, 10, 11, 12],
    ], dtype=np.int64)
    Wq = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
    ], dtype=np.int64)  # e=4, d=2
    Wk = np.array([
        [2, 1],
        [4, 3],
        [6, 5],
        [8, 7],
    ], dtype=np.int64)

    # Reduce into F_P
    X_int  = X  % P
    Wq_int = Wq % P
    Wk_int = Wk % P

    # Choose a sample (i,j)
    i, j = 1, 2

    # ─── Build the polynomial F ───────────────────────────
    num_bits, F = build_attention_F(X_int, Wq_int, Wk_int, i, j)

    # ─── Compute the direct triple‐sum S_{i,j} mod P ───────
    S = 0
    e = X_int.shape[1]
    d = Wq_int.shape[1]
    for p in range(e):
        for q in range(d):
            for h in range(d):
                # accumulate each product, reducing mod P
                S = (S
                     + (X_int[i,p] * Wq_int[p,q] % P)
                     * (X_int[j,q] * Wk_int[q,h] % P)
                    ) % P

    # ─── Compute the hypercube sum via F ───────────────────
    S_F = 0
    for z in binary_vectors_lsb(num_bits):
        S_F = (S_F + F(z)) % P

    # ─── Assert they match exactly ─────────────────────────
    assert S_F == S, f"Triple-sum mismatch: hypercube {S_F} != direct {S}"
    print("test_build_attention_F_int passed!")
# %%
def test_sumcheck():
    # ─── Toy data ─────────────────────────────────────────
    X = np.array([
        [1,  2,  3,  4],
        [5,  6,  7,  8],
        [9, 10, 11, 12],
    ], dtype=np.int64)
    Wq = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
    ], dtype=np.int64)  # e=4, d=2
    Wk = np.array([
        [2, 1],
        [4, 3],
        [6, 5],
        [8, 7],
    ], dtype=np.int64)

    # ─── Reduce into F_P ───────────────────────────────────
    X_int  = X  % P
    Wq_int = Wq % P
    Wk_int = Wk % P

    # ─── Pick (i,j) to verify ──────────────────────────────
    i, j = 1, 2

    # ─── Build the polynomial F for S_{i,j} ───────────────
    num_bits, F = build_attention_F(X_int, Wq_int, Wk_int, i, j)

    # ─── Compute direct triple‐sum S_{i,j} mod P ───────────
    e = X_int.shape[1]
    d = Wq_int.shape[1]
    S_direct = 0
    for p in range(e):
        for q in range(d):
            for h in range(d):
                S_direct = (S_direct
                            + X_int[i,p] * Wq_int[p,q] % P
                            * X_int[j,q] * Wk_int[q,h] % P
                           ) % P 

    # ─── Sanity: hypercube‐sum of F also equals S_direct ────
    S_hyper = 0
    for z in binary_vectors_lsb(num_bits):
        S_hyper = (S_hyper + F(z)) % P
    assert S_hyper == S_direct, f"Hypercube sum {S_hyper} != direct {S_direct}"

    # ─── Finally: run the modular sum-check on F ───────────
    cheat = -1
    ok = sumcheck(F, num_bits, cheat)
    assert ok, "Sum-check failed"

    print("passed!")
# %%
# ---------------------------
# Run all tests
# ---------------------------
if __name__ == "__main__":
    test_binary_vectors(); print("binary_vectors_lsb OK")
    test_mle();             print("MLE OK")
    test_build_attention_F(); print("build_attention_F OK")
    test_sumcheck();        print("sumcheck OK")
    
# %%
