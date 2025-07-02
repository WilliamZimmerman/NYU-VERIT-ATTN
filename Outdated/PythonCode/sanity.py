# %%
import numpy as np
for i in range(0,1000):
    # Given matrices
    X = np.matrix([[1,2,3,4],
                [5,6,7,8],
                [9,10,11,12],
                [13,14,15,16]])
    Wq = np.matrix([[1,2,3,4],
                    [5,6,7,8],
                    [9,10,11,12],
                    [13,14,15,16]])
    Wk = np.matrix([[2,1,4,3],
                    [6,5,8,7],
                    [10,9,12,11],
                    [14,13,16,15]])

    # 1) Standard attention via matrix multiplies
    Q = X @ Wq
    K = X @ Wk
    attn_matrix = Q @ K.T

    # 2) Triple-sum inline computation
    n, e = X.shape
    d = Wq.shape[1]
    attn_inline = np.zeros((n,n), dtype=int)

    for i in range(n):
        for j in range(n):
            s = 0
            for p in range(e):
                for q in range(e):
                    for h in range(d):
                        s += X[i,p] * Wq[p,h] * X[j,q] * Wk[q,h]
            attn_inline[i, j] = s

    # Display and verify
    # print("Attention via matrix multiply:\n", attn_matrix)
    # print("\nAttention via inline triple-sum:\n", attn_inline)
    assert np.array_equal(attn_matrix, attn_inline), "Mismatch between methods!"