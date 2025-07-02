#include <cstdint>
#include <vector>
#include <random>
#include <iostream>
#include <cassert>
#include <cmath>
using namespace std;
using u64 = uint64_t;
static constexpr u64 P = (1ULL<<61) - 1;
u64 inv_sqrt_d;
// ─── 1) Field arithmetic ───────────────────────────────────
inline u64 fadd(u64 a, u64 b) { u64 c = a + b; return c >= P ? c - P : c; }
inline u64 fmul(u64 a, u64 b){ return (u64)((__uint128_t)a * b % P); }

// fast exponentiation for inverses
u64 modexp(u64 a, u64 e) {
    u64 res = 1;
    while(e) {
        if(e & 1) res = fmul(res, a);
        a = fmul(a, a);
        e >>= 1;
    }
    return res;
}

// ─── 2) Enumerate {0,1}^b in LSB-first order ───────────────
vector<vector<uint8_t>> binary_vectors_lsb(int b) {
    int N = 1 << b;
    vector<vector<uint8_t>> out(N, vector<uint8_t>(b));
    for(int i = 0; i < N; i++)
      for(int k = 0; k < b; k++)
        out[i][k] = (i >> k) & 1;
    return out;
}

// ─── 3) Multilinear extension of an N×D matrix ─────────────
struct MLE {
    int N, D, b_row, b_col;
    const u64* M;
    vector<vector<uint8_t>> row_bits, col_bits;

    MLE(const u64* data, int N_, int D_)
      : N(N_), D(D_),
        b_row((int)log2(N_)), b_col((int)log2(D_)),
        M(data),
        row_bits(binary_vectors_lsb(b_row)),
        col_bits(binary_vectors_lsb(b_col))
    {}

    // Evaluate ~M(s,t) = Σ_{i,j} M[i,j] L_i(s) L_j(t)
    u64 eval(const u64* s, const u64* t) const {
        u64 total = 0;
        for(int i = 0; i < N; i++){
            u64 Li = 1;
            for(int k = 0; k < b_row; k++){
                u64 bit = row_bits[i][k];
                u64 term = fadd(
                  fmul(fadd(1, P - s[k]), (u64)(1 - bit)),
                  fmul(s[k], bit)
                );
                Li = fmul(Li, term);
            }
            for(int j = 0; j < D; j++){
                u64 Lj = 1;
                for(int m = 0; m < b_col; m++){
                    u64 bit = col_bits[j][m];
                    u64 term = fadd(
                      fmul(fadd(1, P - t[m]), (u64)(1 - bit)),
                      fmul(t[m], bit)
                    );
                    Lj = fmul(Lj, term);
                }
                total = fadd(total,
                             fmul(M[i*D + j], fmul(Li, Lj)));
            }
        }
        return total;
    }
};

// ─── 4) Prover: builds Q, K, C and answers sum-check on F(i,j,k) ───
class Prover {
public:
    int n, d;
    int bi, bj, bk, m;
    vector<u64> Q_arr, K_arr, C_arr, Kt_arr;
    MLE mle_Q, mle_B, mle_C;

    // inv_sqrt_d for the final scaling of C
    Prover(const u64* X, const u64* Wq, const u64* Wk,
           int n_, int e, int d_, u64 inv_sqrt_d)
      : n(n_), d(d_),
        bi((int)log2(n_)), bj((int)log2(n_)), bk((int)log2(d_)),
        m(bi + bj + bk),
        Q_arr(n_*d_), K_arr(n_*d_),
        mle_Q(Q_arr.data(), n_, d_),
        Kt_arr(d_*n_),
        mle_B(Kt_arr.data(), d_, n_),
        C_arr(n_*n_),
        mle_C(C_arr.data(), n_, n_)
    {
    
        // --- compute Q = X·Wq  and  K = X·Wk  (both n×d)
        for(int i = 0; i < n; i++){
            for(int k = 0; k < d; k++){
                u64 sumQ = 0, sumK = 0;
                for(int p = 0; p < e; p++){
                    sumQ = fadd(sumQ,
                                fmul(X[i*e + p], Wq[p*d + k]));
                    sumK = fadd(sumK,
                                fmul(X[i*e + p], Wk[p*d + k]));
                }
                Q_arr[i*d + k] = sumQ;
                K_arr[i*d + k] = sumK;
            }
        }

        for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            u64 sum = 0;
            for(int k = 0; k < d; k++){
            sum = fadd(sum,
                        fmul(Q_arr[i*d + k],
                            K_arr[j*d + k]));
            }
            C_arr[i*n + j] = sum;   //  no fmul(sum, inv_sqrt_d)
        }
        }

        // --- transpose K into Kt_arr (d×n)
        for(int j = 0; j < n; j++){
            for(int k = 0; k < d; k++){
                Kt_arr[k*n + j] = K_arr[j*d + k];//fmul(K_arr[j*d + k], inv_sqrt_d);
            }
        }
    }

    // initial claim: we want Σ_{i,j,k} F(i,j,k) == 0
    u64 initial_claim() const {
        return 0;
    }

    // rounds 1..m: prover sends partial sums h_r(0),h_r(1)
    pair<u64,u64> send_partial_sums(const vector<u64>& chal, int r) const {
        auto h = [&](u64 u) {
            u64 tot = 0;
            int rem = m - (r + 1);
            vector<u64> tail(rem), z;
            for(int tix = 0; tix < (1<<rem); tix++){
                z = chal; 
                z.push_back(u);
                for(int b = 0; b < rem; b++){
                    tail[b] = (tix >> b) & 1;
                    z.push_back(tail[b]);
                }
                // split z into i‐bits, j‐bits, k‐bits
                const u64* zi = z.data();
                const u64* zj = zi + bi;
                const u64* zk = zj + bj;
                // evaluate the three MLEs
                u64 a = mle_Q.eval(zi, zk);    // ~Q(i,k)
                u64 b = mle_B.eval(zk, zj);    // ~K^T(k,j)
                u64 c = mle_C.eval(zi, zj);    // ~C(i,j)
                // F = A*B - C
                u64 Fval = fadd(fmul(a,b), P - c);

                tot = fadd(tot, Fval);
            }
            return tot;
        };
        return { h(0), h(1) };
    }

    // after verifier gives challenge u, compute h_r(u)
    u64 compute_claim_at(const vector<u64>& chal, int r, u64 u) const {
        if(u == 0 || u == 1) {
            auto pr = send_partial_sums(chal,r);
            return u == 0 ? pr.first : pr.second;
        }
        // general u
        u64 tot = 0;
        int rem = m - (r + 1);
        vector<u64> tail(rem), z;
        for(int tix = 0; tix < (1<<rem); tix++){
            z = chal; z.push_back(u);
            for(int b=0;b<rem;b++){
                tail[b] = (tix>>b)&1; z.push_back(tail[b]);
            }
            const u64* zi = z.data();
            const u64* zj = zi + bi;
            const u64* zk = zj + bj;
            u64 a = mle_Q.eval(zi, zk);
            u64 b = mle_B.eval(zk, zj);
            u64 c = mle_C.eval(zi, zj);
            u64 Fval = fadd(fmul(a,b), P - c);
            tot = fadd(tot, Fval);
        }
        return tot;
    }

    // final: prover sends F(challenges) which must equal the last claim
    u64 final_evaluation(const vector<u64>& chal) const {
        const u64* zi = chal.data();
        const u64* zj = zi + bi;
        const u64* zk = zj + bj;
        u64 a = mle_Q.eval(zi, zk);
        u64 b = mle_B.eval(zk, zj);
        u64 c = mle_C.eval(zi, zj);
        return fadd(fmul(a,b), P - c);
    }
};

// ─── 5) Verifier: standard m-round sum‐check ───────────────
class Verifier {
public:
    mt19937_64 rng;
    int m;
    Verifier(int m_) : rng(1234), m(m_) {}

    bool run(Prover &prov) {
        // 0) initial claim
        u64 claim = prov.initial_claim();
        cout << "[V] initial claim = " << claim << "\n";

        // 1) m rounds
        vector<u64> challenges;
        for(int r=0; r<m; r++){
            auto [h0,h1] = prov.send_partial_sums(challenges,r);
            cout << "[P→V] round "<<r<<": h(0)="<<h0<<", h(1)="<<h1<<"\n";
            if(fadd(h0,h1) != claim){
                cout<<"[V] REJECT at round "<<r<<"\n";
                return false;
            }
            u64 rr = rng() % P;
            cout << "[V→P] challenge r_"<<r<<"="<<rr<<"\n";
            u64 new_claim = prov.compute_claim_at(challenges,r,rr);
            challenges.push_back(rr);
            claim = new_claim;
            cout << "[P→V] new claim = " << claim << "\n";
        }

        // 2) final check
        u64 final_val = prov.final_evaluation(challenges);
        cout << "[V] final eval = " << final_val
             << ", expected = " << claim << "\n";
        if(final_val != claim){
            cout<<"[V] FINAL REJECT\n";
            return false;
        }
        cout<<"[V] ACCEPT\n";
        return true;
    }
};

int main(){
    // dimensions
    int n=4, e=4, d=4;
    // you must pick d a perfect square for scaling below:
    int sqrt_d = (int)round(sqrt(double(d)));
    assert(sqrt_d*sqrt_d == d);
    inv_sqrt_d = modexp((u64)sqrt_d, P-2);

    // toy X, Wq, Wk
    u64 X_arr[16] = {
      1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16
    };
    u64 Wq_arr[16] = {
      1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16
    };
    u64 Wk_arr[16] = {
      2,1,4,3, 6,5,8,7, 10,9,12,11, 14,13,16,15
    };

    // build prover & verifier
    Prover prover(X_arr,Wq_arr,Wk_arr,n,e,d,inv_sqrt_d);
    Verifier verifier(prover.m);

    cout<<"=== Running full head sum-check ===\n";
    bool ok = verifier.run(prover);
    cout<<(ok?"✅ ALL OK":"❌ FAILED")<<"\n";
    return ok?0:1;
}
