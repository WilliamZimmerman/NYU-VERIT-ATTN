// sumcheck_attention.cpp
#include <cstdint>
#include <vector>
#include <random>
#include <iostream>
#include <cassert>
#include <stdlib.h>
#include <algorithm>
#include <limits>
using namespace std;
using u64 = uint64_t;
static constexpr u64 P = (1ULL<<61) - 1;

// fast field ops
inline u64 fadd(u64 a, u64 b) { u64 c = a+b; return c>=P?c-P:c; }
inline u64 fmul(u64 a, u64 b){ __uint128_t t=(__uint128_t)a*b; return (u64)(t%P); }

// hypercube enumerator
vector<vector<uint8_t>> binary_vectors_lsb(int b) {
    int N = 1<<b;
    vector<vector<uint8_t>> out(N, vector<uint8_t>(b));
    for(int i=0;i<N;i++)
      for(int k=0;k<b;k++)
        out[i][k] = (i>>k)&1;
    return out;
}

struct MLE {
    int N, D, b_p, b_q;
    const u64* M;                           // pointer to N×D data
    vector<vector<uint8_t>> row_bits;       
    vector<vector<uint8_t>> col_bits;

    // constructor: note semicolon after struct vars above
    MLE(const u64* M_data, int N_, int D_)
      : N(N_), D(D_), 
        b_p((int)log2(N_)), 
        b_q((int)log2(D_)),
        M(M_data)
    {
        row_bits = binary_vectors_lsb(b_p);
        col_bits = binary_vectors_lsb(b_q);
    }

    // evaluate multilinear extension at (s,t)
    u64 eval(const u64* s, const u64* t) const {
        u64 total = 0;
        for(int i=0;i<N;i++){
            // compute L_i(s)
            u64 Li = 1;
            for(int k=0;k<b_p;k++){
                u64 bit = row_bits[i][k];
                u64 term = fadd(
                  fmul(fadd(1, P - s[k]), (u64)(1 - bit)),
                  fmul(s[k], bit)
                );
                Li = fmul(Li, term);
            }
            // loop over columns
            for(int j=0;j<D;j++){
                u64 Lj = 1;
                for(int m=0;m<b_q;m++){
                    u64 bit = col_bits[j][m];
                    u64 term = fadd(
                      fmul(fadd(1, P - t[m]), (u64)(1 - bit)),
                      fmul(t[m], bit)
                    );
                    Lj = fmul(Lj, term);
                }
                // accumulate
                total = fadd(total,
                             fmul(M[i*D + j], fmul(Li, Lj)));
            }
        }
        return total;
    }
};

// ———————— Attention‐Sum Polynomial Builder ————————
struct AttentionPoly {
    int b_p, b_q, m;
    MLE mle_Xi, mle_Wq, mle_Xj, mle_Wk;

    AttentionPoly(const u64 *X_int, const u64 *Wq_int, const u64 *Wk_int,
                  int n, int e, int d, int i, int j)
      : b_p(log2(e)), b_q(log2(d)), m(b_p + 2*b_q),
        mle_Xi(&X_int[i*e], e, 1),      // X_i row: e×1
        mle_Wq(Wq_int, e, d),           // Wq: e×d
        mle_Xj(&X_int[j*e], 1, d),      // X_j row: 1×d
        mle_Wk(Wk_int, d, d)            // Wk: d×d
    {}

    // Evaluate F(z_bits) mod P
    u64 eval(const vector<u64> &z) const {
        const u64 *s = z.data();
        const u64 *t = z.data() + b_p;
        const u64 *u = z.data() + b_p + b_q;
        u64 v1 = mle_Xi.eval(s, nullptr);
        u64 v2 = mle_Wq.eval(s, t);
        u64 v3 = mle_Xj.eval(nullptr, t);
        u64 v4 = mle_Wk.eval(t, u);
        return fmul(fmul(v1, v2), fmul(v3, v4));
    }
};

// ———————— Prover & Verifier Simulation ————————
bool simulate_sumcheck(const AttentionPoly &poly) {
    int m = poly.m;
    // 1) Prover computes initial claim S_direct via triple loop
    u64 claim = 0;
    // (But we can also compute it by summing F(z) outright:)
    vector<u64> z(m);
    for(int i = 0; i < (1<<m); i++){
        for(int k = 0; k < m; k++) z[k] = (i>>k)&1;
        claim = fadd(claim, poly.eval(z));
    }
    cout<<"[Verifier] initial claim = "<<claim<<"\n";

    vector<u64> challenges;
    mt19937_64 rng(12345);

    // 2) m rounds
    for(int r = 0; r < m; r++){
        // Prover sends h0, h1
        auto h = [&](u64 u)->u64 {
            u64 tot = 0;
            int rem = m - (r+1);
            vector<u64> tail(rem), z2;
            for(int j = 0; j < (1<<rem); j++){
                // build tail
                z2 = challenges;
                z2.push_back(u);
                for(int k=0;k<rem;k++){
                    tail[k] = (j>>k)&1;
                    z2.push_back(tail[k]);
                }
                tot = fadd(tot, poly.eval(z2));
            }
            return tot;
        };

        u64 h0 = h(0), h1 = h(1);
        cout<<"[Prover] round "<<r<<": h(0)="<<h0<<", h(1)="<<h1<<"\n";

        // Verifier checks
        if(fadd(h0,h1) != claim){
            cout<<"[Verifier] reject at round "<<r<<"\n";
            return false;
        }

        // Verifier samples challenge r_r
        u64 r_r = rng() % P;
        cout<<"[Verifier] challenge r_r="<<r_r<<"\n";

        // Prover responds with new claim = h(r_r)
        u64 new_claim = h(r_r);
        challenges.push_back(r_r);
        claim = new_claim;
        cout<<"[Prover] new claim = "<<claim<<"\n";
    }

    // 3) Final check
    u64 final_eval = poly.eval(challenges);
    cout<<"[Verifier] final eval = "<<final_eval<<", claim = "<<claim<<"\n";
    if(final_eval != claim){
        cout<<"[Verifier] final mismatch -> reject\n";
        return false;
    }

    cout<<"[Verifier] accepts\n";
    return true;
}

// ———————— Main Demo ————————
int main(){
    // Toy data: 3×4 token embeddings, e=4,d=2
    vector<u64> X = {
         1,2,3,4,
         5,6,7,8,
         9,10,11,12
    };
    vector<u64> Wq = {
        1,2,
        3,4,
        5,6,
        7,8
    };
    vector<u64> Wk = {
        2,1,
        4,3,
        6,5,
        8,7
    };

    int n=3,e=4,d=2, i=1,j=2;
    AttentionPoly poly(X.data(), Wq.data(), Wk.data(), n,e,d, i,j);

    cout<<"=== Simulating modular sum-check ===\n";
    simulate_sumcheck(poly);

    return 0;
}

