#include <cstdint>
#include <vector>
#include <random>
#include <iostream>
#include <cassert>
#include <cmath>
using namespace std;
using u64 = uint64_t;
static constexpr u64 P = (1ULL<<61) - 1;

// ─── Fast field ops ───────────────────────────────────────
inline u64 fadd(u64 a, u64 b) { u64 c = a+b; return c>=P?c-P:c; }
inline u64 fmul(u64 a, u64 b){ return (u64)((__uint128_t)a*b % P); }

// ─── Modular exponentiation for inverses ─────────────────
u64 modexp(u64 a, u64 e) {
    u64 res = 1;
    while(e) {
        if(e & 1) res = fmul(res, a);
        a = fmul(a, a);
        e >>= 1;
    }
    return res;
}

// ─── Hypercube enumerator ─────────────────────────────────
vector<vector<uint8_t>> binary_vectors_lsb(int b) {
    int N = 1<<b;
    vector<vector<uint8_t>> out(N, vector<uint8_t>(b));
    for(int i=0;i<N;i++)
      for(int k=0;k<b;k++)
        out[i][k] = (i>>k)&1;
    return out;
}

// ─── Multilinear‐extension of an N×D matrix ───────────────
struct MLE {
    int N, D, b_p, b_q;
    const u64* M;
    vector<vector<uint8_t>> row_bits, col_bits;

    MLE(const u64* M_data, int N_, int D_)
      : N(N_), D(D_),
        b_p((int)log2(N_)), b_q((int)log2(D_)),
        M(M_data),
        row_bits(binary_vectors_lsb(b_p)),
        col_bits(binary_vectors_lsb(b_q))
    {}

    u64 eval(const u64* s, const u64* t) const {
        u64 total = 0;
        for(int i=0;i<N;i++){
            u64 Li = 1;
            for(int k=0;k<b_p;k++){
                u64 bit = row_bits[i][k];
                u64 term = fadd(
                  fmul(fadd(1, P - s[k]), (u64)(1 - bit)),
                  fmul(s[k], bit)
                );
                Li = fmul(Li, term);
            }
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
                total = fadd(total,
                             fmul(M[i*D + j], fmul(Li, Lj)));
            }
        }
        return total;
    }
};

// ─── Attention polynomial, **now parameterized by inv_scaling** ───
struct AttentionPoly {
    int b_p, b_q, m;
    u64 inv_scaling;                  
    MLE mle_Xi, mle_Wq, mle_Xj, mle_Wk;

    AttentionPoly(const u64 *X_int, const u64 *Wq_int, const u64 *Wk_int,
                  int n, int e, int d, int i, int j,
                  u64 inv_sqrt_d)
      : b_p(log2(e)), b_q(log2(d)), m(b_p + 2*b_q),
        inv_scaling(inv_sqrt_d),          // store it
        mle_Xi(&X_int[i*e], e, 1),
        mle_Wq(Wq_int, e, d),
        mle_Xj(&X_int[j*e], 1, d),
        mle_Wk(Wk_int, d, d)
    {}

    u64 eval(const vector<u64> &z) const {
        const u64 *s = z.data();
        const u64 *t = z.data() + b_p;
        const u64 *u = z.data() + b_p + b_q;
        u64 v1 = mle_Xi.eval(s, nullptr);
        u64 v2 = mle_Wq.eval(s, t);
        u64 v3 = mle_Xj.eval(nullptr, t);
        u64 v4 = mle_Wk.eval(t, u);
        u64 raw = fmul(fmul(v1, v2), fmul(v3, v4));
        return fmul(raw, inv_scaling);
    }
};
// Prover with scaling in calculation
class Prover {
  public:
    // After construction, S_direct holds the *normalized* score = raw_attention / sqrt(d)
    u64 S_direct;
    AttentionPoly poly;
    int m;

    // Constructor now takes inv_sqrt_d and immediately applies it
    Prover(const u64* X, const u64* Wq, const u64* Wk,
           int n, int e, int d, int i, int j,
           u64 inv_sqrt_d)
      : poly(X, Wq, Wk, n, e, d, i, j, inv_sqrt_d),
        m(poly.m)
    {
        // 1) Compute the raw triple-sum attention
        u64 raw = 0;
        for(int p = 0; p < e; p++) {
            for(int q = 0; q < d; q++) {
                for(int h = 0; h < d; h++) {
                    u64 term = fmul(X[i*e + p], Wq[p*d + q]);
                    term = fmul(term, fmul(X[j*e + q], Wk[q*d + h]));
                    raw = fadd(raw, term);
                }
            }
        }
        // 2) Scale once by inv_sqrt_d to normalize
        S_direct = fmul(raw, inv_sqrt_d);
    }

    // Round-0: send the *already-scaled* attention score
    u64 initial_claim() const {
        return S_direct;
    }

    // Rounds 1..m: send h_r(0),h_r(1) computed over poly.eval(), which itself is scaled
    pair<u64,u64> send_partial_sums(const vector<u64>& challenges, int r) const {
        auto h = [&](u64 u)->u64 {
            u64 tot = 0;
            int rem = m - (r + 1);
            vector<u64> tail(rem), z2;
            for(int tix = 0; tix < (1 << rem); tix++){
                z2 = challenges;
                z2.push_back(u);
                for(int k = 0; k < rem; k++){
                    tail[k] = (tix >> k) & 1;
                    z2.push_back(tail[k]);
                }
                // poly.eval(z2) already returns the scaled F'(z2)
                tot = fadd(tot, poly.eval(z2));
            }
            return tot;
        };
        return { h(0), h(1) };
    }

    // After receiving a challenge u, compute the new claim = h_r(u)
    u64 compute_claim_at(const vector<u64>& challenges, int r, u64 u) const {
        // For u==0 or 1 we can reuse send_partial_sums:
        if(u == 0 || u == 1) {
            auto pr = send_partial_sums(challenges, r);
            return u == 0 ? pr.first : pr.second;
        }
        // Otherwise do the full tail‐sum:
        u64 tot = 0;
        int rem = m - (r + 1);
        vector<u64> tail(rem), z2;
        for(int tix = 0; tix < (1 << rem); tix++){
            z2 = challenges;
            z2.push_back(u);
            for(int k = 0; k < rem; k++){
                tail[k] = (tix >> k) & 1;
                z2.push_back(tail[k]);
            }
            tot = fadd(tot, poly.eval(z2));
        }
        return tot;
    }

    // Final step: send poly.eval(challenges), which is already scaled
    u64 final_evaluation(const vector<u64>& challenges) const {
        return poly.eval(challenges);
    }
};

// ─── Prover with scaled initial claim ──────────────────────
// class Prover {
//   public:
//     u64 S_direct;
//     AttentionPoly poly;
//     int m;

//     Prover(const u64* X, const u64* Wq, const u64* Wk,
//            int n, int e, int d, int i, int j,
//            u64 inv_sqrt_d)
//       : poly(X,Wq,Wk,n,e,d,i,j, inv_sqrt_d),
//         m(poly.m)
//     {
//         S_direct = 0;
//         for(int p=0; p<e; p++)
//           for(int q=0; q<d; q++)
//             for(int h=0; h<d; h++){
//               u64 term = fmul(X[i*e + p], Wq[p*d + q]);
//               term = fmul(term, fmul(X[j*e + q], Wk[q*d + h]));
//               S_direct = fadd(S_direct, term);
//             }
//     }

//     // scale the initial claim
//     u64 initial_claim() const {
//         return fmul(S_direct, poly.inv_scaling);
//     }

//     // send_partial_sums, compute_claim_at, final_evaluation
//     pair<u64,u64> send_partial_sums(const vector<u64>& challenges, int r) const {
//         auto h = [&](u64 u)->u64{
//             u64 tot = 0;
//             int rem = m - (r+1);
//             vector<u64> tail(rem), z2;
//             for(int tix=0; tix<(1<<rem); tix++){
//                 z2 = challenges; z2.push_back(u);
//                 for(int k=0;k<rem;k++){
//                     tail[k] = (tix>>k)&1;
//                     z2.push_back(tail[k]);
//                 }
//                 tot = fadd(tot, poly.eval(z2));
//             }
//             return tot;
//         };
//         return { h(0), h(1) };
//     }

//     u64 compute_claim_at(const vector<u64>& challenges, int r, u64 u) const {
//         if(u==0||u==1){
//             auto pr = send_partial_sums(challenges,r);
//             return u==0 ? pr.first : pr.second;
//         }
//         // fallback for non–0/1 u
//         u64 tot = 0;
//         int rem = m - (r+1);
//         vector<u64> tail(rem), z2;
//         for(int tix=0; tix<(1<<rem); tix++){
//             z2 = challenges; z2.push_back(u);
//             for(int k=0;k<rem;k++){
//                 tail[k] = (tix>>k)&1;
//                 z2.push_back(tail[k]);
//             }
//             tot = fadd(tot, poly.eval(z2));
//         }
//         return tot;
//     }

//     u64 final_evaluation(const vector<u64>& challenges) const {
//         return poly.eval(challenges);
//     }
// };

class Verifier {
  public:
    mt19937_64 rng;
    int m;
    Verifier(int m_) : rng(123), m(m_) {}
    bool run(Prover& prover) {
        u64 claim = prover.initial_claim();
        cout<<"[Verifier] initial claim = "<<claim<<"\n";

        vector<u64> challenges;
        for(int r=0; r<m; r++){
            auto [h0,h1] = prover.send_partial_sums(challenges, r);
            cout<<"[Prover→Verifier] round "<<r<<": h(0)="<<h0<<", h(1)="<<h1<<"\n";
            if(fadd(h0,h1) != claim){
                cout<<"[Verifier] REJECT at round "<<r<<"\n"; return false;
            }
            u64 rr = rng() % P;
            cout<<"[Verifier] challenge r_"<<r<<" = "<<rr<<"\n";
            u64 new_claim = prover.compute_claim_at(challenges, r, rr);
            challenges.push_back(rr);
            claim = new_claim;
            cout<<"[Prover→Verifier] new claim = "<<claim<<"\n";
        }
        u64 final_val = prover.final_evaluation(challenges);
        cout<<"[Verifier] final eval = "<<final_val<<", expected = "<<claim<<"\n";
        if(final_val != claim){
            cout<<"[Verifier] FINAL REJECT\n"; return false;
        }
        cout<<"[Verifier] ACCEPT\n"; return true;
    }
};

// ─── Demo in main() ─────────────────────────────────────────
int main(){
    int n=4, e=4, d=4, i=1, j=2;

    // 6) Compute inv_sqrt_d at runtime for any d
    int sqrt_d = (int)std::round(std::sqrt(double(d)));
    assert(sqrt_d*sqrt_d == d && "d must be a perfect square");
    u64 inv_sqrt_d = modexp((u64)sqrt_d, P-2);

    // 1) Prepare data (example fixed or random)
    u64 X_arr[12] = {1,2,3,4, 5,6,7,8, 9,10,11,12};
    u64 Wq_arr[8] = {1,2,3,4,5,6,7,8};
    u64 Wk_arr[8] = {2,1,4,3,6,5,8,7};

    // 2) Build Prover & Verifier with our inv_sqrt_d
    Prover prover(X_arr, Wq_arr, Wk_arr, n,e,d,i,j, inv_sqrt_d);
    Verifier verifier(prover.m);

    cout<<"Running sum-check with 1/sqrt(d)="<<inv_sqrt_d<<"\n";
    verifier.run(prover);
    return 0;
}
