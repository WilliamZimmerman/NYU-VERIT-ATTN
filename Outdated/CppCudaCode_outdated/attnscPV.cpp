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

// Compute a^e mod P in O(log e) time.
u64 modexp(u64 a, u64 e) {
    u64 res = 1;
    while(e) {
        if(e & 1) res = (u64)((__uint128_t)res * a % P);
        a = (u64)((__uint128_t)a * a % P);
        e >>= 1;
    }
    return res;
}

// We'll fill this at runtime once we know d:
u64 inv_sqrt_dk;
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

class Prover {
  public:
    u64 S_direct;
    AttentionPoly poly;
    int m;

    Prover(const u64* X, const u64* Wq, const u64* Wk,
           int n, int e, int d, int i, int j)
      : poly(X,Wq,Wk,n,e,d,i,j),
        m(poly.m)
    {
        // compute S_direct by brute‐force triple loop
        S_direct = 0;
        for(int p=0; p<e; p++)
          for(int q=0; q<d; q++)
            for(int h=0; h<d; h++){
              u64 term = fmul(X[i*e + p], Wq[p*d + q]);
              term = fmul(term, fmul(X[j*e + q], Wk[q*d + h]));
              S_direct = fadd(S_direct, term);
            }
    }

    // round‐0: send initial claim
    u64 initial_claim() const {
        return S_direct;
    }

    // rounds 1..m: send h_r(0), h_r(1)
    pair<u64,u64> send_partial_sums(const vector<u64>& challenges, int r) const {
        auto h = [&](u64 u)->u64{
            u64 tot = 0;
            int rem = m - (r+1);
            vector<u64> tail(rem), z2;
            for(int tix=0; tix<(1<<rem); tix++){
                z2 = challenges;
                z2.push_back(u);
                for(int k=0;k<rem;k++){
                    tail[k] = (tix>>k)&1;
                    z2.push_back(tail[k]);
                }
                tot = fadd(tot, poly.eval(z2));
            }
            return tot;
        };
        return { h(0), h(1) };
    }

    // after receiving challenge u, compute new claim = h_r(u)
    u64 compute_claim_at(const vector<u64>& challenges, int r, u64 u) const {
        auto sums = send_partial_sums(challenges, r);
        if(u==0) return sums.first;
        if(u==1) return sums.second;
        // for general u:
        u64 tot = 0;
        int rem = m - (r+1);
        vector<u64> tail(rem), z2;
        for(int tix=0; tix<(1<<rem); tix++){
            z2 = challenges;
            z2.push_back(u);
            for(int k=0;k<rem;k++){
                tail[k] = (tix>>k)&1;
                z2.push_back(tail[k]);
            }
            tot = fadd(tot, poly.eval(z2));
        }
        return tot;
    }

    // final: send F(challenges)
    u64 final_evaluation(const vector<u64>& challenges) const {
        return poly.eval(challenges);
    }
};

class Verifier {
  public:
    mt19937_64 rng;
    int m;

    Verifier(int m_) : rng(123), m(m_) {}

    bool run(Prover& prover) {
        // Round 0
        u64 claim = prover.initial_claim();
        cout<<"[Verifier] initial claim = "<<claim<<"\n";

        vector<u64> challenges;
        // Rounds 1..m
        for(int r=0; r<m; r++){
            auto [h0,h1] = prover.send_partial_sums(challenges, r); 
            cout<<"[Prover→Verifier] round "<<r<<": h(0)="<<h0<<", h(1)="<<h1<<"\n";
            if(fadd(h0,h1) != claim){
                cout<<"[Verifier] REJECT at round "<<r<<"\n";
                return false;
            }
          
            u64 rr = rng() % P;
            cout<<"[Verifier→Prover] challenge r_"<<r<<" = "<<rr<<"\n";
            
            u64 new_claim = prover.compute_claim_at(challenges, r, rr);
            challenges.push_back(rr);
            claim = new_claim;
            cout<<"[Prover→Verifier] new claim = "<<claim<<"\n";
        }

        // Final check
        u64 final_val = prover.final_evaluation(challenges);
        cout<<"[Verifier] final eval = "<<final_val<<", expected = "<<claim<<"\n";
        if(final_val != claim){
            cout<<"[Verifier] FINAL REJECT\n";
            return false;
        }
        cout<<"[Verifier] ACCEPT\n";
        return true;
    }
};

// ─── Demo in main() ─────────────────────────────────────────
int main(){
    // 1) Specify dimensions at runtime
    int n, e, d;
    cout << "Enter n (# tokens), e (embed dim), d (head dim): ";
    cin >> n >> e >> d;

    // 2) Allocate flat arrays as vectors
    vector<u64> X(n * e);
    vector<u64> Wq(e * d);
    vector<u64> Wk(e * d);  // same shape as Wq

    // 3) Fill them however you like—here, random small ints
    mt19937_64 rng(42);
    for(auto &x : X)  x = rng() % 10;  // values 0..9
    for(auto &w : Wq) w = rng() % 10;
    for(auto &w : Wk) w = rng() % 10;

    // 4) Pick an (i, j) pair to verify
    int i = rng() % n;
    int j = rng() % n;
    cout << "Verifying entry (i="<<i<<", j="<<j<<")\n";

    // 5) Construct prover and verifier
    Prover prover(X.data(), Wq.data(), Wk.data(), n, e, d, i, j);
    Verifier verifier(prover.m);

    // 6) Run the protocol
    cout << "Running sum-check …\n";
    bool ok = verifier.run(prover);
    cout << (ok ? "SUCCESS" : "FAIL") << "\n";

    return ok ? 0 : 1;
}
