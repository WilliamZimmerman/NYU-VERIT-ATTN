#include <cstdint>
#include <vector>
#include <random>
#include <iostream>
#include <cassert>
#include <cmath>
using namespace std;
using u64 = uint64_t;

// A 61-bit Mersenne prime
static constexpr u64 P = (1ULL<<61) - 1;

// ── Field ops ─────────────────────────────────────────────
inline u64 fadd(u64 a, u64 b) {
    u64 c = a + b;
    return c >= P ? c - P : c;
}
inline u64 fmul(u64 a, u64 b) {
    return (u64)((__uint128_t)a * b % P);
}

// ── Enumerate the 0/1 hypercube in LSB order ───────────────
vector<vector<uint8_t>> binary_vectors_lsb(int b) {
    int N = 1<<b;
    vector<vector<uint8_t>> out(N, vector<uint8_t>(b));
    for(int i = 0; i < N; i++){
        for(int k = 0; k < b; k++){
            out[i][k] = (i >> k) & 1;
        }
    }
    return out;
}

// ── Multilinear‐extension for an N×D matrix ───────────────
struct MLE {
    int N,D, rb, cb;
    const u64* M;
    vector<vector<uint8_t>> row_bits, col_bits;

    MLE(const u64* data, int N_, int D_)
      : N(N_), D(D_),
        rb((int)log2(N_)), cb((int)log2(D_)),
        M(data),
        row_bits(binary_vectors_lsb(rb)),
        col_bits(binary_vectors_lsb(cb))
    {}

    // ~M(s,t) = Σ_{i< N, j< D} M[i,j]·L_i(s)·L_j(t)
    u64 eval(const u64* s, const u64* t) const {
        u64 tot = 0;
        for(int i = 0; i < N; i++){
            // build L_i(s)
            u64 Li = 1;
            for(int b = 0; b < rb; b++){
                u64 bit = row_bits[i][b];
                u64 term = fadd(
                    fmul(fadd(1, P - s[b]), (u64)(1 - bit)),
                    fmul(s[b], bit)
                );
                Li = fmul(Li, term);
            }
            for(int j = 0; j < D; j++){
                u64 Lj = 1;
                for(int b = 0; b < cb; b++){
                    u64 bit = col_bits[j][b];
                    u64 term = fadd(
                        fmul(fadd(1, P - t[b]), (u64)(1 - bit)),
                        fmul(t[b], bit)
                    );
                    Lj = fmul(Lj, term);
                }
                tot = fadd(tot, fmul(M[i*D + j], fmul(Li, Lj)));
            }
        }
        return tot;
    }
};

// ── Prover: holds f1, f2, raw; runs sum-check on F(p,q,k) = f1(p,k)*f2(q,k) ───
class Prover {
public:
    int e, d;
    int bp, bq, bk, m;
    vector<u64> f1_arr, f2_arr;
    MLE mle_f1, mle_f2;
    u64 raw_claim;

    Prover(const u64* X, const u64* Wq, const u64* Wk,
           int n, int e_, int d_, int i, int j)
      : e(e_), d(d_),
        bp((int)log2(e)), bq((int)log2(e)), bk((int)log2(d)),
        m(bp + bq + bk),
        f1_arr(e*d), f2_arr(e*d),
        mle_f1(f1_arr.data(), e, d),
        mle_f2(f2_arr.data(), e, d),
        raw_claim(0)
    {
        // 1) build f1(p,k) = X[i,p]*Wq[p,k]
        for(int p = 0; p < e; p++){
            for(int k = 0; k < d; k++){
                f1_arr[p*d + k] = fmul(X[i*e + p], Wq[p*d + k]);
            }
        }
        // 2) build f2(q,k) = X[j,q]*Wk[q,k]
        for(int q = 0; q < e; q++){
            for(int k = 0; k < d; k++){
                f2_arr[q*d + k] = fmul(X[j*e + q], Wk[q*d + k]);
            }
        }
        // 3) compute raw_claim = Σ_{p,q,k} f1(p,k)*f2(q,k)
        for(int p = 0; p < e; p++){
          for(int q = 0; q < e; q++){
            for(int k = 0; k < d; k++){
              raw_claim = fadd(raw_claim,
                            fmul(f1_arr[p*d + k],
                                 f2_arr[q*d + k]));
            }
          }
        }
    }

    // initial claim = that big triple sum
    u64 initial_claim() const {
        return raw_claim;
    }

    // partial sums for round r
    pair<u64,u64> send_partial_sums(const vector<u64>& chal, int r) const {
        return { h(chal, r, 0), h(chal, r, 1) };
    }

    // h_r(u)
    u64 compute_claim_at(const vector<u64>& chal, int r, u64 u) const {
        if(u == 0 || u == 1) {
            auto pr = send_partial_sums(chal, r);
            return u == 0 ? pr.first : pr.second;
        }
        return h(chal, r, u);
    }

    // final F(z) at the full assignment
    u64 final_evaluation(const vector<u64>& chal) const {
        // split bits
        const u64* zp = chal.data();
        const u64* zq = zp + bp;
        const u64* zk = zq + bq;
        u64 a = mle_f1.eval(zp, zk);
        u64 b = mle_f2.eval(zq, zk);
        return fmul(a,b);
    }

private:
    // helper to compute h_r(u)
    u64 h(const vector<u64>& chal, int r, u64 u) const {
        int rem = m - (r + 1);
        u64 tot = 0;
        vector<u64> tail(rem), z;
        for(int tix = 0; tix < (1<<rem); tix++){
            // build full z
            z = chal;             // first r bits
            z.push_back(u);       // bit r
            for(int b = 0; b < rem; b++){
                tail[b] = (tix >> b) & 1;
                z.push_back(tail[b]);
            }
            // evaluate F at z
            const u64* zp = z.data();
            const u64* zq = zp + bp;
            const u64* zk = zq + bq;
            u64 a = mle_f1.eval(zp, zk);
            u64 b = mle_f2.eval(zq, zk);
            tot = fadd(tot, fmul(a,b));
        }
        return tot;
    }
};

// ── Verifier: standard m‐round sum‐check ────────────────────
class Verifier {
public:
    mt19937_64 rng;
    int m;
    Verifier(int m_) : rng(random_device{}()), m(m_) {}

    bool run(Prover &Prov) {
        u64 claim = Prov.initial_claim();
        cout<<"[V] initial claim = "<<claim<<"\n";

        vector<u64> chal;
        for(int r = 0; r < m; r++){
            auto [h0,h1] = Prov.send_partial_sums(chal, r);
            cout<<"[P→V] round "<<r<<": h(0)="<<h0<<", h(1)="<<h1<<"\n";
            if(fadd(h0,h1) != claim){
                cout<<"[V] REJECT at round "<<r<<"\n"; 
                return false;
            }
            u64 u = rng() % P;
            cout<<"[V→P] challenge u_"<<r<<"="<<u<<"\n";
            claim = Prov.compute_claim_at(chal, r, u);
            chal.push_back(u);
            cout<<"[P→V] new claim = "<<claim<<"\n";
        }

        u64 finalv = Prov.final_evaluation(chal);
        cout<<"[V] final eval = "<<finalv
            <<", expected = "<<claim<<"\n";
        if(finalv != claim){
            cout<<"[V] FINAL REJECT\n";
            return false;
        }
        cout<<"[V] ACCEPT\n";
        return true;
    }
};

int main(){
    // toy sizes (powers of two)
    int n=4, e=4, d=4;
    assert((1<<int(log2(e)))==e);
    assert((1<<int(log2(d)))==d);

    // pick two token indices
    int i = 1, j = 2;

    // toy data
    u64 X[16] = {
       1,2,3,4,  5,6,7,8,  9,10,11,12,  13,14,15,16
    };
    // Wq as e×d, Wk as e×d
    u64 Wq[16] = {
       1,2,3,4,  5,6,7,8,  9,10,11,12,  13,14,15,16
    };
    u64 Wk[16] = {
       2,1,4,3,  6,5,8,7,  10,9,12,11,  14,13,16,15
    };

    // build prover & verifier
    Prover   prover(X, Wq, Wk, n, e, d, i, j);
    Verifier verifier(prover.bp + prover.bq + prover.bk);

    cout<<"=== Sum-check for triple-sum attention ===\n";
    bool ok = verifier.run(prover);
    cout<<(ok?"✔︎ Verified":"✘ Failed")<<"\n";
    return 0;
}
