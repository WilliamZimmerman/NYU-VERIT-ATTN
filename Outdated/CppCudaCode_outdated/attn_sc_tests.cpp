// attn_sc_tests.cpp
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <functional>
#include <random>

// ─── Include your implementation here ───────────────────────
// Either paste attn_sc_scaled_updated.cpp contents, or:
#include "attn_sc_scaled_updated.cpp"
// ─────────────────────────────────────────────────────────────

using namespace std;
using u64 = uint64_t;

// Free‐function sumcheck for testing zero‐ and other polys
bool sumcheck(const function<u64(const vector<u64>&)> &F, int m) {
    // 1) initial claim
    u64 claim = 0;
    vector<u64> z(m);
    for(int i = 0; i < (1<<m); i++){
        for(int k = 0; k < m; k++) z[k] = (i>>k)&1;
        claim = fadd(claim, F(z));
    }
    // 2) rounds
    vector<u64> chal;
    for(int r = 0; r < m; r++){
        auto h = [&](u64 u){
            u64 tot = 0;
            int rem = m - (r+1);
            vector<u64> tail(rem), z2;
            for(int tix=0; tix < (1<<rem); tix++){
                z2 = chal;
                z2.push_back(u);
                for(int b=0; b<rem; b++){
                    tail[b] = (tix>>b)&1;
                    z2.push_back(tail[b]);
                }
                tot = fadd(tot, F(z2));
            }
            return tot;
        };
        u64 h0 = h(0), h1 = h(1);
        if(fadd(h0,h1) != claim) return false;
        // use a fixed challenge for determinism
        u64 rr = 1;
        chal.push_back(rr);
        claim = h(rr);
    }
    // 3) final eval
    return F(chal) == claim;
}

// ─── 1) field ops ──────────────────────────────────────────
void test_field_ops() {
    assert(fadd(P-1,1) == 0);
    assert(fadd(5,7) == 12);
    assert(fmul(2,3) == 6);
    u64 inv2 = modexp(2, P-2);
    assert(fmul(2, inv2) == 1);
    cout<<"✅ test_field_ops\n";
}

// ─── 2) hypercube enumeration ──────────────────────────────
void test_binary_vectors() {
    auto v = binary_vectors_lsb(3);
    assert(v.size() == 8);
    assert(v[0] == vector<uint8_t>({0,0,0}));
    assert(v[1] == vector<uint8_t>({1,0,0}));
    assert(v[7] == vector<uint8_t>({1,1,1}));
    cout<<"✅ test_binary_vectors\n";
}

// ─── 3) MLE correctness on 2×2 ─────────────────────────────
void test_mle_basic() {
    u64 M_data[4] = {5,9,2,7};
    MLE mle(M_data, 2, 2);
    // corners
    for(int si=0; si<2; si++){
      for(int tj=0; tj<2; tj++){
        u64 s[1] = { (u64)si };
        u64 t[1] = { (u64)tj };
        u64 want = M_data[si*2 + tj];
        u64 got  = mle.eval(s,t);
        assert(got == want);
      }
    }
    // midpoint: (s,t)=(½,½)
    u64 half = (P+1)/2;
    u64 val = mle.eval(&half, &half);
    // (5+9+2+7)/4 = 23/4 mod P
    u64 inv4 = modexp(4, P-2);
    assert(val == fmul(23, inv4));
    cout<<"✅ test_mle_basic\n";
}

// ─── 4) Prover builds Q,K,Kᵀ,C correctly ──────────────────
void test_prover_builds() {
    int n=2, e=2, d=1;
    // simple X, Wq, Wk
    u64 X[4]  = {1,2,3,4};
    u64 Wq[2] = {1,1};
    u64 Wk[2] = {1,1};
    u64 inv_sqrt = 1;  // sqrt(d)=1
    Prover P(X,Wq,Wk,n,e,d,inv_sqrt);
    // Q = [1+2, 3+4] = [3,7]
    assert(P.Q_arr == vector<u64>({3,7}));
    // K same
    assert(P.K_arr == vector<u64>({3,7}));
    // Kᵀ is (1×2): [3,7]
    assert(P.Kt_arr == vector<u64>({3,7}));
    // C = Q·Kᵀ = [[9,21],[21,49]]
    assert(P.C_arr == vector<u64>({9,21,21,49}));
    cout<<"✅ test_prover_builds\n";
}

// ─── 5) sumcheck on zero‐poly ─────────────────────────────
void test_sumcheck_zero() {
    auto zeroF = [&](const vector<u64>&){ return 0ULL; };
    assert(sumcheck(zeroF, 3));
    cout<<"✅ test_sumcheck_zero\n";
}

// ─── 6) full head protocol acceptance/rejection ──────────
void test_full_head() {
    int n=2, e=2, d=1;
    u64 inv_sqrt = 1;
    u64 X[4]  = {1,2,3,4};
    u64 Wq[2] = {1,1};
    u64 Wk[2] = {1,1};
    // honest
    Prover P1(X,Wq,Wk,n,e,d,inv_sqrt);
    Verifier V1(P1.m);
    assert(V1.run(P1));
    // tamper C_arr
    Prover P2 = P1;
    P2.C_arr[0] = (P2.C_arr[0] + 1) % P;
    Verifier V2(P2.m);
    assert(!V2.run(P2));
    cout<<"✅ test_full_head\n";
}

int main(){
    test_field_ops();
    test_binary_vectors();
    test_mle_basic();
    test_prover_builds();
    test_sumcheck_zero();
    test_full_head();
    cout<<"All tests passed!\n";
    return 0;
}
