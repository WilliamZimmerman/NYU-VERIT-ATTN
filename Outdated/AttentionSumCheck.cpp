// sumcheck_attention_bitwise.cpp
// “Proper” bitwise‐multilinear‐extension sum‐check
// Verifies *all* raw‐attention entries S = X Wᵩ Wᵏᵀ at once.

#include <iostream>    // cout, endl
#include <vector>      // std::vector
#include <array>       // std::array
#include <random>      // std::mt19937_64, std::random_device
#include <cstdint>     // uint64_t
#include <cmath>       // std::log2
#include <cassert>     // assert

using u64 = uint64_t;
static constexpr u64 P = (1ULL<<61) - 1;

// -------- modular arithmetic --------
inline u64 modAdd(u64 a, u64 b) {
    u64 c = a + b;
    if (c >= P) c -= P;
    return c;
}
inline u64 modSub(u64 a, u64 b) {
    return a >= b ? a - b : P + a - b;
}
inline u64 modMul(u64 a, u64 b) {
    __uint128_t t = ( __uint128_t )a * b;
    return (u64)(t % P);
}
u64 modPow(u64 base, u64 exp) {
    u64 res = 1;
    while (exp) {
        if (exp & 1) res = modMul(res, base);
        base = modMul(base, base);
        exp >>= 1;
    }
    return res;
}
inline u64 modInv(u64 x) {
    // P prime
    return modPow(x, P-2);
}

// uniform field element
u64 randomField(){
    static std::mt19937_64 rng(std::random_device{}());
    return rng() % P;
}

using Matrix = std::vector<std::vector<u64>>;

// convert integer x ∈ [0,2^m) to its m‐bit representation (LSB first)
std::vector<u64> intToBits(int x, int m) {
    std::vector<u64> bits(m);
    for(int k=0; k<m; ++k) {
        bits[k] = (x>>k)&1;
    }
    return bits;
}

// 1D multilinear extension (over {0,1}^m → F) at point r_bits[0..m-1]
u64 evalMLE1D_bit(const std::vector<u64>& T, const std::vector<u64>& r_bits) {
    int m = (int)r_bits.size();
    int N = 1<<m;
    u64 acc = 0;
    // Sum_{b∈{0,1}^m} T[b_index] * ∏_{k<b:m}( b[k] ? r_bits[k] : (1-r_bits[k]) )
    for(int idx=0; idx<N; ++idx) {
        u64 w = 1;
        for(int k=0; k<m; ++k) {
            if ((idx>>k)&1)        w = modMul(w, r_bits[k]);
            else                   w = modMul(w, modSub(1, r_bits[k]));
        }
        acc = modAdd(acc, modMul(T[idx], w));
    }
    return acc;
}

// 2D multilinear‐extension for A[N0×N1] at (r0_bits, r1_bits)
u64 evalMLE2D_bit(const Matrix& A,
                  const std::vector<u64>& r0_bits,
                  const std::vector<u64>& r1_bits) {
    int m0 = (int)r0_bits.size(), m1 = (int)r1_bits.size();
    int N0 = 1<<m0, N1 = 1<<m1;

    // First collapse rows → B[j] = ∑_i A[i][j]·weight_r0(i_bits)
    std::vector<u64> B(N1);
    for(int j=0; j<N1; ++j) {
        std::vector<u64> col(N0);
        for(int i=0; i<N0; ++i) col[i] = A[i][j];
        B[j] = evalMLE1D_bit(col, r0_bits);
    }
    // Then collapse cols:
    return evalMLE1D_bit(B, r1_bits);
}

enum Round { P_BIT, Q_BIT, H_BIT, DONE };

class Prover {
public:
    const Matrix &X, &Wq, &Wk;
    int n, d, m;
    std::vector<u64> ri_bits, rj_bits;
    std::vector<u64> rp_bits, rq_bits, rh_bits;
    int phase;      // 0..3*m
    bool cheat;

    Prover(const Matrix& X_, const Matrix& Wq_, const Matrix& Wk_, bool c=false)
      : X(X_), Wq(Wq_), Wk(Wk_), 
        n((int)X_.size()), d((int)Wq_.size()),
        m(std::log2(d)), 
        ri_bits(std::log2(n)), rj_bits(std::log2(n)),
        rp_bits(m), rq_bits(m), rh_bits(m),
        phase(0), cheat(c)
    {}

    // Σ = ∑_{p,q,h} G(ri_bits, rj_bits, p_bits, q_bits, h_bits)
    u64 commit() {
        u64 sum = 0;
        // loop over all assignments p,q,h ∈ [0..d)
        for(int p=0; p<d; ++p) {
            auto p_bits = intToBits(p, m);
            for(int q=0; q<d; ++q) {
                auto q_bits = intToBits(q, m);
                for(int h=0; h<d; ++h) {
                    auto h_bits = intToBits(h, m);
                    // G = ˜X(ri_bits,p_bits) * ˜Wq(p_bits,h_bits)
                    //   * ˜X(rj_bits,q_bits) * ˜Wk(q_bits,h_bits)
                    u64 A = evalMLE2D_bit(X,  ri_bits, p_bits);
                    u64 B = evalMLE2D_bit(Wq, p_bits, h_bits);
                    u64 C = evalMLE2D_bit(X,  rj_bits, q_bits);
                    u64 D = evalMLE2D_bit(Wk, q_bits, h_bits);
                    sum = modAdd(sum, modMul(modMul(A,B), modMul(C,D)));
                }
            }
        }
        if (cheat) sum = modAdd(sum, 1);
        std::cout<<"Prover: Σ="<<sum<<"\n";
        return sum;
    }

    std::array<u64,2> sendBitPoly() {
    // --- 0. Which variable and which bit? ---
    int bitIdx, varClass;
    if      (phase <   m) { bitIdx = phase;     varClass = 0; }  // p-bit
    else if (phase < 2*m) { bitIdx = phase - m; varClass = 1; }  // q-bit
    else                  { bitIdx = phase - 2*m; varClass = 2; }// h-bit

    // --- 1. Prepare storage for g(0),g(1) ---
    std::array<u64,2> poly = {0,0};

    // --- 2. For each t in {0,1}, fix the bit and sum out the rest ---
    for(int t = 0; t < 2; ++t) {
        // 2a) fix the current bit to t
        if      (varClass == 0) rp_bits[bitIdx] = t;
        else if (varClass == 1) rq_bits[bitIdx] = t;
        else                    rh_bits[bitIdx] = t;

        u64 localSum = 0;

        // --- 3A) p-bit: sum over all p whose bitIdx==t, all q, all h ---
        if (varClass == 0) {
            for(int p = 0; p < d; ++p) {
                auto p_bits = intToBits(p, m);
                if (p_bits[bitIdx] != t) continue;   // keep only bitIdx=t
                for(int q = 0; q < d; ++q) {
                    auto q_bits = intToBits(q, m);
                    for(int h = 0; h < d; ++h) {
                        auto h_bits = intToBits(h, m);
                        u64 A = evalMLE2D_bit(X,   ri_bits, p_bits);
                        u64 B = evalMLE2D_bit(Wq,  p_bits, h_bits);
                        u64 C = evalMLE2D_bit(X,   rj_bits, q_bits);
                        u64 D = evalMLE2D_bit(Wk,  q_bits,  h_bits);
                        localSum = modAdd(localSum,
                                         modMul(modMul(A,B),
                                                modMul(C,D)));
                    }
                }
            }
        }
        // --- 3B) q-bit: p fixed, sum over q whose bitIdx==t, all h ---
        else if (varClass == 1) {
            for(int q = 0; q < d; ++q) {
                auto q_bits = intToBits(q, m);
                if (q_bits[bitIdx] != t) continue;
                for(int h = 0; h < d; ++h) {
                    auto h_bits = intToBits(h, m);
                    u64 A = evalMLE2D_bit(X,   ri_bits, rp_bits);
                    u64 B = evalMLE2D_bit(Wq,  rp_bits, h_bits);
                    u64 C = evalMLE2D_bit(X,   rj_bits, q_bits);
                    u64 D = evalMLE2D_bit(Wk,  q_bits,  h_bits);
                    localSum = modAdd(localSum,
                                     modMul(modMul(A,B),
                                            modMul(C,D)));
                }
            }
        }
        // --- 3C) h-bit: p,q fixed → single evaluation at h=t ---
        else {
            // rh_bits[bitIdx] already = t
            u64 A = evalMLE2D_bit(X,   ri_bits, rp_bits);
            u64 B = evalMLE2D_bit(Wq,  rp_bits, rh_bits);
            u64 C = evalMLE2D_bit(X,   rj_bits, rq_bits);
            u64 D = evalMLE2D_bit(Wk,  rq_bits,  rh_bits);
            localSum = modMul(modMul(A,B),
                              modMul(C,D));            
        }

        // 2c) store g(t)
        poly[t] = localSum;
    }

    // --- 4. Debug and return ---
    std::cout<<"Prover: poly for bit "<<phase<<" → ["
             <<poly[0]<<","<<poly[1]<<"]\n";
    return poly;
}



    // receive the verifier’s random challenge
    void recvChallenge(u64 r) {
        int bitIdx, varClass;
        if      (phase <   m) { bitIdx=phase; varClass=0; rp_bits[bitIdx]=r; }
        else if (phase < 2*m) { bitIdx=phase-m; varClass=1; rq_bits[bitIdx]=r; }
        else                  { bitIdx=phase-2*m; varClass=2; rh_bits[bitIdx]=r; }
        phase++;
    }
};

class Verifier {
public:
    int n,d,m,phase;
    std::vector<u64> ri_bits, rj_bits;
    Verifier(int _n, int _d)
      : n(_n), d(_d), m(std::log2(_d)), phase(0),
        ri_bits(std::log2(_n)), rj_bits(std::log2(_n))
    {}

    // pick random field bits for ri, rj
    void init() {
        for(auto &b: ri_bits) b = randomField();
        for(auto &b: rj_bits) b = randomField();
        std::cout<<"Verifier: init ri_bits,rj_bits\n";
    }

    u64 challenge() { return randomField(); }

    void run(Prover &P) {
    // 1) Send the random i/j bits to the prover
    P.ri_bits = ri_bits;
    P.rj_bits = rj_bits;

    // 2) Prover commits to Σ = MLE(rawS, ri_bits, rj_bits)
    u64 last = P.commit();
    std::cout << "Verifier: got Σ=" << last << "\n";

    // 3) Run the 3*m bitwise sum-check rounds
    for (phase = 0; phase < 3*m; ++phase) {
        // Prover sends the two evaluations g(0), g(1)
        auto poly = P.sendBitPoly();
        u64 s0 = poly[0];
        u64 s1 = poly[1];
        u64 sum = modAdd(s0, s1);

        // Debug output
        std::cout << "--- debugging bit " << phase << " ---\n"
                  << "  previous last = " << last << "\n"
                  << "  poly[0]       = " << s0   << "\n"
                  << "  poly[1]       = " << s1   << "\n"
                  << "  sum           = " << sum << "\n";

        // Consistency check: g(0) + g(1) == last
        if (sum != last) {
            std::cout << "Mismatch at bit " << phase << "\n";
            return;
        }

        // Sample random challenge r in the field
        u64 r = challenge();
        std::cout << "Verifier: bit" << phase
                  << " challenge r=" << r << "\n";

        // Reduce last = g(r) = (1-r)*g(0) + r*g(1)
        last = modAdd(modMul(modSub(1, r), s0),
                      modMul(r,           s1));

        // Send the challenge back to the prover
        P.recvChallenge(r);
    }

    // 4) Final spot‐check: evaluate G at fully fixed bits
    u64 A = evalMLE2D_bit(P.X,  ri_bits, P.rp_bits);
    u64 B = evalMLE2D_bit(P.Wq, P.rp_bits, P.rh_bits);
    u64 C = evalMLE2D_bit(P.X,  rj_bits, P.rq_bits);
    u64 D = evalMLE2D_bit(P.Wk, P.rq_bits, P.rh_bits);
    u64 rhs = modMul(modMul(A, B), modMul(C, D));

    // 5) Compare final last vs. rhs
    if (last != rhs) {
        std::cout << "Final check failed: " << last
                  << " != " << rhs << "\n";
    } else {
        std::cout << "✅ Sum-check verified *all* raw-attention entries!\n";
    }
    }
};

int main(){
    // example: n=4 (seq len), d=4 (model dim), both powers of two
    int n=4, d=4;
    Matrix X(n, std::vector<u64>(d)),
           Wq(d,std::vector<u64>(d)),
           Wk(d,std::vector<u64>(d));
    // random data
    for(auto &r:X)  for(auto &v:r) v = randomField();
    for(auto &r:Wq) for(auto &v:r) v = randomField();
    for(auto &r:Wk) for(auto &v:r) v = randomField();

    Verifier V(n,d);
    Prover   P(X,Wq,Wk,false);

    V.init();
    V.run(P);

    return 0;
}



// // sumcheck_attention.cpp
// // Full Sum-Check protocol in C++ verifying all raw-attention entries

// #include <iostream>
// #include <vector>
// #include <random>
// #include <cassert>
// #include <cstdint>

// using u64 = uint64_t;
// static constexpr u64 P = (1ULL<<61) - 1;

// // --- modular arithmetic ---
// inline u64 modAdd(u64 a, u64 b) {
//     u64 c = a + b;
//     if (c >= P) c -= P;
//     return c;
// }
// inline u64 modSub(u64 a, u64 b) {
//     return a >= b ? a - b : P + a - b;
// }
// inline u64 modMul(u64 a, u64 b) {
//     __uint128_t t = ( __uint128_t ) a * b;
//     return (u64)(t % P);
// }
// u64 modPow(u64 base, u64 exp) {
//     u64 res = 1 % P;
//     while (exp) {
//         if (exp & 1) res = modMul(res, base);
//         base = modMul(base, base);
//         exp >>= 1;
//     }
//     return res;
// }
// inline u64 modInv(u64 x) {
//     // P is prime → x^(P-2) mod P
//     return modPow(x, P-2);
// }

// // uniform field element
// u64 randomField(){
//     static std::mt19937_64 rng(std::random_device{}());
//     return rng() % P;
// }

// typedef std::vector<std::vector<u64>> Matrix;

// // 1D Lagrange interpolation from values at {0,1,...,d-1}
// u64 evalLagrange(const std::vector<u64>& vals, u64 r){
//     int d = (int)vals.size();
//     // 1) build denominators denom[i] = ∏_{k≠i}(i - k)
//     std::vector<u64> denom(d,1);
//     for(int i=0;i<d;++i){
//         for(int k=0;k<d;++k) if(k!=i)
//             denom[i] = modMul(denom[i], modSub((u64)i,(u64)k));
//     }
//     // 2) invert them
//     std::vector<u64> invDen(d);
//     for(int i=0;i<d;++i) invDen[i] = modInv(denom[i]);
//     // 3) interpolate
//     u64 res = 0;
//     for(int i=0;i<d;++i){
//         // numerator = ∏_{k≠i}(r - k)
//         u64 num = 1;
//         for(int k=0;k<d;++k) if(k!=i)
//             num = modMul(num, modSub(r,(u64)k));
//         // basis = numerator * invDen[i]
//         u64 li = modMul(num, invDen[i]);
//         // accumulate vals[i] * li
//         res = modAdd(res, modMul(vals[i], li));
//     }
//     return res;
// }

// // Multilinear extension of an N0×N1 table A at (r0,r1)
// u64 evalMLE2D(const Matrix& A, u64 r0, u64 r1){
//     int N0 = (int)A.size();
//     int N1 = (int)A[0].size();
//     // 1) build denominators for each row and column
//     std::vector<u64> denom0(N0,1), denom1(N1,1);
//     for(int i=0;i<N0;++i)
//       for(int k=0;k<N0;++k) if(k!=i)
//         denom0[i] = modMul(denom0[i], modSub((u64)i,(u64)k));
//     for(int j=0;j<N1;++j)
//       for(int k=0;k<N1;++k) if(k!=j)
//         denom1[j] = modMul(denom1[j], modSub((u64)j,(u64)k));
//     // 2) invert denominators
//     std::vector<u64> inv0(N0), inv1(N1);
//     for(int i=0;i<N0;++i) inv0[i] = modInv(denom0[i]);
//     for(int j=0;j<N1;++j) inv1[j] = modInv(denom1[j]);
//     // 3) sum up A[i][j] * ℓ_i(r0) * ℓ_j(r1)
//     u64 acc = 0;
//     for(int i=0;i<N0;++i){
//         // ℓ_i(r0) numerator
//         u64 Li = 1;
//         for(int k=0;k<N0;++k) if(k!=i)
//             Li = modMul(Li, modSub(r0,(u64)k));
//         Li = modMul(Li, inv0[i]);
//         for(int j=0;j<N1;++j){
//             // ℓ_j(r1) numerator
//             u64 Lj = 1;
//             for(int k=0;k<N1;++k) if(k!=j)
//                 Lj = modMul(Lj, modSub(r1,(u64)k));
//             Lj = modMul(Lj, inv1[j]);
//             acc = modAdd(acc, modMul(A[i][j], modMul(Li,Lj)));
//         }
//     }
//     return acc;
// }

// enum Round { P_ROUND, Q_ROUND, H_ROUND, DONE };

// class Prover {
// public:
//     const Matrix &X, &Wq, &Wk;
//     Matrix rawS;
//     u64 ri, rj, rp, rq;
//     Round rnd;
//     bool cheat;

//     Prover(const Matrix& X_, const Matrix& Wq_, const Matrix& Wk_, bool c=false)
//       : X(X_), Wq(Wq_), Wk(Wk_), rnd(P_ROUND), cheat(c) {}

//     // build full raw-attention S[i][j]
//     void computeRaw(){
//         int n = (int)X.size(), d = (int)Wq.size();
//         rawS.assign(n, std::vector<u64>(n,0));
//         for(int i=0;i<n;++i) for(int j=0;j<n;++j){
//             u64 s = 0;
//             for(int h=0;h<d;++h){
//                 u64 qi=0, kj=0;
//                 for(int p=0;p<d;++p) qi = modAdd(qi, modMul(X[i][p], Wq[p][h]));
//                 for(int q=0;q<d;++q) kj = modAdd(kj, modMul(X[j][q], Wk[q][h]));
//                 s = modAdd(s, modMul(qi, kj));
//             }
//             rawS[i][j] = s;
//         }
//         std::cout<<"Prover: computed raw-attention"
//                  << (cheat?" (cheat)":"")<<"\n";
//     }

//     void init(u64 _ri, u64 _rj){
//         ri = _ri; rj = _rj; rnd = P_ROUND;
//         computeRaw();
//         std::cout<<"Prover:init r_i,r_j=("<<ri<<","<<rj<<")\n";
//     }

//     // Σ = MLE(rawS, ri, rj)
//     u64 commit(){
//         u64 v = evalMLE2D(rawS, ri, rj);
//         if(cheat) v = modAdd(v,1);
//         std::cout<<"Prover: Σ="<<v<<"\n";
//         return v;
//     }

//     // send g_r(t) for current round r, as evaluations at t=0..d-1
//     std::vector<u64> sendUni(){
//         int d = (int)Wq.size();
//         std::vector<u64> poly(d, 0);
//         for(int t=0;t<d;++t){
//             u64 sum = 0;
//             if(rnd==P_ROUND){
//                 for(int q=0;q<d;++q) for(int h=0;h<d;++h){
//                     u64 a = evalMLE2D(X, ri, t);
//                     u64 b = evalMLE2D(Wq, t, h);
//                     u64 c = evalMLE2D(X, rj, q);
//                     u64 d2= evalMLE2D(Wk, q, h);
//                     sum = modAdd(sum, modMul(modMul(a,b), modMul(c,d2)));
//                 }
//             }
//             else if(rnd==Q_ROUND){
//                 for(int h=0;h<d;++h){
//                     u64 a = evalMLE2D(X, ri, rp);
//                     u64 b = evalMLE2D(Wq, rp, h);
//                     u64 c = evalMLE2D(X, rj, t);
//                     u64 d2= evalMLE2D(Wk, t, h);
//                     sum = modAdd(sum, modMul(modMul(a,b), modMul(c,d2)));
//                 }
//             }
//             else if(rnd==H_ROUND){
//                 u64 a = evalMLE2D(X, ri, rp);
//                 u64 b = evalMLE2D(Wq, rp, t);
//                 u64 c = evalMLE2D(X, rj, rq);
//                 u64 d2= evalMLE2D(Wk, rq, t);
//                 sum = modMul(modMul(a,b), modMul(c,d2));
//             }
//             poly[t] = sum;
//         }
//         std::cout<<"Prover: poly rnd="<<rnd<<"\n";
//         return poly;
//     }

//     void recvC(u64 c){
//         std::cout<<"Prover: recv "<<c<<" for rnd="<<rnd<<"\n";
//         if(rnd==P_ROUND) rp = c;
//         else if(rnd==Q_ROUND) rq = c;
//         if(rnd!=DONE) rnd = Round(rnd+1);
//     }
// };

// class Verifier {
// public:
//     u64 ri, rj;
//     Verifier() {}

//     void init(int n){
//         ri = randomField();
//         rj = randomField();
//         std::cout<<"Verifier: r_i,r_j=("<<ri<<","<<rj<<")\n";
//     }

//     u64 chall(){ return randomField(); }

//     void run(Prover &P){
//         P.init(ri,rj);
//         u64 last = P.commit();
//         std::cout<<"Verifier: got Σ="<<last<<"\n";

//         // store challenges for final spot-check
//         u64 rp=0, rq=0, rh=0;

//         // three rounds
//         for(int round=0; round<3; ++round){
//             auto poly = P.sendUni();
//             // 1) sum-check: sum poly[t] vs last
//             u64 sum=0;
//             for(auto &v: poly) sum=modAdd(sum,v);
//             if(sum!=last){
//                 std::cout<<"Mismatch round "<<round
//                          <<" (sum="<<sum<<" vs "<<last<<")\n";
//                 return;
//             }
//             std::cout<<"Round "<<round<<" ok\n";
//             // 2) sample challenge
//             u64 r = chall();
//             std::cout<<"Verifier: r"<<round<<"="<<r<<"\n";
//             P.recvC(r);
//             // save for final
//             if(round==0) rp=r;
//             if(round==1) rq=r;
//             if(round==2) rh=r;
//             // 3) interpolate g(r) and set last = g(r)
//             last = evalLagrange(poly, r);
//         }

//         // final spot-check: compare last vs direct MLE product
//         u64 a = evalMLE2D(P.X,  ri, rp);
//         u64 b = evalMLE2D(P.Wq, rp, rh);
//         u64 c = evalMLE2D(P.X,  rj, rq);
//         u64 d2= evalMLE2D(P.Wk, rq, rh);
//         u64 rhs = modMul(modMul(a,b), modMul(c,d2));

//         if(last != rhs)
//             std::cout<<"Final fail "<<last<<" != "<<rhs<<"\n";
//         else
//             std::cout<<"Sum-check verified all entries!\n";
//     }
// };

// int main(){
//     int n = 4, d = 4;
//     Matrix X(n, std::vector<u64>(d)),
//            Wq(d, std::vector<u64>(d)),
//            Wk(d, std::vector<u64>(d));
//     // random instance
//     for(auto &r:X)  for(auto &v:r)  v = randomField();
//     for(auto &r:Wq) for(auto &v:r)  v = randomField();
//     for(auto &r:Wk) for(auto &v:r)  v = randomField();

//     Verifier V;
//     std::cout<<"-- honest --\n";
//     Prover P1(X,Wq,Wk,false);
//     V.init(n);
//     V.run(P1);

//     // Uncomment to test cheating:
//     // std::cout<<"-- cheat --\n";
//     // Prover P2(X,Wq,Wk,true);
//     // V.init(n);
//     // V.run(P2);

//     return 0;
// }
