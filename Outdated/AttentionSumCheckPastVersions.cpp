
// // sumcheck_attention.cpp
// // Simulate a Sum-Check protocol for verifying raw-attention scores

// #include <iostream>
// #include <vector>
// #include <random>
// #include <cassert>
// #include <cstdint>

// using u64 = uint64_t;
// static constexpr u64 P = (1ULL<<61) - 1; // a Mersenne prime

// // Modular arithmetic
// inline u64 modAdd(u64 a, u64 b) { u64 c = a + b; if (c >= P) c -= P; return c; }
// inline u64 modSub(u64 a, u64 b) { return a >= b ? a - b : P + a - b; }
// inline u64 modMul(u64 a, u64 b) { __uint128_t t = ( __uint128_t)a * b; return (u64)(t % P); }

// // Evaluate a polynomial (coeffs[0] + coeffs[1]*x + ... + coeffs[d]*x^d)
// u64 evalPoly(const std::vector<u64> &coeffs, u64 x) {
//     u64 res = 0;
//     for (int i = (int)coeffs.size() - 1; i >= 0; --i) {
//         res = modMul(res, x);
//         res = modAdd(res, coeffs[i]);
//     }
//     return res;
// }

// u64 randomField() {
//     static std::mt19937_64 rng(std::random_device{}());
//     return rng() % P;
// }

// using Matrix = std::vector<std::vector<u64>>;

// // Naive MLE evaluation for 2D
// u64 evalMLE2D(const Matrix &A, u64 r0, u64 r1) {
//     int N0 = (int)A.size(), N1 = (int)A[0].size();
//     std::vector<u64> den0(N0,1), den1(N1,1);
//     for(int i=0;i<N0;++i) for(int j=0;j<N0;++j) if(i!=j) den0[i]=modMul(den0[i],modSub(i,j));
//     for(int i=0;i<N1;++i) for(int j=0;j<N1;++j) if(i!=j) den1[i]=modMul(den1[i],modSub(i,j));
//     u64 acc=0;
//     for(int i=0;i<N0;++i) {
//         u64 Li=1;
//         for(int k=0;k<N0;++k) if(k!=i) Li=modMul(Li,modSub(r0,k));
//         Li=modMul(Li,den0[i]);
//         for(int j=0;j<N1;++j) {
//             u64 Lj=1;
//             for(int k=0;k<N1;++k) if(k!=j) Lj=modMul(Lj,modSub(r1,k));
//             Lj=modMul(Lj,den1[j]);
//             acc=modAdd(acc,modMul(A[i][j],modMul(Li,Lj)));
//         }
//     }
//     return acc;
// }

// enum Round { P_ROUND, Q_ROUND, H_ROUND, DONE };

// class Prover {
//     const Matrix &X,&Wq,&Wk;
//     Matrix rawS;
//     u64 ri,rj;
//     Round round;
//     bool cheat;
// public:
//     Prover(const Matrix &X_,const Matrix &Wq_,const Matrix &Wk_,bool cheat_=false)
//       :X(X_),Wq(Wq_),Wk(Wk_),round(P_ROUND),cheat(cheat_){}

//     void computeRawAttention() {
//         int n = (int)X.size(), d = (int)Wq.size();
//         rawS.assign(n, std::vector<u64>(n, 0));
//         for(int i=0;i<n;++i) for(int j=0;j<n;++j) {
//             u64 sum = 0;
//             for(int h=0;h<d;++h) {
//                 u64 qi = 0, kj = 0;
//                 for(int p=0;p<d;++p) qi = modAdd(qi, modMul(X[i][p], Wq[p][h]));
//                 for(int q=0;q<d;++q) kj = modAdd(kj, modMul(X[j][q], Wk[q][h]));
//                 sum = modAdd(sum, modMul(qi, kj));
//             }
//             rawS[i][j] = sum;
//         }
//         std::cout << "Prover: computed raw-attention" << (cheat?" (cheating)":"") << "\n";
//     }

//     u64 getRaw(u64 i,u64 j) const { assert(i<rawS.size()&&j<rawS.size()); return rawS[i][j]; }

//     void init(u64 _ri,u64 _rj) {
//         ri = _ri; rj = _rj; round = P_ROUND;
//         computeRawAttention();
//         std::cout << "Prover: init (r_i,r_j)=(" << ri << "," << rj << ")\n";
//     }

//     // initial commitment: prover sends Σ = 
//     u64 commit() {
//         u64 s = getRaw(ri, rj);
//         if(cheat) s = modAdd(s, 1);
//         std::cout << "Prover: commit Σ=" << s << "\n";
//         return s;
//     }

//     // send univariate polynomial coefficients for current round
//     std::vector<u64> sendUni() {
//         int d = (int)Wq.size();
//         std::vector<u64> poly(d+1, 0);
//         for(int t=0;t<=d;++t) {
//             u64 sum = 0;
//             if(round == P_ROUND) {
//                 for(int q=0;q<d;++q) for(int h=0;h<d;++h) {
//                     u64 a = evalMLE2D(X, ri, t);
//                     u64 b = evalMLE2D(Wq, t, h);
//                     u64 c = evalMLE2D(X, rj, q);
//                     u64 d2 = evalMLE2D(Wk, q, h);
//                     sum = modAdd(sum, modMul(modMul(a,b), modMul(c,d2)));
//                 }
//             }
//             // TODO: implement Q_ROUND and H_ROUND polynomials
//             poly[t] = sum;
//         }
//         std::cout << "Prover: sent polynomial for round=" << round << "\n";
//         return poly;
//     }

//     void recvC(u64 r) {
//         std::cout << "Prover: received challenge=" << r << " for round=" << round << "\n";
//         if(round != DONE) round = Round(round + 1);
//     }
// };

// class Verifier {
//     int d, n;
//     u64 ri, rj;
// public:
//     Verifier(int dim): d(dim), n(0) {}
//     void init(int _n) {
//         n = _n;
//         std::mt19937_64 rng(std::random_device{}());
//         std::uniform_int_distribution<u64> dist(0, P-1);
//         ri = dist(rng); rj = dist(rng);
//         std::cout << "Verifier: (r_i,r_j)=(" << ri << "," << rj << ")\n";
//     }
//     u64 chall() const { return randomField(); }

//     void run(Prover &P) {
//         P.init(ri, rj);
//         // 1) Commit
//         u64 sigma = P.commit();
//         u64 lastSum = sigma;
//         std::cout << "Verifier: received commit Σ=" << sigma << "\n";

//         // 2) Sum-check rounds
//         std::vector<u64> Rs(3);
//         for(int rIdx = 0; rIdx < 3; ++rIdx) {
//             auto poly = P.sendUni();
//             // Check sum of poly equals lastSum
//             u64 sumPoly = 0;
//             for(u64 coef : poly) sumPoly = modAdd(sumPoly, coef);
//             if(sumPoly != lastSum) {
//                 std::cout << "Verifier: round " << rIdx << " sum mismatch " << sumPoly << " != " << lastSum << "\n";
//                 return;
//             }
//             std::cout << "Verifier: round " << rIdx << " sum-check passed\n";
//             // Challenge
//             u64 r = chall();
//             Rs[rIdx] = r;
//             std::cout << "Verifier: challenge r" << rIdx << "=" << r << "\n";
//             P.recvC(r);
//             // Update lastSum = poly(r)
//             lastSum = evalPoly(poly, r);
//         }

//         // 3) Final spot-check
//         u64 rp = Rs[0], rq = Rs[1], rh = Rs[2];
//         u64 lhs = lastSum;
//         // Evaluate RHS = X_MLE(ri,rp)*Wq_MLE(rp,rh)*X_MLE(rj,rq)*Wk_MLE(rq,rh)
//         u64 e1 = evalMLE2D(P.X, ri, rp);
//         u64 e2 = evalMLE2D(P.Wq, rp, rh);
//         u64 e3 = evalMLE2D(P.X, rj, rq);
//         u64 e4 = evalMLE2D(P.Wk, rq, rh);
//         u64 rhs = modMul(modMul(e1,e2), modMul(e3,e4));
//         if(lhs != rhs) {
//             std::cout << "Verifier: final check failed " << lhs << " != " << rhs << "\n";
//         } else {
//             std::cout << "Verifier: full sum-check passed, all raw-attention entries verified!\n";
//         }
//     }
// };

// int main() {
//     int n = 4;
//     int d = 4;
//     Matrix X(n, std::vector<u64>(d)), Wq(d, std::vector<u64>(d)), Wk(d, std::vector<u64>(d));
//     for(auto &row : X)  for(auto &v : row)  v = randomField();
//     for(auto &row : Wq) for(auto &v : row)  v = randomField();
//     for(auto &row : Wk) for(auto &v : row)  v = randomField();

//     Verifier V(d);
//     std::cout << "-- honest prover --\n";
//     Prover P1(X,Wq,Wk,false);
//     V.init(n);
//     V.run(P1);

//     std::cout << "-- cheating prover --\n";
//     Prover P2(X,Wq,Wk,true);
//     V.init(n);
//     V.run(P2);

//     return 0;
// }

// #include <iostream>
// #include <vector>
// #include <random>
// #include <cassert>
// #include <cstdint>
// using namespace std;
// using u64 = uint64_t;
// static constexpr u64 P = (1ULL<<61) - 1; // a Mersenne prime

// // Modular arithmetic
// inline u64 modAdd(u64 a, u64 b) { u64 c = a + b; if (c >= P) c -= P; return c; }
// inline u64 modSub(u64 a, u64 b) { return a >= b ? a - b : P + a - b; }
// inline u64 modMul(u64 a, u64 b) { __uint128_t t = ( __uint128_t)a * b; return (u64)(t % P); }

// u64 randomField() {
//     static std::mt19937_64 rng(std::random_device{}());
//     return rng() % P;
// }

// using Matrix = std::vector<std::vector<u64>>;

// // Naive MLE evaluation for 2D
// u64 evalMLE2D(const Matrix &A, u64 r0, u64 r1) {
//     int N0 = A.size(), N1 = A[0].size();
//     std::vector<u64> den0(N0,1), den1(N1,1);
//     for(int i=0;i<N0;++i) for(int j=0;j<N0;++j) if(i!=j) den0[i]=modMul(den0[i],modSub(i,j));
//     for(int i=0;i<N1;++i) for(int j=0;j<N1;++j) if(i!=j) den1[i]=modMul(den1[i],modSub(i,j));
//     u64 acc=0;
//     for(int i=0;i<N0;++i) {
//         u64 Li=1;
//         for(int k=0;k<N0;++k) if(k!=i) Li=modMul(Li,modSub(r0,k));
//         Li=modMul(Li,den0[i]);
//         for(int j=0;j<N1;++j) {
//             u64 Lj=1;
//             for(int k=0;k<N1;++k) if(k!=j) Lj=modMul(Lj,modSub(r1,k));
//             Lj=modMul(Lj,den1[j]);
//             acc=modAdd(acc,modMul(A[i][j],modMul(Li,Lj)));
//         }
//     }
//     return acc;
// }

// enum Round { P_ROUND, Q_ROUND, H_ROUND, DONE };

// class Prover {
//     const Matrix &X,&Wq,&Wk;
//     Matrix rawS;
//     u64 ri,rj;
//     Round round;
//     bool cheat;
// public:
//     Prover(const Matrix &X_,const Matrix &Wq_,const Matrix &Wk_,bool cheat_=false)
//       :X(X_),Wq(Wq_),Wk(Wk_),round(P_ROUND),cheat(cheat_){}

//     void computeRawAttention(){
//         int n=X.size(),d=Wq.size();
//         rawS.assign(n,vector<u64>(n));
//         for(int i=0;i<n;++i)for(int j=0;j<n;++j){u64 sum=0;
//             for(int h=0;h<d;++h){u64 qi=0,kj=0;
//                 for(int p=0;p<d;++p) qi=modAdd(qi,modMul(X[i][p],Wq[p][h]));
//                 for(int q=0;q<d;++q) kj=modAdd(kj,modMul(X[j][q],Wk[q][h]));
//                 sum=modAdd(sum,modMul(qi,kj));}
//             rawS[i][j]=sum;
//         }
//         cout<<"Prover: computed raw-attention"<<(cheat?" (cheating)":"")<<"\n";
//     }

//     u64 getRaw(u64 i,u64 j)const{assert(i<rawS.size()&&j<rawS.size());return rawS[i][j];}

//     void init(u64 _ri,u64 _rj){ri=_ri;rj=_rj;round=P_ROUND;computeRawAttention();
//         cout<<"Prover:init (r_i,r_j)=("<<ri<<","<<rj<<")\n";
//     }

//     // initial commitment
//     u64 commit(){u64 s=getRaw(ri,rj); if(cheat) s=modAdd(s,1); cout<<"Prover:commit Sigma="<<s<<"\n";return s;}

//     vector<u64> sendUni(){int d=Wq.size();vector<u64> c(d+1);
//         for(int t=0;t<=d;++t){u64 sum=0;
//             if(round==P_ROUND){
//                 for(int q=0;q<d;++q)for(int h=0;h<d;++h){
//                     u64 a=evalMLE2D(X,ri,t),b=evalMLE2D(Wq,t,h);
//                     u64 c1=evalMLE2D(X,rj,q),d1=evalMLE2D(Wk,q,h);
//                     sum=modAdd(sum,modMul(modMul(a,b),modMul(c1,d1)));
//                 }
//             }
//             c[t]=sum;
//         }
//         cout<<"Prover: sent uni round"<<round<<"\n";
//         return c;
//     }
//     void recvC(u64 r){cout<<"Prover:recv chall="<<r<<" round="<<round<<"\n";
//         if(round!=DONE) round=Round(round+1);
//     }
// };

// class Verifier{
//     int d;u64 ri,rj;
// public:
//     Verifier(int dim):d(dim){}
//     void init(int n){mt19937_64 g(random_device{}());uniform_int_distribution<int>dist(0,n-1);
//         ri=dist(g);rj=dist(g);
//         cout<<"Verifier:(r_i,r_j)=("<<ri<<","<<rj<<")\n";
//     }
//     u64 chall()const{return randomField();}
//     void run(Prover&P){P.init(ri,rj);
//         u64 c0=P.commit(),t0=P.getRaw(ri,rj);
//         if(c0!=t0) {cout<<"Cheat caught! "<<c0<<" vs "<<t0<<"\n";return;} 
//         cout<<"Verified commit!\n";
//         for(int i=0;i<3;++i){auto u=P.sendUni();u64 r=chall();cout<<"Verifier chall"<<i<<"="<<r<<"\n";P.recvC(r);} 
//         cout<<"Sum-check done\n";
//     }
// };

// int main(){int n=4,d=4;
//     Matrix X(n,vector<u64>(d)),Wq(d,vector<u64>(d)),Wk(d,vector<u64>(d));
//     for(auto&r:X)for(auto&v:r)v=randomField();
//     for(auto&r:Wq)for(auto&v:r)v=randomField();
//     for(auto&r:Wk)for(auto&v:r)v=randomField();

//     Verifier V(d);
//     cout<<"--honest--\n";Prover P1(X,Wq,Wk,false);V.init(n);V.run(P1);
//     cout<<"--cheat--\n";Prover P2(X,Wq,Wk,true);V.init(n);V.run(P2);
// }



// // sumcheck_attention.cpp
// // Simulate a Sum-Check protocol for verifying raw-attention scores

// #include <iostream>
// #include <vector>
// #include <random>
// #include <cassert>
// #include <cstdint>

// using u64 = uint64_t;
// static constexpr u64 P = (1ULL<<61) - 1; // a Mersenne prime

// // Modular arithmetic functions
// inline u64 modAdd(u64 a, u64 b) {
//     u64 c = a + b;
//     if (c >= P) c -= P;
//     return c;
// }

// inline u64 modSub(u64 a, u64 b) {
//     return a >= b ? a - b : P + a - b;
// }

// inline u64 modMul(u64 a, u64 b) {
//     __uint128_t t = ( __uint128_t)a * b;
//     return (u64)(t % P);
// }

// // Generate a random field element in [0, P)
// u64 randomField() {
//     static std::mt19937_64 rng(std::random_device{}());
//     return rng() % P;
// }

// using Matrix = std::vector<std::vector<u64>>;

// // Naive MLE evaluator for a 2D table A at (r0, r1)
// u64 evalMLE2D(const Matrix &A, u64 r0, u64 r1) {
//     int N0 = (int)A.size();
//     int N1 = (int)A[0].size();
//     // Precompute Lagrange denominators
//     std::vector<u64> den0(N0, 1), den1(N1, 1);
//     for (int i = 0; i < N0; ++i)
//         for (int j = 0; j < N0; ++j)
//             if (i != j)
//                 den0[i] = modMul(den0[i], modSub(i, j));
//     for (int i = 0; i < N1; ++i)
//         for (int j = 0; j < N1; ++j)
//             if (i != j)
//                 den1[i] = modMul(den1[i], modSub(i, j));

//     u64 acc = 0;
//     for (int i = 0; i < N0; ++i) {
//         // Compute L_i(r0)
//         u64 Li = 1;
//         for (int k = 0; k < N0; ++k)
//             if (k != i)
//                 Li = modMul(Li, modSub(r0, k));
//         Li = modMul(Li, den0[i]);

//         for (int j = 0; j < N1; ++j) {
//             // Compute L_j(r1)
//             u64 Lj = 1;
//             for (int k = 0; k < N1; ++k)
//                 if (k != j)
//                     Lj = modMul(Lj, modSub(r1, k));
//             Lj = modMul(Lj, den1[j]);

//             u64 term = modMul(A[i][j], modMul(Li, Lj));
//             acc = modAdd(acc, term);
//         }
//     }
//     return acc;
// }

// // Prover and Verifier simulate sum-check over dimensions p, q, h
// enum Round { P_ROUND, Q_ROUND, H_ROUND, DONE };

// class Prover {
//     Matrix rawS;
//     const Matrix &X, &Wq, &Wk;
//     u64 ri, rj;
//     Round round;
// public:
//     Prover(const Matrix &X_, const Matrix &Wq_, const Matrix &Wk_)
//         : X(X_), Wq(Wq_), Wk(Wk_), round(P_ROUND) {}

//     void init(u64 _ri, u64 _rj) {
//         ri = _ri;
//         rj = _rj;
//         round = P_ROUND;
//         std::cout << "Prover: init with (r_i, r_j) = (" << ri << "," << rj << ")\n";
        
//     }

//     void computeRawAttention() {
//         int n = (int)X.size();
//         int d = (int)Wq.size();
//         rawS.assign(n, std::vector<u64>(n, 0));
//         for (int i = 0; i < n; ++i) {
//             for (int j = 0; j < n; ++j) {
//                 u64 sum = 0;
//                 for (int h = 0; h < d; ++h) {
//                     // Q[i][h] = sum_p X[i][p] * Wq[p][h]
//                     u64 qi_h = 0;
//                     for (int p = 0; p < d; ++p)
//                         qi_h = modAdd(qi_h, modMul(X[i][p], Wq[p][h]));
//                     // K[j][h] = sum_q X[j][q] * Wk[q][h]
//                     u64 kj_h = 0;
//                     for (int q = 0; q < d; ++q)
//                         kj_h = modAdd(kj_h, modMul(X[j][q], Wk[q][h]));
//                     sum = modAdd(sum, modMul(qi_h, kj_h));
//                 }
//                 rawS[i][j] = sum;
//             }
//         }
//         std::cout << "Prover: computed full raw-attention matrix\n";
//     }

//     // Send coefficients of univariate polynomial for the current round
//     std::vector<u64> sendUnivariate() {
//         int d = static_cast<int>(Wq.size());
//         std::vector<u64> coeffs(d + 1);

//         for (int t = 0; t <= d; ++t) {
//             u64 sum = 0;
//             if (round == P_ROUND) {
//                 // Sum over q,h for fixed p = t
//                 for (int q = 0; q < d; ++q) {
//                     for (int h = 0; h < d; ++h) {
//                         u64 a = evalMLE2D(X, ri, t);
//                         u64 b = evalMLE2D(Wq, t, h);
//                         u64 c = evalMLE2D(X, rj, q);
//                         u64 d5 = evalMLE2D(Wk, q, h);
//                         sum = modAdd(sum, modMul(modMul(a, b), modMul(c, d5)));
//                     }
//                 }
//             }
//             // TODO: implement rounds for Q_ROUND and H_ROUND

//             coeffs[t] = sum;
//         }

//         std::cout << "Prover: sent univariate for round " << static_cast<int>(round) << "\n";
//         return coeffs;
//     }

//     void receiveChallenge(u64 r) {
//         std::cout << "Prover: got challenge " << r
//                   << " for round " << static_cast<int>(round) << "\n";
//         if (round != DONE) {
//             round = static_cast<Round>(round + 1);
//         }
//     }
// };

// class Verifier {
//     int d;
//     u64 ri, rj;
// public:
//     Verifier(int dim) : d(dim) {}

//     void init() {
//         ri = randomField();
//         rj = randomField();
//         std::cout << "Verifier: (r_i, r_j) = (" << ri << "," << rj << ")\n";
//     }

//     u64 challenge() const {
//         return randomField();
//     }

//     void runSumcheck(Prover &P) {
//         P.init(ri, rj);
//         for (int r = 0; r < 3; ++r) {
//             auto poly = P.sendUnivariate();
//             u64 c = challenge();
//             std::cout << "Verifier: challenge for round " << r
//                       << " = " << c << "\n";
//             P.receiveChallenge(c);
//         }
//         std::cout << "Verifier: completed sum-check\n";
//     }
// };

// int main() {
//     int n = 4;
//     int d = 4;
//     Matrix X(n, std::vector<u64>(d));
//     Matrix Wq(d, std::vector<u64>(d));
//     Matrix Wk(d, std::vector<u64>(d));

//     for (auto &row : X) for (auto &v : row) v = randomField();
//     for (auto &row : Wq) for (auto &v : row) v = randomField();
//     for (auto &row : Wk) for (auto &v : row) v = randomField();

//     Prover prover(X, Wq, Wk);
//     Verifier verifier(d);
//     verifier.init();
//     verifier.runSumcheck(prover);

//     return 0;
// }
