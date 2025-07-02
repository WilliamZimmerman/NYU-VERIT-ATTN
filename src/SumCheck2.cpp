#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <random>
#include <iomanip>
#include <stdexcept>

// --- HELPER FUNCTION FOR MODULAR EXPONENTIATION (Needed for division) ---
long long power(long long base, long long exp, long long mod) {
    long long res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp % 2 == 1) res = (static_cast<__int128>(res) * base) % mod;
        base = (static_cast<__int128>(base) * base) % mod;
        exp /= 2;
    }
    return res;
}

// --- CORE FIELD AND MATRIX CLASSES ---
class FieldElement {
public:
    static const long long PRIME = 2147483647; 
    long long val;
    FieldElement(long long v = 0) : val(((v % PRIME) + PRIME) % PRIME) {}
    FieldElement operator+(const FieldElement& other) const { return FieldElement((val + other.val) % PRIME); }
    FieldElement operator-(const FieldElement& other) const { return FieldElement((val - other.val + PRIME) % PRIME); }
    FieldElement operator*(const FieldElement& other) const { return FieldElement((static_cast<__int128>(val) * other.val) % PRIME); }
    FieldElement inverse() const { return FieldElement(power(val, PRIME - 2, PRIME)); }
    FieldElement operator/(const FieldElement& other) const { return *this * other.inverse(); }
    bool operator==(const FieldElement& other) const { return val == other.val; }
    friend std::ostream& operator<<(std::ostream& os, const FieldElement& fe) { return os << fe.val; }
};

class MatrixMLE {
private:
    std::vector<std::vector<FieldElement>> data;
    int rows, cols, log_rows, log_cols, total_vars;
public:
    MatrixMLE(const std::vector<std::vector<FieldElement>>& matrix) : data(matrix) {
        rows = matrix.size();
        cols = matrix.empty() ? 0 : matrix[0].size();
        log_rows = (rows <= 1) ? 1 : (int)ceil(log2(rows));
        log_cols = (cols <= 1) ? 1 : (int)ceil(log2(cols));
        total_vars = log_rows + log_cols;
    }
    FieldElement evaluate(const std::vector<FieldElement>& point) const {
        assert(point.size() == total_vars);
        FieldElement result(0);
        int max_row_vals = 1 << log_rows;
        int max_col_vals = 1 << log_cols;
        for (int row_bits = 0; row_bits < max_row_vals; row_bits++) {
            for (int col_bits = 0; col_bits < max_col_vals; col_bits++) {
                if (row_bits < rows && col_bits < cols) {
                    FieldElement indicator(1);
                    for (int bit = 0; bit < log_rows; bit++) {
                        indicator = indicator * ((FieldElement(1) - point[bit]) * FieldElement(1 - ((row_bits >> bit) & 1)) + point[bit] * FieldElement((row_bits >> bit) & 1));
                    }
                    for (int bit = 0; bit < log_cols; bit++) {
                        indicator = indicator * ((FieldElement(1) - point[log_rows + bit]) * FieldElement(1 - ((col_bits >> bit) & 1)) + point[log_rows + bit] * FieldElement((col_bits >> bit) & 1));
                    }
                    result = result + data[row_bits][col_bits] * indicator;
                }
            }
        }
        return result;
    }
    int get_total_vars() const { return total_vars; }
};

// --- PROTOCOL MESSAGE AND ACTOR CLASSES ---
struct ProverMessage { std::vector<FieldElement> coeffs; };
struct VerifierMessage { FieldElement challenge; };

class ThalerAttentionProver {
private:
    MatrixMLE X_mle, Q_mle, K_mle;
    int n, d, d_k, sumcheck_vars;
    std::vector<FieldElement> r_i, r_j, sumcheck_challenges;
public:
    ThalerAttentionProver(const std::vector<std::vector<FieldElement>>& X, const std::vector<std::vector<FieldElement>>& Q, const std::vector<std::vector<FieldElement>>& K)
        : X_mle(X), Q_mle(Q), K_mle(K) {
        n = X.size(); d = X[0].size(); d_k = Q[0].size();
        int log_d = (d <= 1) ? 1 : (int)ceil(log2(d));
        int log_d_k = (d_k <= 1) ? 1 : (int)ceil(log2(d_k));
        sumcheck_vars = 2 * log_d + log_d_k;
    }
    void setup_random_challenges() {
        std::random_device rd; std::mt19937 gen(rd());
        std::uniform_int_distribution<long long> dist(0, FieldElement::PRIME - 1);
        int log_n = (n <= 1) ? 1 : (int)ceil(log2(n));
        r_i.clear(); r_j.clear();
        for (int bit = 0; bit < log_n; bit++) {
            r_i.push_back(FieldElement(dist(gen))); r_j.push_back(FieldElement(dist(gen)));
        }
    }
    FieldElement compute_claimed_sum() {
        FieldElement sum(0); int log_d = (d <= 1) ? 1 : (int)ceil(log2(d)); int log_d_k = (d_k <= 1) ? 1 : (int)ceil(log2(d_k));
        for (int p = 0; p < d; p++) for (int q = 0; q < d; q++) for (int h = 0; h < d_k; h++) {
            std::vector<FieldElement> pt;
            for (int bit = 0; bit < log_d; bit++) pt.push_back(FieldElement((p >> bit) & 1));
            for (int bit = 0; bit < log_d; bit++) pt.push_back(FieldElement((q >> bit) & 1));
            for (int bit = 0; bit < log_d_k; bit++) pt.push_back(FieldElement((h >> bit) & 1));
            sum = sum + evaluate_reduced_polynomial(pt);
        }
        return sum;
    }
    FieldElement evaluate_reduced_polynomial(const std::vector<FieldElement>& pqh_point) const {
        assert(pqh_point.size() == sumcheck_vars); int log_d = (d <= 1) ? 1 : (int)ceil(log2(d)); int log_d_k = (d_k <= 1) ? 1 : (int)ceil(log2(d_k));
        std::vector<FieldElement> p(pqh_point.begin(), pqh_point.begin() + log_d), q(pqh_point.begin() + log_d, pqh_point.begin() + 2 * log_d), h(pqh_point.begin() + 2 * log_d, pqh_point.end());
        std::vector<FieldElement> xip = r_i; xip.insert(xip.end(), p.begin(), p.end());
        std::vector<FieldElement> qph = p; qph.insert(qph.end(), h.begin(), h.end());
        std::vector<FieldElement> xjq = r_j; xjq.insert(xjq.end(), q.begin(), q.end());
        std::vector<FieldElement> kqh = q; kqh.insert(kqh.end(), h.begin(), h.end());
        return X_mle.evaluate(xip) * Q_mle.evaluate(qph) * X_mle.evaluate(xjq) * K_mle.evaluate(kqh);
    }
    
   
    ProverMessage compute_sumcheck_round(int round) {
        const int degree = 2; std::vector<FieldElement> evaluations(degree + 1, 0);
        for (int eval_point = 0; eval_point <= degree; ++eval_point) {
            FieldElement current_sum(0); int rem_vars = sumcheck_vars - round - 1;
            for (long long mask = 0; mask < (1LL << rem_vars); ++mask) {
                std::vector<FieldElement> point = sumcheck_challenges;
                point.push_back(FieldElement(eval_point));
                for (int bit = 0; bit < rem_vars; ++bit) point.push_back(FieldElement((mask >> bit) & 1));
                current_sum = current_sum + evaluate_reduced_polynomial(point);
            }
            evaluations[eval_point] = current_sum;
        }
        std::cout << "  Prover: Computed h(" << round + 1 << ", 0) = " << evaluations[0] << std::endl;
        std::cout << "  Prover: Computed h(" << round + 1 << ", 1) = " << evaluations[1] << std::endl;
        std::cout << "  Prover: Computed h(" << round + 1 << ", 2) = " << evaluations[2] << " (for interpolation)" << std::endl;

        FieldElement v0 = evaluations[0], v1 = evaluations[1], v2 = evaluations[2], two(2);
        FieldElement c0 = v0, c2 = (v2 - v1 * two + v0) / two, c1 = v1 - v0 - c2;
        std::cout << "  Prover: Calculated coefficients {c0=" << c0 << ", c1=" << c1 << ", c2=" << c2 << "}" << std::endl;
        return {{c0, c1, c2}};
    }
    void receive_sumcheck_challenge(FieldElement c) { sumcheck_challenges.push_back(c); }
    FieldElement final_evaluation() const { return evaluate_reduced_polynomial(sumcheck_challenges); }
    int get_sumcheck_vars() const { return sumcheck_vars; }
};

class ThalerAttentionVerifier {
private:
    FieldElement expected_value; std::mt19937 rng;
public:
    ThalerAttentionVerifier() : rng(std::random_device{}()) {}
    void receive_claim(FieldElement claim) { expected_value = claim; }
    
    // --- MODIFIED with print statements ---
    VerifierMessage process_sumcheck_round(const ProverMessage& msg) {
        assert(msg.coeffs.size() == 3);
        FieldElement c0 = msg.coeffs[0], c1 = msg.coeffs[1], c2 = msg.coeffs[2];
        
        FieldElement h0 = c0;
        FieldElement h1 = c0 + c1 + c2;
        FieldElement current_sum = h0 + h1;
        
        std::cout << "  Verifier: Received coefficients. Checking sum..." << std::endl;
        std::cout << "  Verifier:   Expected sum = " << expected_value << std::endl;
        std::cout << "  Verifier:   Calculated sum (from coeffs) h(0)+h(1) = " << current_sum << std::endl;

        if (!(current_sum == expected_value)) { throw std::runtime_error("Verification failed: Sum check mismatch."); }
        std::cout << "  Verifier: ✅ Sum check PASSED." << std::endl;

        std::uniform_int_distribution<long long> dist(1, FieldElement::PRIME - 1);
        FieldElement challenge(dist(rng));
        
        expected_value = c0 + c1 * challenge + c2 * challenge * challenge;
        std::cout << "  Verifier: Generated challenge r = " << challenge << std::endl;
        std::cout << "  Verifier: New expected value for next round is h(r) = " << expected_value << std::endl;
        return {challenge};
    }
    
    // --- MODIFIED with print statements ---
    bool final_check(FieldElement prover_final_eval) {
        std::cout << "  Verifier: Received Prover's final evaluation = " << prover_final_eval << std::endl;
        std::cout << "  Verifier: My final expected value = " << expected_value << std::endl;
        bool success = (prover_final_eval == expected_value);
        if (success) { std::cout << "  Verifier: ✅ Final check PASSED." << std::endl; }
        else { std::cout << "  Verifier: ❌ Final check FAILED." << std::endl; }
        return success;
    }
};

// --- HELPER AND MAIN DRIVER ---
std::vector<std::vector<FieldElement>> create_random_matrix(int r, int c, std::mt19937& rng) {
    std::vector<std::vector<FieldElement>> m(r, std::vector<FieldElement>(c));
    std::uniform_int_distribution<long long> dist(0, 100);
    for (int i = 0; i < r; ++i) for (int j = 0; j < c; ++j) m[i][j] = FieldElement(dist(rng));
    return m;
}

int main() {
    try {
        const int n = 16, d = 16, d_k = 16; // Small 2x2 case for a clear 3-round trace

        std::cout << "THALER-STYLE ATTENTION SUMCHECK PROTOCOL (VERBOSE TRACE)" << std::endl;
        std::cout << "Running with " << n << "x" << d << " and " << d << "x" << d_k << " matrices." << std::endl;
        std::cout << "==========================================================" << std::endl;

        std::mt19937 rng(std::random_device{}());
        auto X = create_random_matrix(n, d, rng), Q = create_random_matrix(d, d_k, rng), K = create_random_matrix(d, d_k, rng);
        
        ThalerAttentionProver prover(X, Q, K);
        ThalerAttentionVerifier verifier;

        prover.setup_random_challenges();
        
        std::cout << "\n--- Phase 2: Prover Computes Claim ---" << std::endl;
        FieldElement claimed_sum = prover.compute_claimed_sum();
        verifier.receive_claim(claimed_sum);
        std::cout << "Prover -> Verifier: My claimed total sum is " << claimed_sum << std::endl;
        
        int num_rounds = prover.get_sumcheck_vars();
        
        for (int round = 0; round < num_rounds; round++) {
            std::cout << "\n------------------ Round " << round + 1 << "/" << num_rounds << " ------------------" << std::endl;
            ProverMessage msg = prover.compute_sumcheck_round(round);
            std::cout << "Prover -> Verifier: Sending polynomial for this round." << std::endl;
            VerifierMessage response = verifier.process_sumcheck_round(msg);
            std::cout << "Verifier -> Prover: Sending challenge for next round." << std::endl;
            prover.receive_sumcheck_challenge(response.challenge);
        }
        
        std::cout << "\n------------------ Final Check ------------------" << std::endl;
        FieldElement final_prover_eval = prover.final_evaluation();
        bool success = verifier.final_check(final_prover_eval);

        std::cout << "\n==========================================================" << std::endl;
        if (success) { std::cout << "✅ PROTOCOL SUCCESSFUL: Verification passed!" << std::endl; }
        else { std::cout << "❌ PROTOCOL FAILED: Verification did not pass." << std::endl; }
        std::cout << "==========================================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error during protocol: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

// MALICIOUS PROVER CLASS 

// class MaliciousProver : public ThalerAttentionProver {
// public:
//     // Inherit the constructor
//     using ThalerAttentionProver::ThalerAttentionProver;

//     // Override the round computation to introduce a bug
//     ProverMessage compute_sumcheck_round(int round, bool be_malicious = false) {
//         // Get the honest-prover's result first
//         ProverMessage honest_message = ThalerAttentionProver::compute_sumcheck_round(round);
        
//         if (be_malicious) {
//             std::cout << "\n  >> MALICIOUS PROVER: Tampering with coefficients for this round! <<\n" << std::endl;
//             // Introduce a simple bug: add 1 to the constant coefficient
//             honest_message.coeffs[0] = honest_message.coeffs[0] + FieldElement(1);
//         }
        
//         return honest_message;
//     }
// };


// void test_malicious_round() {
//     std::cout << "\n\n--- TEST 1: MALICIOUS PROVER (BAD ROUND POLYNOMIAL) ---\n";
//     const int n = 2, d = 2, d_k = 2;
//     std::mt19937 rng(123); // Use a fixed seed for reproducible tests
//     auto X = create_random_matrix(n, d, rng), Q = create_random_matrix(d, d_k, rng), K = create_random_matrix(d, d_k, rng);

//     MaliciousProver prover(X, Q, K); // Use the malicious prover
//     ThalerAttentionVerifier verifier;

//     prover.setup_random_challenges();
//     FieldElement claimed_sum = prover.compute_claimed_sum();
//     verifier.receive_claim(claimed_sum);

//     try {
//         // Run first round honestly
//         ProverMessage msg1 = prover.compute_sumcheck_round(0, false);
//         VerifierMessage resp1 = verifier.process_sumcheck_round(msg1);
//         prover.receive_sumcheck_challenge(resp1.challenge);

//         // Run second round maliciously
//         ProverMessage msg2 = prover.compute_sumcheck_round(1, true); // be_malicious = true
//         verifier.process_sumcheck_round(msg2); // This line should throw an exception

//         std::cout << "  [TEST FAILED]: Verifier did not detect the malicious prover.\n";

//     } catch (const std::exception& e) {
//         std::cout << "  [TEST PASSED]: Verifier successfully caught the malicious prover.\n";
//         std::cout << "  Verifier's error message: " << e.what() << "\n";
//     }
// }


// void test_bad_initial_claim() {
//     std::cout << "\n\n--- TEST 2: MALICIOUS PROVER (BAD INITIAL CLAIM) ---\n";
//     const int n = 2, d = 2, d_k = 2;
//     std::mt19937 rng(456);
//     auto X = create_random_matrix(n, d, rng), Q = create_random_matrix(d, d_k, rng), K = create_random_matrix(d, d_k, rng);
    
//     ThalerAttentionProver prover(X, Q, K);
//     ThalerAttentionVerifier verifier;
    
//     prover.setup_random_challenges();

//     // Prover computes the honest sum, but then lies.
//     FieldElement honest_sum = prover.compute_claimed_sum();
//     FieldElement malicious_claim = honest_sum + FieldElement(1);
    
//     std::cout << "  Prover computed honest sum " << honest_sum << ", but claims " << malicious_claim << ".\n";
//     verifier.receive_claim(malicious_claim);

//     try {
//         // Prover generates the first round message based on the *honest* computation
//         ProverMessage msg = prover.compute_sumcheck_round(0);
//         verifier.process_sumcheck_round(msg); // This should fail

//         std::cout << "  [TEST FAILED]: Verifier did not detect the bad initial claim.\n";
//     } catch (const std::exception& e) {
//         std::cout << "  [TEST PASSED]: Verifier successfully caught the bad initial claim.\n";
//         std::cout << "  Verifier's error message: " << e.what() << "\n";
//     }
// }


// void test_bad_final_evaluation() {
//     std::cout << "\n\n--- TEST 3: MALICIOUS PROVER (BAD FINAL EVALUATION) ---\n";
//     const int n = 2, d = 2, d_k = 2;
//     std::mt19937 rng(789);
//     auto X = create_random_matrix(n, d, rng), Q = create_random_matrix(d, d_k, rng), K = create_random_matrix(d, d_k, rng);
    
//     ThalerAttentionProver prover(X, Q, K);
//     ThalerAttentionVerifier verifier;
    
//     prover.setup_random_challenges();
//     FieldElement claimed_sum = prover.compute_claimed_sum();
//     verifier.receive_claim(claimed_sum);

//     // Run all rounds honestly
//     int num_rounds = prover.get_sumcheck_vars();
//     for (int round = 0; round < num_rounds; ++round) {
//         ProverMessage msg = prover.compute_sumcheck_round(round);
//         VerifierMessage resp = verifier.process_sumcheck_round(msg);
//         prover.receive_sumcheck_challenge(resp.challenge);
//     }

//     // Prover computes the honest final value, but then lies.
//     FieldElement honest_final_eval = prover.final_evaluation();
//     FieldElement malicious_final_eval = honest_final_eval + FieldElement(1);

//     std::cout << "  Prover computed honest final value " << honest_final_eval << ", but sends " << malicious_final_eval << ".\n";
//     bool success = verifier.final_check(malicious_final_eval);

//     if (success) {
//         std::cout << "  [TEST FAILED]: Verifier did not detect the bad final evaluation.\n";
//     } else {
//         std::cout << "  [TEST PASSED]: Verifier successfully caught the bad final evaluation.\n";
//     }
// }

// int main() {
//     
//    
    
//     // --- RUN ROBUSTNESS TESTS ---
//     std::cout << "\n\n======================================================\n";
//     std::cout << "             STARTING ROBUSTNESS TESTS\n";
//     std::cout << "======================================================\n";

//     test_malicious_round();
//     test_bad_initial_claim();
//     test_bad_final_evaluation();

//     return 0;
// }