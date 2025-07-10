#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <random>
#include <iomanip>
#include <stdexcept>
#include <fstream>
#include <chrono>
#include <tuple>

/**
 * @brief Computes (base^exp) % mod efficiently using binary exponentiation.
 */
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

/**
 * @class FieldElement
 * @brief Implements arithmetic over a finite field F_p.
 */
class FieldElement {
public:
    static const long long PRIME = 2147483647; 
    long long val;

    /** @brief Constructs a FieldElement, ensuring the value is in the range [0, PRIME-1]. */
    FieldElement(long long v = 0) : val(((v % PRIME) + PRIME) % PRIME) {}
    
    /** @brief Overloads the + operator for field addition. */
    FieldElement operator+(const FieldElement& other) const { return FieldElement((val + other.val) % PRIME); }
    /** @brief Overloads the - operator for field subtraction. */
    FieldElement operator-(const FieldElement& other) const { return FieldElement((val - other.val + PRIME) % PRIME); }
    /** @brief Overloads the * operator for field multiplication. */
    FieldElement operator*(const FieldElement& other) const { return FieldElement((static_cast<__int128>(val) * other.val) % PRIME); }
    /** @brief Overloads the == operator for equality comparison. */
    bool operator==(const FieldElement& other) const { return val == other.val; }
    
    /** @brief Computes the modular multiplicative inverse*/
    FieldElement inverse() const { return FieldElement(power(val, PRIME - 2, PRIME)); }
    /** @brief Overloads the / operator for field division. */
    FieldElement operator/(const FieldElement& other) const { return *this * other.inverse(); }

    /** @brief Enables printing FieldElement objects to an output stream. */
    friend std::ostream& operator<<(std::ostream& os, const FieldElement& fe) { return os << fe.val; }
};

/**
 * @class MatrixMLE
 * @brief Implements the Multilinear Extension (MLE) of a matrix.
 * This transforms the discrete matrix data into a continuous low-degree polynomial.
 */
class MatrixMLE {
private:
    std::vector<std::vector<FieldElement>> data;
    int rows, cols, log_rows, log_cols, total_vars;
public:
    /**
     * @brief Constructs the MLE from a matrix, calculating necessary parameters.
     * @param matrix The input matrix data.
     */
    MatrixMLE(const std::vector<std::vector<FieldElement>>& matrix) : data(matrix) {
        rows = matrix.size();
        cols = matrix.empty() ? 0 : matrix[0].size();
        log_rows = (rows <= 1) ? 1 : (int)ceil(log2(rows));
        log_cols = (cols <= 1) ? 1 : (int)ceil(log2(cols));
        total_vars = log_rows + log_cols;
    }

    /**
     * @brief Evaluates the MLE polynomial at a given point in the field.
     * Implements the formal definition f̃(x) = Σ_{v∈{0,1}ⁿ} f(v) * χᵥ(x).
     * @param point A vector of FieldElements representing the evaluation point.
     * @return The value of the polynomial at the given point.
     */
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
                        int v_bit = (row_bits >> bit) & 1;
                        indicator = indicator * ((FieldElement(1) - point[bit]) * FieldElement(1 - v_bit) + point[bit] * FieldElement(v_bit));
                    }
                    for (int bit = 0; bit < log_cols; bit++) {
                        int v_bit = (col_bits >> bit) & 1;
                        indicator = indicator * ((FieldElement(1) - point[log_rows + bit]) * FieldElement(1 - v_bit) + point[log_rows + bit] * FieldElement(v_bit));
                    }
                    result = result + data[row_bits][col_bits] * indicator;
                }
            }
        }
        return result;
    }

    /** @brief Returns the total number of variables in the MLE polynomial. */
    int get_total_vars() const { return total_vars; }
};

// --- PROTOCOL MESSAGE AND ACTOR CLASSES ---
struct ProverMessage { std::vector<FieldElement> coeffs; };
struct VerifierMessage { FieldElement challenge; };

/**
 * @class ThalerAttentionProver
 * @brief The powerful, untrusted party that performs the computation and generates proofs.
 */
class ThalerAttentionProver {
private:
    MatrixMLE X_mle, Q_mle, K_mle;
    int n, d, d_k, sumcheck_vars;
    std::vector<FieldElement> r_i, r_j, sumcheck_challenges;
public:
    /**
     * @brief Constructs the Prover with the problem definition.
     * @param X, Q, K The raw matrices for the computation.
     * @param r_i_challenges The random challenges for index 'i', provided by the Verifier.
     * @param r_j_challenges The random challenges for index 'j', provided by the Verifier.
     */
    ThalerAttentionProver(
        const std::vector<std::vector<FieldElement>>& X, 
        const std::vector<std::vector<FieldElement>>& Q, 
        const std::vector<std::vector<FieldElement>>& K,
        const std::vector<FieldElement>& r_i_challenges,
        const std::vector<FieldElement>& r_j_challenges
    )   : X_mle(X), Q_mle(Q), K_mle(K), r_i(r_i_challenges), r_j(r_j_challenges) {
        n = X.size(); d = X[0].size(); d_k = Q[0].size();
        int log_d = (d <= 1) ? 1 : (int)ceil(log2(d));
        int log_d_k = (d_k <= 1) ? 1 : (int)ceil(log2(d_k));
        sumcheck_vars = 2 * log_d + log_d_k;
    }

    /**
     * @brief Computes the initial, total sum to be claimed by brute-force.
     * This is the most computationally expensive single step of the protocol.
     * @return The total sum S' = Σ_{p,q,h} g(p,q,h).
     */
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

    /**
     * @brief The core computational engine: evaluates the reduced polynomial g(p,q,h) at a single point.
     * g(p,q,h) = X̃(rᵢ,p) * Q̃(p,h) * X̃(rⱼ,q) * K̃(q,h).
     * @param pqh_point A flattened vector containing the variable assignments for {p,q,h}.
     * @return The value of g at the specified point.
     */
    FieldElement evaluate_reduced_polynomial(const std::vector<FieldElement>& pqh_point) const {
        assert(pqh_point.size() == sumcheck_vars); int log_d = (d <= 1) ? 1 : (int)ceil(log2(d)); int log_d_k = (d_k <= 1) ? 1 : (int)ceil(log2(d_k));
        std::vector<FieldElement> p(pqh_point.begin(), pqh_point.begin() + log_d), q(pqh_point.begin() + log_d, pqh_point.begin() + 2 * log_d), h(pqh_point.begin() + 2 * log_d, pqh_point.end());
        std::vector<FieldElement> xip = r_i; xip.insert(xip.end(), p.begin(), p.end());
        std::vector<FieldElement> qph = p; qph.insert(qph.end(), h.begin(), h.end());
        std::vector<FieldElement> xjq = r_j; xjq.insert(xjq.end(), q.begin(), q.end());
        std::vector<FieldElement> kqh = q; kqh.insert(kqh.end(), h.begin(), h.end());
        return X_mle.evaluate(xip) * Q_mle.evaluate(qph) * X_mle.evaluate(xjq) * K_mle.evaluate(kqh);
    }
    
    /**
     * @brief Computes the Prover's message for a single round of the sum-check protocol.
     * This involves calculating the coefficients of the univariate polynomial for the current round's variable.
     * @param round The current round number (0-indexed).
     * @return A ProverMessage containing the polynomial coefficients.
     */
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
        FieldElement v0 = evaluations[0], v1 = evaluations[1], v2 = evaluations[2], two(2);
        FieldElement c0 = v0, c2 = (v2 - v1 * two + v0) / two, c1 = v1 - v0 - c2;
        return {{c0, c1, c2}};
    }
    
    /** @brief Stores the challenge received from the Verifier for use in the next round. */
    void receive_sumcheck_challenge(FieldElement c) { sumcheck_challenges.push_back(c); }
    
    /** @brief After all rounds, evaluates g at the final point composed of all challenges. */
    FieldElement final_evaluation() const { return evaluate_reduced_polynomial(sumcheck_challenges); }
    
    /** @brief Returns the total number of variables being summed over. */
    int get_sumcheck_vars() const { return sumcheck_vars; }
};

/**
 * @class ThalerAttentionVerifier
 * @brief The weak, trusted party that performs cheap checks on the Prover's messages.
 */
class ThalerAttentionVerifier {
private:
    FieldElement expected_value; 
    std::mt19937 rng; 
public:
    /** @brief Constructs the Verifier and its random number generator. */
    ThalerAttentionVerifier() : rng(std::random_device{}()) {}
    
    /** @brief Receives and stores the Prover's initial claimed sum. */
    void receive_claim(FieldElement claim) { expected_value = claim; }
    
    /**
     * @brief Processes the Prover's message for one round.
     * It performs the core sum-check test and generates a new challenge if successful.
     * @param msg The Prover's message containing polynomial coefficients.
     * @return A VerifierMessage containing a new random challenge.
     */
    VerifierMessage process_sumcheck_round(const ProverMessage& msg) {
        assert(msg.coeffs.size() == 3);
        FieldElement c0 = msg.coeffs[0], c1 = msg.coeffs[1], c2 = msg.coeffs[2];
        FieldElement h0 = c0, h1 = c0 + c1 + c2;
        FieldElement current_sum = h0 + h1;
        if (!(current_sum == expected_value)) { throw std::runtime_error("Verification failed: Sum check mismatch."); }
        std::uniform_int_distribution<long long> dist(1, FieldElement::PRIME - 1);
        FieldElement challenge(dist(rng));
        expected_value = c0 + c1 * challenge + c2 * challenge * challenge;
        return {challenge};
    }
    
    /**
     * @brief Performs the final check after all interactive rounds are complete.
     * @param prover_final_eval The Prover's direct evaluation of g at the final challenge point.
     * @return True if verification is successful, false otherwise.
     */
    bool final_check(FieldElement prover_final_eval) {
        return (prover_final_eval == expected_value);
    }
};


//Helper
std::vector<std::vector<FieldElement>> create_random_matrix(int r, int c, std::mt19937& rng) {
    std::vector<std::vector<FieldElement>> m(r, std::vector<FieldElement>(c));
    std::uniform_int_distribution<long long> dist(0, 100);
    for (int i = 0; i < r; ++i) for (int j = 0; j < c; ++j) m[i][j] = FieldElement(dist(rng));
    return m;
}

// struct LLMConfig {
//     std::string name;
//     int n; // Context Length
//     int d; // Embedding Dim
//     int d_k; // Head Dim
// };

struct ModelConfig {
    std::string name;
    int n; // Sequence/Context Length
    int d; // Embedding Dimension
    int d_k; // Head Dimension
};

// int main() {
//     // --- BENCHMARKING SUITE SETUP ---
//     // A list of typical Vision Transformer and other model dimensions.
//     // Larger models are commented out by default due to extremely long run times.
//     std::vector<ModelConfig> configs = {
//         {"Tiny (Test)", 128, 128, 32},
//         {"ViT-Base (224px)", 197, 768, 64}
//         // {"ViT-Base (384px)", 577, 768, 64},  // WARNING: SLOW - Tests scaling with sequence length 'n'
//         // {"ViT-Large (224px)", 197, 1024, 64} // WARNING: VERY SLOW - Tests scaling with embedding dim 'd'
//     };

//     std::ofstream results_file("vision_transformer_benchmark.csv");
//     results_file << "ConfigName,N,D,D_k,NumRounds,ProverClaimTime_s,ProverRoundsTime_s,TotalProverTime_s,VerifierAvgRoundTime_us,VerifierFinalCheckTime_us\n";

//     std::cout << "Starting Vision Transformer Benchmark Suite..." << std::endl;

//     for (const auto& config : configs) {
//         std::cout << "\n==================================================================\n";
//         std::cout << "--- Testing Config: " << config.name << " (n=" << config.n 
//                   << ", d=" << config.d << ", d_k=" << config.d_k << ") ---\n";
//         std::cout << "==================================================================\n";

//         // --- SETUP ---
//         std::mt19937 rng(config.n + config.d); // Use config for reproducible seed
//         auto X = create_random_matrix(config.n, config.d, rng);
//         auto Q = create_random_matrix(config.d, config.d_k, rng);
//         auto K = create_random_matrix(config.d, config.d_k, rng);
        
//         std::vector<FieldElement> r_i, r_j;
//         int log_n = (config.n <= 1) ? 1 : (int)ceil(log2(config.n));
//         std::uniform_int_distribution<long long> dist(0, FieldElement::PRIME - 1);
//         for (int bit = 0; bit < log_n; bit++) {
//             r_i.push_back(FieldElement(dist(rng)));
//             r_j.push_back(FieldElement(dist(rng)));
//         }

//         ThalerAttentionProver prover(X, Q, K, r_i, r_j);
//         ThalerAttentionVerifier verifier;
//         int num_rounds = prover.get_sumcheck_vars();
//         std::cout << "Sum-check will run for " << num_rounds << " rounds.\n";

//         // --- BENCHMARK PROVER ---
//         std::cout << "[Prover] Starting 'compute_claimed_sum'. This may take a long time...\n";
//         auto start_claim = std::chrono::high_resolution_clock::now();
//         FieldElement claimed_sum = prover.compute_claimed_sum();
//         auto end_claim = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double, std::milli> claim_time_ms = end_claim - start_claim;
//         verifier.receive_claim(claimed_sum);
//         std::cout << "[Prover] 'compute_claimed_sum' finished.\n";
        
//         std::cout << "[Prover] Starting " << num_rounds << " sum-check rounds computation...\n";
//         auto start_prover_rounds = std::chrono::high_resolution_clock::now();
//         for (int round = 0; round < num_rounds; ++round) {
//             prover.compute_sumcheck_round(round);
//             prover.receive_sumcheck_challenge(FieldElement(1));
//         }
//         auto end_prover_rounds = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double, std::milli> prover_rounds_time_ms = end_prover_rounds - start_prover_rounds;
//         std::cout << "[Prover] All rounds finished.\n";
        
//         double prover_claim_s = claim_time_ms.count() / 1000.0;
//         double prover_rounds_s = prover_rounds_time_ms.count() / 1000.0;
//         double total_prover_s = prover_claim_s + prover_rounds_s;

//         // --- BENCHMARK VERIFIER ---
//         ThalerAttentionProver honest_prover_for_verifier(X, Q, K, r_i, r_j);
//         verifier.receive_claim(claimed_sum);
//         std::chrono::duration<double, std::micro> total_verifier_round_time_us(0);

//         for (int round = 0; round < num_rounds; ++round) {
//             ProverMessage msg = honest_prover_for_verifier.compute_sumcheck_round(round);
//             auto start_verifier_round = std::chrono::high_resolution_clock::now();
//             VerifierMessage response = verifier.process_sumcheck_round(msg);
//             auto end_verifier_round = std::chrono::high_resolution_clock::now();
//             total_verifier_round_time_us += (end_verifier_round - start_verifier_round);
//             honest_prover_for_verifier.receive_sumcheck_challenge(response.challenge);
//         }
//         double avg_verifier_round_time_us = total_verifier_round_time_us.count() / num_rounds;

//         FieldElement final_prover_eval = honest_prover_for_verifier.final_evaluation();
//         auto start_final_check = std::chrono::high_resolution_clock::now();
//         verifier.final_check(final_prover_eval);
//         auto end_final_check = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double, std::micro> final_check_time_us = end_final_check - start_final_check;

//         // --- PRINT AND SAVE RESULTS ---
//         std::cout << "\n--- Results for " << config.name << " ---" << std::endl;
//         std::cout << std::fixed << std::setprecision(4);
//         std::cout << "  Prover Claim Time  : " << prover_claim_s << " s\n";
//         std::cout << "  Prover Rounds Time : " << prover_rounds_s << " s\n";
//         std::cout << "  Total Prover Time  : " << total_prover_s << " s\n";
//         std::cout << "  ---------------------------------------\n";
//         std::cout << "  Verifier Avg Round : " << avg_verifier_round_time_us << " us\n";
//         std::cout << "  Verifier Final Check: " << final_check_time_us.count() << " us\n";

//         results_file << config.name << "," << config.n << "," << config.d << "," << config.d_k << ","
//                      << num_rounds << ","
//                      << prover_claim_s << "," << prover_rounds_s << "," << total_prover_s << ","
//                      << avg_verifier_round_time_us << "," << final_check_time_us.count() << "\n";
//     }

//     results_file.close();
//     std::cout << "\nBenchmark suite finished. Results saved to vision_transformer_benchmark.csv" << std::endl;

//     return 0;
// }


int main() {
    try {
        const int n = 16, d = 16, d_k = 16;

        std::cout << "THALER-STYLE ATTENTION SUMCHECK PROTOCOL" << std::endl;
        std::cout << "========================================" << std::endl;

        std::mt19937 rng(std::random_device{}());
        auto X = create_random_matrix(n, d, rng);
        auto Q = create_random_matrix(d, d_k, rng);
        auto K = create_random_matrix(d, d_k, rng);
        
        // Phase 1: Verifier generates Thaler challenges for outer indices {i,j}
        std::vector<FieldElement> r_i, r_j;
        int log_n = (n <= 1) ? 1 : (int)ceil(log2(n));
        std::uniform_int_distribution<long long> dist(0, FieldElement::PRIME - 1);
        for (int bit = 0; bit < log_n; bit++) {
            r_i.push_back(FieldElement(dist(rng)));
            r_j.push_back(FieldElement(dist(rng)));
        }

        // Phase 2: Prover is created using the challenges and computes its claim
        ThalerAttentionProver prover(X, Q, K, r_i, r_j);
        ThalerAttentionVerifier verifier;
        FieldElement claimed_sum = prover.compute_claimed_sum();
        verifier.receive_claim(claimed_sum);
        
        // Phase 3: Interactive Sum-Check Protocol
        int num_rounds = prover.get_sumcheck_vars();
        for (int round = 0; round < num_rounds; round++) {
            ProverMessage msg = prover.compute_sumcheck_round(round);
            VerifierMessage response = verifier.process_sumcheck_round(msg);
            prover.receive_sumcheck_challenge(response.challenge);
        }
        
        // Final Check
        FieldElement final_prover_eval = prover.final_evaluation();
        bool success = verifier.final_check(final_prover_eval);

        if (success) { std::cout << "\nPROTOCOL SUCCESSFUL" << std::endl; }
        else { std::cout << "\nPROTOCOL FAILED" << std::endl; }

    } catch (const std::exception& e) {
        std::cerr << "Error during protocol: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}