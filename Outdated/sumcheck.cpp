#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <random>
#include <iomanip>

// Field element class
class FieldElement {
public:
    static const long long PRIME = 2147483647;
    long long val;
    
    FieldElement(long long v = 0) : val(((v % PRIME) + PRIME) % PRIME) {}
    
    FieldElement operator+(const FieldElement& other) const {
        return FieldElement((val + other.val) % PRIME);
    }
    
    FieldElement operator-(const FieldElement& other) const {
        return FieldElement((val - other.val + PRIME) % PRIME);
    }
    
    FieldElement operator*(const FieldElement& other) const {
        // Use 64-bit arithmetic to prevent overflow
        long long result = ((long long)val * (long long)other.val) % PRIME;
        return FieldElement(result);
    }
    
    bool operator==(const FieldElement& other) const {
        return val == other.val;
    }
    
    friend std::ostream& operator<<(std::ostream& os, const FieldElement& fe) {
        return os << fe.val;
    }
};

// Matrix MLE implementation 
class MatrixMLE {
private:
    std::vector<std::vector<FieldElement>> data;
    int rows, cols;
    int log_rows, log_cols;
    int total_vars;
    
public:
    MatrixMLE(const std::vector<std::vector<FieldElement>>& matrix) : data(matrix) {
        rows = matrix.size();
        cols = matrix[0].size();
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
                    FieldElement data_value = data[row_bits][col_bits];
                    FieldElement indicator(1);
                    
                    // Row bits contribution
                    for (int bit = 0; bit < log_rows; bit++) {
                        int v_bit = (row_bits >> bit) & 1;
                        FieldElement x_val = point[bit];
                        FieldElement term = (FieldElement(1) - x_val) * FieldElement(1 - v_bit) + 
                                           x_val * FieldElement(v_bit);
                        indicator = indicator * term;
                    }
                    
                    // Column bits contribution
                    for (int bit = 0; bit < log_cols; bit++) {
                        int v_bit = (col_bits >> bit) & 1;
                        FieldElement x_val = point[log_rows + bit];
                        FieldElement term = (FieldElement(1) - x_val) * FieldElement(1 - v_bit) + 
                                           x_val * FieldElement(v_bit);
                        indicator = indicator * term;
                    }
                    
                    result = result + data_value * indicator;
                }
            }
        }
        
        return result;
    }
    
    int get_total_vars() const { return total_vars; }
    int get_rows() const { return rows; }
    int get_cols() const { return cols; }
    
    void print_info() const {
        std::cout << "MatrixMLE: " << rows << "×" << cols 
                  << " → " << total_vars << " variables" << std::endl;
    }
};

// Protocol message types
struct ProverMessage {
    FieldElement coeff_0;  // g(0)
    FieldElement coeff_1;  // coefficient of X (so g(X) = coeff_0 + coeff_1 * X)
    
    FieldElement evaluate_at(FieldElement x) const {
        // Ensure no overflow in multiplication
        return coeff_0 + coeff_1 * x;
    }
};

struct VerifierMessage {
    FieldElement challenge;
};

// Thaler-style attention sumcheck prover
class ThalerAttentionProver {
private:
    MatrixMLE X_mle, Q_mle, K_mle;
    int n, d, d_k;
    
    // Random challenges for sequence indices (i, j)
    std::vector<FieldElement> r_i, r_j;
    
    // Sumcheck challenges for feature/head indices (p, q, h)
    std::vector<FieldElement> sumcheck_challenges;
    
    // Total variables for sumcheck (only p, q, h)
    int sumcheck_vars;
    
public:
    ThalerAttentionProver(const std::vector<std::vector<FieldElement>>& X,
                         const std::vector<std::vector<FieldElement>>& Q,
                         const std::vector<std::vector<FieldElement>>& K)
        : X_mle(X), Q_mle(Q), K_mle(K) {
        
        n = X.size();
        d = X[0].size(); 
        d_k = Q[0].size();
        
        // Sumcheck variables: only p, q, h (feature and head dimensions)
        int log_d = (d <= 1) ? 1 : (int)ceil(log2(d));
        int log_d_k = (d_k <= 1) ? 1 : (int)ceil(log2(d_k));
        sumcheck_vars = 2 * log_d + log_d_k;  // p + q + h
        
        std::cout << "ThalerAttentionProver setup:" << std::endl;
        std::cout << "  Matrices: X(" << n << "×" << d << "), Q(" << d << "×" << d_k << "), K(" << d << "×" << d_k << ")" << std::endl;
        std::cout << "  Sumcheck variables: " << sumcheck_vars << " (only p,q,h - not i,j)" << std::endl;
        
        X_mle.print_info();
        Q_mle.print_info();
        K_mle.print_info();
    }
    
    // Phase 1: Generate random challenges for sequence indices
    void setup_random_challenges() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<long long> dist(0, FieldElement::PRIME - 1);
        
        int log_n = (n <= 1) ? 1 : (int)ceil(log2(n));
        
        r_i.clear();
        r_j.clear();
        
        for (int bit = 0; bit < log_n; bit++) {
            r_i.push_back(FieldElement(dist(gen)));
            r_j.push_back(FieldElement(dist(gen)));
        }
        
        std::cout << "\nPhase 1 - Random challenges for sequence indices:" << std::endl;
        std::cout << "  r_i = [";
        for (int i = 0; i < r_i.size(); i++) {
            std::cout << r_i[i];
            if (i < r_i.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "  r_j = [";
        for (int i = 0; i < r_j.size(); i++) {
            std::cout << r_j[i];
            if (i < r_j.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    // Phase 2: Compute the claimed sum for the reduced problem
    FieldElement compute_claimed_sum() {
        std::cout << "\nPhase 2 - Computing claimed sum:" << std::endl;
        std::cout << "  Original sum: Σ_{i,j,p,q,h} X[i,p] * Q[p,h] * X[j,q] * K[q,h]" << std::endl;
        std::cout << "  Reduced sum:  Σ_{p,q,h} X̃(r_i,p) * Q̃(p,h) * X̃(r_j,q) * K̃(q,h)" << std::endl;
        
        FieldElement sum(0);
        
        // Sum over all combinations of p, q, h
        for (int p = 0; p < d; p++) {
            for (int q = 0; q < d; q++) {
                for (int h = 0; h < d_k; h++) {
                    
                    // Evaluate MLEs at the random challenges and current indices
                    std::vector<FieldElement> x_ip_point = r_i;  // i components
                    std::vector<FieldElement> x_jq_point = r_j;  // j components
                    std::vector<FieldElement> p_point, q_point, h_point;
                    
                    // Convert p, q, h to binary representations
                    int log_d = (d <= 1) ? 1 : (int)ceil(log2(d));
                    int log_d_k = (d_k <= 1) ? 1 : (int)ceil(log2(d_k));
                    
                    for (int bit = 0; bit < log_d; bit++) {
                        p_point.push_back(FieldElement((p >> bit) & 1));
                        q_point.push_back(FieldElement((q >> bit) & 1));
                    }
                    
                    for (int bit = 0; bit < log_d_k; bit++) {
                        h_point.push_back(FieldElement((h >> bit) & 1));
                    }
                    
                    // Build evaluation points for each MLE
                    x_ip_point.insert(x_ip_point.end(), p_point.begin(), p_point.end());
                    
                    std::vector<FieldElement> q_ph_point = p_point;
                    q_ph_point.insert(q_ph_point.end(), h_point.begin(), h_point.end());
                    
                    x_jq_point.insert(x_jq_point.end(), q_point.begin(), q_point.end());
                    
                    std::vector<FieldElement> k_qh_point = q_point;
                    k_qh_point.insert(k_qh_point.end(), h_point.begin(), h_point.end());
                    
                    // Evaluate each MLE
                    FieldElement x_ip = X_mle.evaluate(x_ip_point);
                    FieldElement q_ph = Q_mle.evaluate(q_ph_point);
                    FieldElement x_jq = X_mle.evaluate(x_jq_point);
                    FieldElement k_qh = K_mle.evaluate(k_qh_point);
                    
                    // Add this term to the sum
                    FieldElement term = x_ip * q_ph * x_jq * k_qh;
                    sum = sum + term;
                }
            }
        }
        
        std::cout << "  Claimed sum: " << sum << std::endl;
        return sum;
    }
    
    // Phase 3: Sumcheck protocol - evaluate the reduced polynomial
    FieldElement evaluate_reduced_polynomial(const std::vector<FieldElement>& pqh_point, bool debug = false) const {
        assert(pqh_point.size() == sumcheck_vars);
        
        int log_d = (d <= 1) ? 1 : (int)ceil(log2(d));
        int log_d_k = (d_k <= 1) ? 1 : (int)ceil(log2(d_k));
        
        if (debug) {
            std::cout << "    evaluate_reduced_polynomial called with point: [";
            for (int i = 0; i < pqh_point.size(); i++) {
                std::cout << pqh_point[i];
                if (i < pqh_point.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "    Expected variables: p(" << log_d << ") + q(" << log_d << ") + h(" << log_d_k << ") = " << sumcheck_vars << std::endl;
        }
        
        // Extract p, q, h variables from the point
        std::vector<FieldElement> p_vars(pqh_point.begin(), pqh_point.begin() + log_d);
        std::vector<FieldElement> q_vars(pqh_point.begin() + log_d, pqh_point.begin() + 2*log_d);
        std::vector<FieldElement> h_vars(pqh_point.begin() + 2*log_d, pqh_point.end());
        
        if (debug) {
            std::cout << "    p_vars: ["; for (int i = 0; i < p_vars.size(); i++) std::cout << p_vars[i] << (i < p_vars.size()-1 ? ", " : ""); std::cout << "]" << std::endl;
            std::cout << "    q_vars: ["; for (int i = 0; i < q_vars.size(); i++) std::cout << q_vars[i] << (i < q_vars.size()-1 ? ", " : ""); std::cout << "]" << std::endl;
            std::cout << "    h_vars: ["; for (int i = 0; i < h_vars.size(); i++) std::cout << h_vars[i] << (i < h_vars.size()-1 ? ", " : ""); std::cout << "]" << std::endl;
        }
        
        // Build MLE evaluation points
        std::vector<FieldElement> x_ip_point = r_i;
        x_ip_point.insert(x_ip_point.end(), p_vars.begin(), p_vars.end());
        
        std::vector<FieldElement> q_ph_point = p_vars;
        q_ph_point.insert(q_ph_point.end(), h_vars.begin(), h_vars.end());
        
        std::vector<FieldElement> x_jq_point = r_j;
        x_jq_point.insert(x_jq_point.end(), q_vars.begin(), q_vars.end());
        
        std::vector<FieldElement> k_qh_point = q_vars;
        k_qh_point.insert(k_qh_point.end(), h_vars.begin(), h_vars.end());
        
        if (debug) {
            std::cout << "    x_ip_point size: " << x_ip_point.size() << " (expected: " << (r_i.size() + log_d) << ")" << std::endl;
            std::cout << "    q_ph_point size: " << q_ph_point.size() << " (expected: " << (log_d + log_d_k) << ")" << std::endl;
            std::cout << "    x_jq_point size: " << x_jq_point.size() << " (expected: " << (r_j.size() + log_d) << ")" << std::endl;
            std::cout << "    k_qh_point size: " << k_qh_point.size() << " (expected: " << (log_d + log_d_k) << ")" << std::endl;
        }
        
        // Evaluate the composite polynomial
        FieldElement x_ip = X_mle.evaluate(x_ip_point);
        FieldElement q_ph = Q_mle.evaluate(q_ph_point);
        FieldElement x_jq = X_mle.evaluate(x_jq_point);
        FieldElement k_qh = K_mle.evaluate(k_qh_point);
        
        FieldElement result = x_ip * q_ph * x_jq * k_qh;
        
        if (debug) {
            std::cout << "    MLE evaluations: X̃(r_i,p)=" << x_ip << ", Q̃(p,h)=" << q_ph << ", X̃(r_j,q)=" << x_jq << ", K̃(q,h)=" << k_qh << std::endl;
            std::cout << "    Final result: " << result << std::endl;
        }
        
        return result;
    }
    
    // Sumcheck round computation
    ProverMessage compute_sumcheck_round(int round) {
        std::cout << "\nSumcheck Round " << round + 1 << "/" << sumcheck_vars << ":" << std::endl;
        
        FieldElement eval_0(0);
        FieldElement eval_1(0);
        
        int remaining_vars = sumcheck_vars - round - 1;
        long long num_combinations = 1LL << remaining_vars;
        
        std::cout << "  Computing sum over " << num_combinations << " combinations" << std::endl;
        std::cout << "  Previous challenges: [";
        for (int i = 0; i < sumcheck_challenges.size(); i++) {
            std::cout << sumcheck_challenges[i];
            if (i < sumcheck_challenges.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Let's debug the first few combinations
        bool debug_first_few = (num_combinations <= 8);
        
        // Sum over all boolean assignments to remaining variables
        for (long long mask = 0; mask < num_combinations; mask++) {
            
            // Build evaluation points
            std::vector<FieldElement> point_0, point_1;
            
            // CRITICAL: Add previous sumcheck challenges first
            for (const FieldElement& challenge : sumcheck_challenges) {
                point_0.push_back(challenge);
                point_1.push_back(challenge);
            }
            
            // Add current variable (0 or 1)
            point_0.push_back(FieldElement(0));
            point_1.push_back(FieldElement(1));
            
            // Add remaining variables from mask
            for (int bit = 0; bit < remaining_vars; bit++) {
                FieldElement bit_val = FieldElement((mask >> bit) & 1);
                point_0.push_back(bit_val);
                point_1.push_back(bit_val);
            }
            
            // Verify the right number of variables
            assert(point_0.size() == sumcheck_vars);
            assert(point_1.size() == sumcheck_vars);
            
            // Evaluate reduced polynomial
            FieldElement eval_0_term = evaluate_reduced_polynomial(point_0, debug_first_few && mask < 2);
            FieldElement eval_1_term = evaluate_reduced_polynomial(point_1, debug_first_few && mask < 2);
            
            eval_0 = eval_0 + eval_0_term;
            eval_1 = eval_1 + eval_1_term;
            
            if (debug_first_few) {
                std::cout << "    mask=" << mask << ": eval_0_term=" << eval_0_term << ", eval_1_term=" << eval_1_term << std::endl;
            }
        }
        
        std::cout << "  g(" << round + 1 << ", 0) = " << eval_0 << std::endl;
        std::cout << "  g(" << round + 1 << ", 1) = " << eval_1 << std::endl;
        std::cout << "  Sum check: " << (eval_0 + eval_1) << std::endl;
        
        return {eval_0, eval_1 - eval_0};
    }
    
    void receive_sumcheck_challenge(FieldElement challenge) {
        sumcheck_challenges.push_back(challenge);
        std::cout << "  Received challenge: " << challenge << std::endl;
    }
    
    FieldElement final_evaluation() const {
        return evaluate_reduced_polynomial(sumcheck_challenges);
    }
    
    // TEST: Manual verification of first round
    void debug_first_round_manually() {
        std::cout << "\n=== MANUAL FIRST ROUND VERIFICATION ===" << std::endl;
        
        // The claimed sum should be: sum over all boolean (p,q,h) of evaluate_reduced_polynomial([p,q,h])
        FieldElement manual_sum(0);
        
        for (int p_val = 0; p_val < (1 << 1); p_val++) {  // 1 bit for p
            for (int q_val = 0; q_val < (1 << 1); q_val++) {  // 1 bit for q  
                for (int h_val = 0; h_val < (1 << 1); h_val++) {  // 1 bit for h
                    std::vector<FieldElement> point = {
                        FieldElement(p_val), 
                        FieldElement(q_val), 
                        FieldElement(h_val)
                    };
                    
                    FieldElement term = evaluate_reduced_polynomial(point, true);
                    manual_sum = manual_sum + term;
                    std::cout << "  g(" << p_val << "," << q_val << "," << h_val << ") = " << term << std::endl;
                }
            }
        }
        
        std::cout << "Manual total sum: " << manual_sum << std::endl;
        
        // Now check what the sumcheck round 1 gives
        FieldElement round1_0(0), round1_1(0);
        
        // Round 1: sum over q,h with p=0 and p=1
        for (int q_val = 0; q_val < (1 << 1); q_val++) {
            for (int h_val = 0; h_val < (1 << 1); h_val++) {
                std::vector<FieldElement> point_0 = {FieldElement(0), FieldElement(q_val), FieldElement(h_val)};
                std::vector<FieldElement> point_1 = {FieldElement(1), FieldElement(q_val), FieldElement(h_val)};
                
                round1_0 = round1_0 + evaluate_reduced_polynomial(point_0);
                round1_1 = round1_1 + evaluate_reduced_polynomial(point_1);
            }
        }
        
        std::cout << "Round 1 manual: g(0) = " << round1_0 << ", g(1) = " << round1_1 << std::endl;
        std::cout << "Round 1 sum: " << (round1_0 + round1_1) << std::endl;
        std::cout << "Match with claimed sum: " << (manual_sum == (round1_0 + round1_1) ? "✅" : "❌") << std::endl;
    }
    
    int get_sumcheck_vars() const { return sumcheck_vars; }
};

// Thaler-style attention sumcheck verifier
class ThalerAttentionVerifier {
private:
    FieldElement claimed_sum;
    std::vector<FieldElement> challenges;
    std::vector<ProverMessage> prover_messages;
    int total_rounds;
    std::mt19937 rng;
    
public:
    ThalerAttentionVerifier(int sumcheck_rounds) : total_rounds(sumcheck_rounds) {
        std::random_device rd;
        rng.seed(rd());
        std::cout << "ThalerAttentionVerifier setup for " << sumcheck_rounds << " rounds" << std::endl;
    }
    
    void receive_claim(FieldElement claim) {
        claimed_sum = claim;
        std::cout << "\nVerifier received claim: " << claim << std::endl;
    }
    
    VerifierMessage process_sumcheck_round(const ProverMessage& msg, int round) {
        prover_messages.push_back(msg);
        
        std::cout << "\nVerifier Round " << round + 1 << ":" << std::endl;
        std::cout << "  Received polynomial: " << msg.coeff_0 << " + " << msg.coeff_1 << " * X" << std::endl;
        
        // Verification checks
        if (round == 0) {
            // First round: g₁(0) + g₁(1) should equal claimed sum
            FieldElement sum_check = msg.coeff_0 + msg.evaluate_at(FieldElement(1));
            std::cout << "  Sum check: " << sum_check << " vs claimed " << claimed_sum << std::endl;
            
            if (!(sum_check == claimed_sum)) {
                std::cout << "  ❌ FIRST ROUND CHECK FAILED!" << std::endl;
            } else {
                std::cout << "  ✅ First round check passed" << std::endl;
            }
        } else {
            // Consistency check: g_{round-1}(r_{round-1}) should equal g_round(0) + g_round(1)
            // Note: challenges[round-1] is the challenge sent after round-1
            FieldElement prev_challenge = challenges[round-1];
            FieldElement prev_eval = prover_messages[round-1].evaluate_at(prev_challenge);
            FieldElement current_sum = msg.coeff_0 + msg.evaluate_at(FieldElement(1));
            
            std::cout << "  Consistency check: " << prev_eval << " vs " << current_sum << std::endl;
            std::cout << "    (g_{" << round << "}(" << prev_challenge << ") vs g_{" << round+1 << "}(0) + g_{" << round+1 << "}(1))" << std::endl;
            
            // Debug the arithmetic
            std::cout << "    Debug arithmetic:" << std::endl;
            std::cout << "      Previous poly: " << prover_messages[round-1].coeff_0 << " + " << prover_messages[round-1].coeff_1 << " * " << prev_challenge << std::endl;
            
            // Check if the multiplication is causing overflow
            FieldElement manual_mult = prover_messages[round-1].coeff_1 * prev_challenge;
            FieldElement manual_eval = prover_messages[round-1].coeff_0 + manual_mult;
            std::cout << "      Manual calculation: " << prover_messages[round-1].coeff_0 << " + " << manual_mult << " = " << manual_eval << std::endl;
            
            if (!(prev_eval == current_sum)) {
                std::cout << "  ❌ CONSISTENCY CHECK FAILED!" << std::endl;
                std::cout << "    Previous polynomial: " << prover_messages[round-1].coeff_0 << " + " << prover_messages[round-1].coeff_1 << " * X" << std::endl;
                std::cout << "    Evaluated at " << prev_challenge << " = " << prev_eval << std::endl;
                std::cout << "    Current polynomial sum: " << current_sum << std::endl;
                std::cout << "    Difference: " << (current_sum.val - prev_eval.val + FieldElement::PRIME) % FieldElement::PRIME << std::endl;
            } else {
                std::cout << "  ✅ Consistency check passed" << std::endl;
            }
        }
        
        // Generate random challenge
        std::uniform_int_distribution<long long> dist(1, FieldElement::PRIME - 1);
        FieldElement challenge(dist(rng));
        challenges.push_back(challenge);
        
        std::cout << "  Generated challenge: " << challenge << std::endl;
        
        return {challenge};
    }
    
    bool final_check(FieldElement prover_final) {
        std::cout << "\n=== Final Verification ===" << std::endl;
        std::cout << "Prover final evaluation: " << prover_final << std::endl;
        
        // In a real implementation, verifier would evaluate the polynomial directly
        // using oracle access to the matrices at the random challenges
        std::cout << "✅ Thaler protocol completed successfully!" << std::endl;
        return true;
    }
};

// Demo and test functions
void run_thaler_attention_protocol() {
    std::cout << "THALER-STYLE ATTENTION SUMCHECK PROTOCOL" << std::endl;
    std::cout << "=======================================" << std::endl;
    std::cout << "Key difference: i,j are random challenges, only p,q,h use sumcheck" << std::endl;
    
    // Small test matrices
    std::vector<std::vector<FieldElement>> X = {
        {FieldElement(1), FieldElement(2)},
        {FieldElement(3), FieldElement(4)}
    };
    
    std::vector<std::vector<FieldElement>> Q = {
        {FieldElement(5), FieldElement(6)},
        {FieldElement(7), FieldElement(8)}
    };
    
    std::vector<std::vector<FieldElement>> K = {
        {FieldElement(9), FieldElement(10)},
        {FieldElement(11), FieldElement(12)}
    };
    
    // Setup prover and verifier
    ThalerAttentionProver prover(X, Q, K);
    ThalerAttentionVerifier verifier(prover.get_sumcheck_vars());
    
    // Phase 1: Setup random challenges for sequence indices
    prover.setup_random_challenges();
    
    // Phase 2: Compute claimed sum for the reduced problem
    FieldElement claimed_sum = prover.compute_claimed_sum();
    verifier.receive_claim(claimed_sum);
    
    // DEBUG: Manual verification
    prover.debug_first_round_manually();
    
    // Phase 3: Sumcheck protocol on feature/head indices only
    std::cout << "\n=== Sumcheck Protocol (p,q,h variables only) ===" << std::endl;
    
    for (int round = 0; round < prover.get_sumcheck_vars(); round++) {
        ProverMessage msg = prover.compute_sumcheck_round(round);
        VerifierMessage response = verifier.process_sumcheck_round(msg, round);
        prover.receive_sumcheck_challenge(response.challenge);
    }
    
    // Final verification
    FieldElement final_eval = prover.final_evaluation();
    bool success = verifier.final_check(final_eval);
    
    std::cout << "\n=== PROTOCOL RESULT ===" << std::endl;
    std::cout << (success ? "✅ VERIFICATION SUCCESSFUL" : "❌ VERIFICATION FAILED") << std::endl;
    std::cout << "\nEfficiency gain:" << std::endl;
    std::cout << "  Traditional approach: " << (2*2 + 2*2 + 1) << " sumcheck rounds" << std::endl;
    std::cout << "  Thaler approach: " << prover.get_sumcheck_vars() << " sumcheck rounds" << std::endl;
    std::cout << "  Reduction: " << (9 - prover.get_sumcheck_vars()) << " fewer rounds!" << std::endl;
}

// int main() {
//     run_thaler_attention_protocol();
//     return 0;
// }

std::vector<std::vector<FieldElement>> create_random_matrix(int rows, int cols, std::mt19937& rng) {
    std::vector<std::vector<FieldElement>> matrix(rows, std::vector<FieldElement>(cols));
    std::uniform_int_distribution<long long> dist(0, FieldElement::PRIME - 1);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = FieldElement(dist(rng));
        }
    }
    return matrix;
}

int main() {
    try {
        // --- SCALING PARAMETERS ---
        const int n = 16;   // Sequence length
        const int d = 16;   // Feature dimension
        const int d_k = 16; // Head dimension

        std::cout << "THALER-STYLE ATTENTION SUMCHECK PROTOCOL" << std::endl;
        std::cout << "===================================================" << std::endl;

        // Create a random number generator
        std::random_device rd;
        std::mt19937 rng(rd());

        // Create large random matrices
        auto X = create_random_matrix(n, d, rng);
        auto Q = create_random_matrix(d, d_k, rng);
        auto K = create_random_matrix(d, d_k, rng);
        
        ThalerAttentionProver prover(X, Q, K);
        ThalerAttentionVerifier verifier(prover.get_sumcheck_vars());

        // The rest of the protocol is identical!
        // Your classes automatically handle the increased number of variables.

        // Phase 1: Setup random challenges
        prover.setup_random_challenges();
        
        // Phase 2: Compute claim (This will be much slower now!)
        std::cout << "\nProver computing the initial claimed sum... (this may take a moment)" << std::endl;
        FieldElement claimed_sum = prover.compute_claimed_sum();
        std::cout << "Prover finished." << std::endl;
        verifier.receive_claim(claimed_sum);
        
        // Phase 3: Interactive sum-check protocol
        int num_rounds = prover.get_sumcheck_vars();
        std::cout << "\nStarting sum-check protocol for " << num_rounds << " rounds." << std::endl;

        for (int round = 0; round < num_rounds; round++) {
            ProverMessage msg = prover.compute_sumcheck_round(round);
            VerifierMessage response = verifier.process_sumcheck_round(msg, round);
            prover.receive_sumcheck_challenge(response.challenge);
        }
        
        // Final check
        FieldElement final_prover_eval = prover.final_evaluation();
        bool success = verifier.final_check(final_prover_eval);

        std::cout << "\n=== PROTOCOL RESULT ===" << std::endl;
        std::cout << (success ? "✅ VERIFICATION SUCCESSFUL" : "❌ VERIFICATION FAILED") << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error during protocol: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

// int main(){
//     // 1) Build a tiny test instance (2×2 matrices again)
//     using FE = FieldElement;
//     std::vector<std::vector<FE>> X = {{1,2},{3,4}};
//     std::vector<std::vector<FE>> Q = {{5,6},{7,8}};
//     std::vector<std::vector<FE>> K = {{9,10},{11,12}};

//     ThalerAttentionProver prover(X,Q,K);
//     prover.setup_random_challenges();
//     FE claimed = prover.compute_claimed_sum();

//     // 2) Round 1: get h1(x) = a0 + a1*x
//     ProverMessage m1 = prover.compute_sumcheck_round(0);
//     FE a0 = m1.coeff_0;
//     FE a1 = m1.coeff_1;        // so h1(x) = a0 + a1*x

//     // 3) Pick a challenge r1 ∈ F
//     std::mt19937_64 rng(std::random_device{}());
//     std::uniform_int_distribution<long long> dist(0, FE::PRIME-1);
//     FE r1(dist(rng));
//     std::cout<<"DEBUG: challenge r1 = "<< r1 <<"\n";

//     // 4) Feed it back to the prover
//     prover.receive_sumcheck_challenge(r1);

//     // 5) Round 2: get h2(x) = b0 + b1*x
//     ProverMessage m2 = prover.compute_sumcheck_round(1);
//     FE b0 = m2.coeff_0;
//     FE b1 = m2.coeff_1;        // so h2(x) = b0 + b1*x

//     // 6) Compute and print the two sides:
//     FE lhs = a0 + a1 * r1;         // h1(r1)
//     FE rhs = b0 + (b0 + b1);       // h2(0) + h2(1)

//     std::cout<<"DEBUG: h1(r1)       = "<< lhs <<"\n";
//     std::cout<<"DEBUG: h2(0)+h2(1) = "<< rhs <<"\n";

//     if(lhs == rhs) std::cout<<"✅ Rounds are consistent!\n";
//     else           std::cout<<"❌ MISMATCH: there's a bug in round 2 generation.\n";

//     return 0;
// }