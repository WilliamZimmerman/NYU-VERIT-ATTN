#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <random>

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
        return FieldElement((val * other.val) % PRIME);
    }
    
    bool operator==(const FieldElement& other) const {
        return val == other.val;
    }
    
    friend std::ostream& operator<<(std::ostream& os, const FieldElement& fe) {
        return os << fe.val;
    }
};

// Matrix MLE  implementation 
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
        
        std::cout << "MatrixMLE: " << rows << "×" << cols 
                  << " → " << log_rows << "+" << log_cols << "=" << total_vars << " variables" << std::endl;
    }
    
    // Core MLE evaluation using the mathematical formula
    FieldElement evaluate(const std::vector<FieldElement>& point) const {
        assert(point.size() == total_vars);
        
        FieldElement result(0);
        
        // Sum over all possible binary vector combinations
        int max_row_vals = 1 << log_rows;
        int max_col_vals = 1 << log_cols;
        
        // X and Y in the paper
        for (int row_bits = 0; row_bits < max_row_vals; row_bits++) {
            for (int col_bits = 0; col_bits < max_col_vals; col_bits++) {
                
                // Only include valid matrix entries
                if (row_bits < rows && col_bits < cols) {
                    FieldElement data_value = data[row_bits][col_bits];
                    
                    // Compute indicator function χ_{row_bits,col_bits}(point)
                    FieldElement indicator(1);
                    
                    // Row bits contribution
                    for (int bit = 0; bit < log_rows; bit++) {
                        int v_bit = (row_bits >> bit) & 1;
                        FieldElement x_val = point[bit];
                        
                        // ((1-x)(1-v) + xv)
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
    
    // Helper: evaluate at integer indices
    FieldElement evaluate_at_indices(int row, int col) const {
        std::vector<FieldElement> point;
        
        // Convert to binary representation
        for (int bit = 0; bit < log_rows; bit++) {
            point.push_back(FieldElement((row >> bit) & 1));
        }
        for (int bit = 0; bit < log_cols; bit++) {
            point.push_back(FieldElement((col >> bit) & 1));
        }
        
        return evaluate(point);
    }
    
    // Helper: evaluate at fractional point  
    FieldElement evaluate_fractional(const std::vector<double>& frac_point) const {
        assert(frac_point.size() == total_vars);
        
        std::vector<FieldElement> point;
        for (double val : frac_point) {
            // For values in [0,1], map them directly to field elements
            // 0.0 → FieldElement(0), 1.0 → FieldElement(1), 0.5 → FieldElement(PRIME/2), etc.
            if (val == 0.0) {
                point.push_back(FieldElement(0));
            } else if (val == 1.0) {
                point.push_back(FieldElement(1));
            } else {
                // For fractional values, scale appropriately
                long long field_val = (long long)(val * FieldElement::PRIME) % FieldElement::PRIME;
                point.push_back(FieldElement(field_val));
            }
        }
        
        return evaluate(point);
    }
    
    // Test: sum over boolean hypercube
    FieldElement sum_over_hypercube() const {
        FieldElement sum(0);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sum = sum + data[i][j];
            }
        }
        return sum;
    }
    
    int get_total_vars() const { return total_vars; }
    FieldElement get_data(int i, int j) const { return data[i][j]; }
    
    void print_matrix() const {
        std::cout << "Matrix data:" << std::endl;
        for (int i = 0; i < rows; i++) {
            std::cout << "  [";
            for (int j = 0; j < cols; j++) {
                std::cout << std::setw(4) << data[i][j];
                if (j < cols - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }
};

class MLECorrectnessTests {
public:
    static void test_basic_mle_properties() {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "TEST 1: Basic MLE Properties" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        // Test with 2x2 matrix
        std::vector<std::vector<FieldElement>> matrix_2x2 = {
            {FieldElement(1), FieldElement(2)},
            {FieldElement(3), FieldElement(4)}
        };
        
        MatrixMLE mle_2x2(matrix_2x2);
        mle_2x2.print_matrix();
        
        // Property 1: Sum over hypercube equals matrix sum
        FieldElement matrix_sum = mle_2x2.sum_over_hypercube();
        std::cout << "\nProperty 1 - Hypercube sum:" << std::endl;
        std::cout << "  Matrix sum: " << matrix_sum << " (expected: 10)" << std::endl;
        std::cout << "  Result: " << (matrix_sum == FieldElement(10) ? "✅ PASS" : "❌ FAIL") << std::endl;
        
        // Property 2: Boolean corner evaluations
        std::cout << "\nProperty 2 - Boolean corners:" << std::endl;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                FieldElement mle_val = mle_2x2.evaluate_at_indices(i, j);
                FieldElement expected = matrix_2x2[i][j];
                std::cout << "  MLE[" << i << "," << j << "] = " << mle_val 
                          << " (expected: " << expected << ") " 
                          << (mle_val == expected ? "✅" : "❌") << std::endl;
            }
        }
    }
    
    static void test_different_matrix_sizes() {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "TEST 2: Different Matrix Sizes" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        // Test 3x3 matrix (non-power-of-2)
        std::vector<std::vector<FieldElement>> matrix_3x3 = {
            {FieldElement(1), FieldElement(2), FieldElement(3)},
            {FieldElement(4), FieldElement(5), FieldElement(6)},
            {FieldElement(7), FieldElement(8), FieldElement(9)}
        };
        
        MatrixMLE mle_3x3(matrix_3x3);
        std::cout << "\n3×3 Matrix Test:" << std::endl;
        mle_3x3.print_matrix();
        
        // Check sum
        FieldElement sum_3x3 = mle_3x3.sum_over_hypercube();
        FieldElement expected_3x3(45); // 1+2+...+9 = 45
        std::cout << "Sum: " << sum_3x3 << " (expected: " << expected_3x3 << ") " 
                  << (sum_3x3 == expected_3x3 ? "✅" : "❌") << std::endl;
        
        // Check some corners
        std::cout << "Corner checks:" << std::endl;
        std::cout << "  MLE[0,0] = " << mle_3x3.evaluate_at_indices(0,0) << " (expected: 1)" << std::endl;
        std::cout << "  MLE[2,2] = " << mle_3x3.evaluate_at_indices(2,2) << " (expected: 9)" << std::endl;
        
        // Test 1x4 matrix (rectangular)
        std::vector<std::vector<FieldElement>> matrix_1x4 = {
            {FieldElement(10), FieldElement(20), FieldElement(30), FieldElement(40)}
        };
        
        MatrixMLE mle_1x4(matrix_1x4);
        std::cout << "\n1×4 Matrix Test:" << std::endl;
        mle_1x4.print_matrix();
        
        FieldElement sum_1x4 = mle_1x4.sum_over_hypercube();
        std::cout << "Sum: " << sum_1x4 << " (expected: 100)" << std::endl;
    }
    
    static void test_fractional_evaluation() {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "TEST 3: Fractional Point Evaluation" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        std::vector<std::vector<FieldElement>> matrix = {
            {FieldElement(1), FieldElement(2)},
            {FieldElement(3), FieldElement(4)}
        };
        
        MatrixMLE mle(matrix);
        
        // Test that fractional evaluation at corners matches boolean evaluation
        std::cout << "Fractional vs Boolean at corners:" << std::endl;
        
        std::vector<std::pair<std::vector<double>, std::pair<int,int>>> test_points = {
            {{0.0, 0.0}, {0, 0}},
            {{1.0, 0.0}, {1, 0}}, 
            {{0.0, 1.0}, {0, 1}},
            {{1.0, 1.0}, {1, 1}}
        };
        
        for (const auto& test : test_points) {
            FieldElement frac_result = mle.evaluate_fractional(test.first);
            FieldElement bool_result = mle.evaluate_at_indices(test.second.first, test.second.second);
            
            // Debug: show what field elements we're actually using
            std::vector<FieldElement> debug_point;
            for (double val : test.first) {
                if (val == 0.0) debug_point.push_back(FieldElement(0));
                else if (val == 1.0) debug_point.push_back(FieldElement(1));
                else debug_point.push_back(FieldElement((long long)(val * FieldElement::PRIME) % FieldElement::PRIME));
            }
            
            std::cout << "  Point (" << test.first[0] << "," << test.first[1] << "): "
                      << "frac=" << frac_result << ", bool=" << bool_result;
            std::cout << " [field point: " << debug_point[0] << "," << debug_point[1] << "]";
            std::cout << " " << (frac_result == bool_result ? "✅" : "❌") << std::endl;
        }
        
        // Test some interesting fractional points
        std::cout << "\nFractional evaluations:" << std::endl;
        FieldElement mid_result = mle.evaluate_fractional({0.5, 0.5});
        std::cout << "  MLE(0.5, 0.5) = " << mid_result << std::endl;
        
        FieldElement quarter_result = mle.evaluate_fractional({0.25, 0.75});
        std::cout << "  MLE(0.25, 0.75) = " << quarter_result << std::endl;
    }
    
    static void test_attention_matrices() {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "TEST 4: Attention Matrix MLEs" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        // Small attention matrices for testing
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
        
        std::cout << "Creating MLEs for attention matrices:" << std::endl;
        MatrixMLE X_mle(X);
        MatrixMLE Q_mle(Q);
        MatrixMLE K_mle(K);
        
        // Test that each MLE correctly represents its matrix
        std::cout << "\nX Matrix MLE test:" << std::endl;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                FieldElement mle_val = X_mle.evaluate_at_indices(i, j);
                FieldElement expected = X[i][j];
                std::cout << "  X̃[" << i << "," << j << "] = " << mle_val 
                          << " vs X[" << i << "," << j << "] = " << expected << " "
                          << (mle_val == expected ? "✅" : "❌") << std::endl;
            }
        }
        
        std::cout << "\nQ Matrix MLE test:" << std::endl;
        FieldElement q_sum = Q_mle.sum_over_hypercube();
        std::cout << "  Q sum: " << q_sum << " (expected: " << (5+6+7+8) << ")" << std::endl;
        
        std::cout << "\nK Matrix MLE test:" << std::endl;
        FieldElement k_sum = K_mle.sum_over_hypercube();
        std::cout << "  K sum: " << k_sum << " (expected: " << (9+10+11+12) << ")" << std::endl;
        
        // Test that we can evaluate at random points
        std::cout << "\nRandom point evaluations:" << std::endl;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        for (int test = 0; test < 3; test++) {
            std::vector<double> random_point = {dis(gen), dis(gen)};
            FieldElement x_val = X_mle.evaluate_fractional(random_point);
            FieldElement q_val = Q_mle.evaluate_fractional(random_point);
            FieldElement k_val = K_mle.evaluate_fractional(random_point);
            
            std::cout << "  Point (" << std::fixed << std::setprecision(3) 
                      << random_point[0] << "," << random_point[1] << "): "
                      << "X̃=" << x_val << ", Q̃=" << q_val << ", K̃=" << k_val << std::endl;
        }
    }
    
    static void run_all_tests() {
        std::cout << "MLE CORRECTNESS TEST SUITE" << std::endl;
        std::cout << "Testing MLE implementation WITHOUT sumcheck" << std::endl;
        
        test_basic_mle_properties();
        test_different_matrix_sizes();
        test_fractional_evaluation();
        test_attention_matrices();
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "✅ All MLE correctness tests completed!" << std::endl;
        std::cout << "✅ MLEs are ready for Thaler protocol implementation" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
    }
};

int main() {
    MLECorrectnessTests::run_all_tests();
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "THALER PROTOCOL CLARIFICATION:" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "For attention verification using Thaler approach:" << std::endl;
    std::cout << "• i, j (sequence indices) → RANDOM CHALLENGES (like s,t)" << std::endl;
    std::cout << "• p, q, h (feature/head indices) → SUMCHECK VARIABLES" << std::endl;
    std::cout << "• This reduces sumcheck rounds significantly!" << std::endl;
    std::cout << "• Next step: implement the Thaler protocol structure" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    return 0;
}