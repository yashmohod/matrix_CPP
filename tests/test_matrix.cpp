#include "../include/matrix.hpp"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cassert>

// ── Test harness ──────────────────────────────────────────────────────────────

int tests_run    = 0;
int tests_passed = 0;

void check(bool condition, const std::string& test_name) {
    tests_run++;
    if (condition) {
        tests_passed++;
        std::cout << "  PASS  " << test_name << "\n";
    } else {
        std::cout << "  FAIL  " << test_name << "\n";
    }
}

bool approx(double a, double b, double eps = 1e-9) {
    return std::abs(a - b) < eps;
}

bool mat_approx(const Matrix<double>& A, const Matrix<double>& B, double eps = 1e-9) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) return false;
    for (size_t i = 0; i < A.rows(); i++)
        for (size_t j = 0; j < A.cols(); j++)
            if (!approx(A(i,j), B(i,j), eps)) return false;
    return true;
}

template <typename ExceptionType, typename Func>
bool throws(Func f) {
    try { f(); return false; }
    catch (const ExceptionType&) { return true; }
    catch (...) { return false; }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

void test_construction() {
    std::cout << "\n── Construction ──\n";

    Matrix<double> a(2, 4);
    check(a.rows() == 2, "zeros - matrix rows"); 
    check(a.cols() == 4, "zeros - matrix cols"); 
    check(approx(a(0,0),0) && approx(a(1,1),0), "zeros - matrix zero values"); 

    Matrix<double> b(3, 5,1.0);
    check(b.rows() == 3, "ones - matrix rows"); 
    check(b.cols() == 5, "ones - matrix cols"); 
    check(approx(b(0,0), 1), "ones - (0,0) == 1");
    check(approx(b(1,2), 1), "ones - (1,3) == 1");
    check(approx(b(2,4), 1), "ones - (2,4) == 1");
    
    Matrix<double> c(2, 2, {1,2,3,4});
    check(c.rows() == 2, "list - matrix rows"); 
    check(c.cols() == 2, "list - matrix cols"); 
    check(approx(c(0,0), 1.0), "list - (0,0) == 1");
    check(approx(c(0,1), 2.0), "list - (0,1) == 2");
    check(approx(c(1,0), 3.0), "list - (1,0) == 3");
    check(approx(c(1,1), 4.0), "list - (1,1) == 4");
    
    
    Matrix<double> d = Matrix<double>::identity(5);
    check(d.rows() == 5, "identity - matrix rows"); 
    check(d.cols() == 5, "identity - matrix cols"); 
    check(approx(d(0,0), 1), "identity - (0,0) == 1");
    check(approx(d(1,1), 1), "identity - (1,1) == 1");
    check(approx(d(2,3), 0), "identity - (2,3) == 0");
    check(approx(d(3,1), 0), "identity - (3,1) == 0");
    
    Matrix<double> e = Matrix<double>::ones(2, 2);
    check(e.rows() == 2, "ones - matrix rows"); 
    check(e.cols() == 2, "ones - matrix cols"); 
    check(approx(e(0,0), 1.0), "ones - (0,0) == 1");
    check(approx(e(0,1), 1.0), "ones - (0,1) == 1");
    check(approx(e(1,0), 1.0), "ones - (1,0) == 1");
    check(approx(e(1,1), 1.0), "ones - (1,1) == 1");

    // zero dimensions
    check(throws<std::invalid_argument>([]{ Matrix<double> bad(0, 2); }), "zero rows throws");
    check(throws<std::invalid_argument>([]{ Matrix<double> bad(2, 0); }), "zero cols throws");
    
    // initializer list wrong size
    check(throws<std::invalid_argument>([]{ Matrix<double> bad(2, 2, {1, 2, 3}); }), "list too short throws");
    check(throws<std::invalid_argument>([]{ Matrix<double> bad(2, 2, {1, 2, 3, 4, 5}); }), "list too long throws");
    
    // identity of size zero
    check(throws<std::invalid_argument>([]{ Matrix<double>::identity(0); }), "identity(0) throws");
    
}

void test_element_access() {
    std::cout << "\n── Element access ──\n";

    Matrix<double> a(2, 4);
    check(approx(a(0,0), 0.0),    "read init value");
    a(0,0) = 333.3333;
    check(approx(a(0,0), 333.3333), "write then read");
    a(0,0) = 2.55;
    check(approx(a(0,0), 2.55),   "overwrite existing value");
    check(throws<std::out_of_range>([&]{ a(5,0); }), "row out of range throws");
    check(throws<std::out_of_range>([&]{ a(0,5); }), "col out of range throws");
    check(throws<std::out_of_range>([&]{ a(2,4); }), "exact boundary throws");

}

void test_shape() {
    std::cout << "\n── Shape inspection ──\n";

    Matrix<double> a(2, 4);
    check(a.rows() == 2,    "rows()");
    check(a.cols() == 4,    "cols()");
    check(a.size() == 8,    "size()");
    
    auto [r, c] = a.shape();
    check(r == 2 && c == 4, "shape() structured binding");
    
    Matrix<double> sq(3, 3);
    check(sq.is_square(),        "is_square() true");
    check(sq.is_zero(),          "is_zero() true");
    check(!a.is_square(),        "is_square() false on non-square");
    
    Matrix<double> id = Matrix<double>::identity(3);
    check(id.is_identity(),      "is_identity() true");
    check(id.is_symmetric(),     "is_symmetric() true — identity is symmetric");
    
    Matrix<double> sym(2, 2, {1, 2, 2, 5});
    check(sym.is_symmetric(),    "is_symmetric() true");
    
    Matrix<double> asym(2, 2, {1, 2, 3, 5});
    check(!asym.is_symmetric(),  "is_symmetric() false");
    
    check(!a.is_symmetric(),     "is_symmetric() false on non-square");

}

void test_transpose() {
    std::cout << "\n── Transpose ──\n";

    Matrix<double> b(2, 5);
    check(b.rows() == 2, "Shape test row");
    check(b.cols() == 5, "Shape test col");
    auto a = b.transpose();
    check(a.rows() == 5, "Shape test row");
    check(a.cols() == 2, "Shape test col");
    
    Matrix<double> c(2, 3, {1, 2, 3, 4, 5, 6});
    auto cT = c.transpose();
    check(approx(cT(0,0), 1.0), "transpose (0,0)");
    check(approx(cT(1,0), 2.0), "transpose (1,0)");
    check(approx(cT(2,0), 3.0), "transpose (2,0)");
    check(approx(cT(0,1), 4.0), "transpose (0,1)");

    check(approx(c(0,0), 1.0), "original (0,0) unmodified after transpose");
    check(approx(c(0,1), 2.0), "original (0,1) unmodified after transpose");
    check(approx(c(1,2), 6.0), "original (1,2) unmodified after transpose");

    check(mat_approx(c.transpose().transpose(), c), "double transpose == original");
    
    Matrix<double> sq(2, 2, {1, 2, 3, 4});
    auto sqT = sq.transpose();
    check(approx(sqT(0,1), 3.0), "square transpose (0,1)");
    check(approx(sqT(1,0), 2.0), "square transpose (1,0)");



}

void test_arithmetic() {
    std::cout << "\n── Arithmetic ──\n";

    Matrix<double> a(2,2,{2,4,6,8});
    Matrix<double> b(2,2,{2,2,2,2});
    double c = 10.0;
    Matrix<double> d = a;

    auto sumM = a+b;
    d+=b; 
    check(approx(sumM(0,0), 4.0), "mat + mat check (0,0)");
    check(approx(sumM(0,1), 6.0), "mat + mat check (0,1)");
    check(approx(sumM(1,0), 8.0), "mat + mat check (1,0)");
    check(approx(sumM(1,1), 10.0), "mat + mat check (1,1)");
    check(mat_approx(sumM,d), "mat + mat ==  mat += mat");
    
    auto sumC = a+c;
    d = a;
    d +=c;
    check(approx(sumC(0,0), 12.0), "mat + const check (0,0)");
    check(approx(sumC(0,1), 14.0), "mat + const check (0,1)");
    check(approx(sumC(1,0), 16.0), "mat + const check (1,0)");
    check(approx(sumC(1,1), 18.0), "mat + const check (1,1)");
    check(mat_approx(sumC,d), "mat + mat ==  mat += mat");

    auto diffM = a-b;
    d = a;
    d -=b;
    check(approx(diffM(0,0), 0.0), "mat - mat check (0,0)");
    check(approx(diffM(0,1), 2.0), "mat - mat check (0,1)");
    check(approx(diffM(1,0), 4.0), "mat - mat check (1,0)");
    check(approx(diffM(1,1), 6.0), "mat - mat check (1,1)");
    check(mat_approx(diffM,d), "mat - mat ==  mat -= mat");
    
    auto diffC = a-c;
    d = a;
    d -=c;
    check(approx(diffC(0,0), -8.0), "mat - const check (0,0)");
    check(approx(diffC(0,1), -6.0), "mat - const check (0,1)");
    check(approx(diffC(1,0), -4.0), "mat - const check (1,0)");
    check(approx(diffC(1,1), -2.0), "mat - const check (1,1)");
    check(mat_approx(diffC,d), "mat - const ==  mat -= const");

    auto divC = a/c;
    d = a;
    d /=c;
    check(approx(divC(0,0), 0.2), "mat/const check (0,0)");
    check(approx(divC(0,1), 0.4), "mat/const check (0,1)");
    check(approx(divC(1,0), 0.6), "mat/const check (1,0)");
    check(approx(divC(1,1), 0.8), "mat/const check (1,1)");
    check(mat_approx(divC,d), "mat/cosnt ==  mat /= const");
    
    auto mulM = a*b;
    d = a;
    d *= b;
    check(approx(mulM(0,0), 12.0), "mat x mat check (0,0)");
    check(approx(mulM(0,1), 12.0), "mat x mat check (0,1)");
    check(approx(mulM(1,0), 28.0), "mat x mat check (1,0)");
    check(approx(mulM(1,1), 28.0), "mat x mat check (1,1)");
    check(mat_approx(mulM,d), "mat x mat ==  mat *= mat");

    auto mulC = a*c;
    auto mulC2 = c*a;
    d = a;
    d *= c;
    check(approx(mulC(0,0), 20.0), "mat x const check (0,0)");
    check(approx(mulC(0,1), 40.0), "mat x const check (0,1)");
    check(approx(mulC(1,0), 60.0), "mat x const check (1,0)");
    check(approx(mulC(1,1), 80.0), "mat x const check (1,1)");
    check(mat_approx(mulC,mulC2), "mat x const == const x mat");
    check(mat_approx(mulC,d), "mat x const == mat *= const");

    
    d = -a;
    b = a*-1.0;
    check(mat_approx(b,d), "mat * -1 == -mat");


    Matrix<double> wide = Matrix<double>::ones(2,4);
    Matrix<double> tall = Matrix<double>::ones(3,2);
    d = a * wide; 
    check(d.rows() == 2, " rows check after mat mul");
    check(d.cols() == 4, " cols check after mat mul");
    check(d.size() == 8, " size check after mat mul");

    check(throws<std::invalid_argument>([&]{ auto x = a + tall; }), "add mismatch throws");
    check(throws<std::invalid_argument>([&]{ auto x = a - tall; }), "sub mismatch throws");
    check(throws<std::invalid_argument>([&]{ auto x = a * tall; }), "mul mismatch throws");
    check(throws<std::invalid_argument>([&]{ auto x = a / 0.0; }), "divide by zero throws");

    Matrix<double> id = Matrix<double>::identity(2);
    Matrix<double> z(2, 2);
    check(mat_approx(a + z, a),    "A + 0 == A");
    check(mat_approx(a - a, z),    "A - A == 0");
    check(mat_approx(a * id, a),   "A * I == A");
    check(mat_approx(id * a, a),   "I * A == A");

}

void test_elementwise() {
    std::cout << "\n── Element-wise ops ──\n";
    
    Matrix<double> a(2,2,{2,4,6,8});
    Matrix<double> b(2,2,{1,2,3,4}); 
    Matrix<double> tall = Matrix<double>::ones(3,2);

    auto hadamard = a.hadamard(b);
    check(approx(hadamard(0,0), 2.0), "hadamard check (0,0)");
    check(approx(hadamard(0,1), 8.0), "hadamard check (0,1)");
    check(approx(hadamard(1,0), 18.0),"hadamard check (1,0)");
    check(approx(hadamard(1,1), 32.0),"hadamard check (1,1)");
    check(approx(a(0,0), 2.0), "original unmodified after hadamard");
    check(approx(b(0,0), 1.0), "original unmodified after hadamard");
    check(mat_approx(a.hadamard(b), b.hadamard(a)), "hadamard is commutative");
    check(throws<std::invalid_argument>([&]{ auto x = a.hadamard(tall); }), "hadamard mismatch throws");
    

    Matrix<double> c(2,2,{1,4,9,16});
 
    auto sq_root  = c.apply([](double x){return std::sqrt(x);});
    check(approx(sq_root(0,0), 1.0), "sq_root check (0,0)");
    check(approx(sq_root(0,1), 2.0), "sq_root check (0,1)");
    check(approx(sq_root(1,0), 3.0),"sq_root check (1,0)");
    check(approx(sq_root(1,1), 4.0),"sq_root check (1,1)"); 
    check(approx(c(0,0), 1.0), "original unmodified after apply");

    auto doubled  = b.apply([](double x) { return x * 2.0; });
    check(mat_approx(doubled,a), "doubled check");

}

void test_row_col() {
    std::cout << "\n── Row / col extraction ──\n";
    
    Matrix<double> a(2,2,{1,2,3,4});
    auto a_row = a.row(0);
    auto a_col = a.col(0);
    
    check(approx(a_row(0,0), 1.0), "a_row check (0,0)");
    check(approx(a_row(0,1), 2.0), "a_row check (0,1)");
    check(approx(a_col(0,0), 1.0), "a_col check (0,0)");
    check(approx(a_col(1,0), 3.0), "a_col check (1,0)");
     
    check(a_row.rows() == 1, "a_row check rows");
    check(a_row.cols() == 2, "a_row check cols");
    check(a_col.rows() == 2, "a_col check rows");
    check(a_col.cols() == 1, "a_col check cols");

    check(throws<std::out_of_range>([&]{ auto x = a.row(2); }), "row exact boundary throws");
    check(throws<std::out_of_range>([&]{ auto x = a.col(2); }), "col exact boundary throws");
    check(throws<std::out_of_range>([&]{ auto x = a.row(4); }), "row out of bound throws");
    check(throws<std::out_of_range>([&]{ auto x = a.col(4); }), "col out of bound throws");


}

void test_linear_algebra() {
    std::cout << "\n── Linear algebra ──\n";
    
    Matrix<double> a(4,4,{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16});
    auto a_det = a.det();
    check(approx(a_det, 0.0) ,"Check determinant");
    
    Matrix<double> b(3,4,{1,2,3,4,5,6,7,8,9,10,11,12});
    check(throws<std::invalid_argument>([&]{auto b_det = b.det();}),"non square mat throws");
    
    Matrix<double> id = Matrix<double>::identity(10);
    check(approx(id.trace(), 10.0) ,"Check trace");

    Matrix<double> inv(2, 2, {1, 2, 4, 5});
    Matrix<double> id_inv = Matrix<double>::identity(2);
    check(mat_approx(inv * inv.inverse(), id_inv, 1e-9), "A * A⁻¹ == I");
    check(mat_approx(inv.inverse() * inv, id_inv, 1e-9), "A⁻¹ * A == I");
    check(approx(inv.det(), -3.0), "det non-zero known value"); 


    Matrix<double> c(2, 2, {1, 3, 5, 2});
    check(approx(c.norm(), std::sqrt(39.0)), "frobenius norm");
    
    Matrix<double> A(2, 2, {1, 2, 3, 4});
    Matrix<double> z(2, 1, {5, 11});
    auto x = A.solve(z);
    check(approx(x(0,0), 1.0), "solve x(0)");
    check(approx(x(1,0), 2.0), "solve x(1)");    
    check(throws<std::invalid_argument>([&]{auto x = A.solve(b);}),"mat size mismatch");
    
    auto sq = A.pow(2);
    check(mat_approx(sq, A*A), "pow(2) == A*A");

    // inverse of singular matrix
    Matrix<double> singular(2, 2, {1, 2, 2, 4});
    check(throws<std::runtime_error>([&]{ singular.inverse(); }), "singular inverse throws");
    
    // non-square throws
    Matrix<double> wide(2, 3);
    check(throws<std::invalid_argument>([&]{ wide.inverse(); }), "non-square inverse throws");
    check(throws<std::invalid_argument>([&]{ wide.trace(); }),   "non-square trace throws");
    check(throws<std::invalid_argument>([&]{ wide.det(); }),     "non-square det throws");

}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    test_construction();
    test_element_access();
    test_shape();
    test_transpose();
    test_arithmetic();
    test_elementwise();
    test_row_col();
    test_linear_algebra();

    std::cout << "\n────────────────────────────────\n";
    std::cout << tests_passed << " / " << tests_run << " tests passed\n";

    return (tests_passed == tests_run) ? 0 : 1;
}
