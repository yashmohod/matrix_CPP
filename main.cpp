#include <iostream>
#include "include/matrix.hpp"

int main() {

    // ── Construction ──────────────────────────────────────────
    Matrix<double> a(2, 2);
    Matrix<double> b(2, 2, 1.0);
    Matrix<double> c(2, 2, {1, 2, 3, 4});
    Matrix<double> d = Matrix<double>::identity(2);
    Matrix<double> f = Matrix<double>::ones(2, 2);

    // ── Element access ────────────────────────────────────────
    double val = c(0, 1);
    c(1, 1) = 99.0;

    // ── Shape inspection ──────────────────────────────────────
    auto [rows, cols] = a.shape();
    a.print_shape();

    // ── Transpose ─────────────────────────────────────────────
    auto cT = c.T();

    // ── Arithmetic ────────────────────────────────────────────
    auto sum        = a + b;
    auto difference = b - a;
    auto product1   = b * 2.0;
    auto product2   = 2.0 * b;
    auto mat_product = a * b;       // matrix multiplication
    auto quotient   = b / 2.0;
    a += b;
    a -= b;
    a *= 2.0;
    a /= 2.0;

    // ── Element-wise ops ──────────────────────────────────────
    auto hadamard = a.hadamard(b);
    auto sq_root  = c.apply(std::sqrt);
    auto doubled  = c.apply([](double x) { return x * 2.0; });

    // ── Row / col extraction ──────────────────────────────────
    auto row0 = c.row(0);           // returns 1×cols Matrix
    auto col1 = c.col(1);           // returns rows×1 Matrix

    // ── Linear algebra ────────────────────────────────────────
    auto determinant = a.det();
    auto trace       = a.trace();
    auto inverse     = a.inverse();
    auto norm        = a.norm();
    auto x           = a.solve(b);
    auto sq          = a.pow(2);

    // ── Output ────────────────────────────────────────────────
    b.print();

    return 0;
}
