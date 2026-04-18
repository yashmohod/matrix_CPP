# matrix

A templated C++ matrix library built from scratch — no dependencies, no external linear algebra abstractions borrowed from elsewhere. Every line written by hand, every algorithm understood before implemented.

---

## What this is

This library is the second project in a C++ learning ladder targeting quant finance infrastructure. The goal was not to replicate Eigen — it was to deeply understand the mechanics that libraries like Eigen are built on: flat memory layout, operator overloading, LU decomposition, Gauss-Jordan elimination, and numerical stability through partial pivoting.

The entire project followed an API-first workflow:

```
design API → write usage code → write tests → implement
```

152 tests were written and reviewed before a single line of implementation existed. Every test was written by hand with explicit expected values computed manually — no auto-generated test cases.

---

## Features

### Construction
```cpp
Matrix<double> a(3, 3);                          // zero-initialized
Matrix<double> b(3, 3, 1.0);                     // fill with constant
Matrix<double> c(2, 2, {1, 2, 3, 4});            // from initializer list
Matrix<double> d = Matrix<double>::identity(3);  // static factory
Matrix<double> e = Matrix<double>::ones(3, 3);   // static factory
```

### Element access
```cpp
double val = c(0, 1);   // read — bounds checked, throws std::out_of_range
c(1, 1) = 99.0;         // write — bounds checked
```

### Shape inspection
```cpp
a.rows();           // size_t
a.cols();           // size_t
a.size();           // rows * cols
a.shape();          // std::pair<size_t, size_t>
a.is_square();      // bool
a.is_zero();        // bool
a.is_identity();    // bool
a.is_symmetric();   // bool — checks A(i,j) == A(j,i), requires square
```

### Arithmetic
```cpp
auto sum  = A + B;          // matrix addition
auto diff = A - B;          // matrix subtraction
auto prod = A * B;          // matrix multiplication (not element-wise)
auto s1   = A * 2.0;        // scalar multiply
auto s2   = 2.0 * A;        // scalar multiply (non-member, commutative)
auto q    = A / 2.0;        // scalar divide
auto neg  = -A;             // unary minus
A += B;  A -= B;            // in-place matrix ops
A *= 2.0;  A /= 2.0;        // in-place scalar ops
```

### Element-wise operations
```cpp
auto h = A.hadamard(B);                              // element-wise multiply
auto s = A.apply(std::sqrt);                         // apply any function
auto d = A.apply([](double x){ return x * 2.0; });  // lambda support
```

### Row and column extraction
```cpp
auto r = A.row(0);   // returns 1 × cols Matrix
auto c = A.col(1);   // returns rows × 1 Matrix
```

### Linear algebra
```cpp
A.trace();        // sum of diagonal — requires square
A.det();          // determinant via LU decomposition with partial pivoting
A.norm();         // Frobenius norm — sqrt(sum of squares of all elements)
A.inverse();      // Gauss-Jordan elimination — throws if singular
A.solve(b);       // solves Ax = b — throws if singular
A.pow(2);         // matrix power — repeated matrix multiplication
A.transpose();    // returns new matrix, original unmodified
```

### Output
```cpp
A.print();        // formatted grid with aligned columns
A.print_shape();  // prints Matrix(3x4)
std::cout << A;   // operator<< delegates to print()
```

---

## Operator conventions

`A * B` is **matrix multiplication** — consistent with Eigen, Armadillo, and every production C++ math library.

`A.hadamard(B)` is **element-wise multiplication** — explicit naming avoids the ambiguity of NumPy's `*` convention, which is a historical artifact from before Python had the `@` operator.

---

## Exception hierarchy

| Exception | When thrown |
|---|---|
| `std::invalid_argument` | Shape mismatch, non-square matrix for square-only ops, zero dimensions, wrong initializer list size |
| `std::runtime_error` | Singular matrix passed to `inverse()` or `solve()` |
| `std::out_of_range` | Index out of bounds in `operator()`, `row()`, `col()` |

---

## Build and test

```bash
# compile
g++ -std=c++17 -o test_runner tests/test_matrix.cpp

# run
./test_runner
```

Expected output:
```
152 / 152 tests passed
```

No external dependencies. No build system required. Single header — everything lives in `include/matrix.hpp` because C++ templates require the full implementation to be visible at the point of instantiation.

---

## File structure

```
matrix/
  include/
    matrix.hpp        ← full class declaration + implementation
  tests/
    test_matrix.cpp   ← 152 tests, hand-written before implementation
  main.cpp            ← usage examples for every operation
  README.md
```

---

## C++ concepts covered

This project was designed as a learning exercise. Each section of the implementation targets a specific concept:

**Templates** — `Matrix<T>` works for `double`, `float`, `int`, or any numeric type. The template instantiation model explains why all implementation lives in the header.

**Rule of Three** — explicit copy constructor, copy assignment operator, and destructor. Written deliberately even though `std::vector` would handle it automatically — the goal was to understand what the compiler generates and why.

**Operator overloading** — member operators for arithmetic, non-member operators for left-scalar multiply (`2.0 * A`) and `operator<<`. Understanding when each form is required.

**Const correctness** — two versions of `operator()`: a const read-only version returning `const T&` and a non-const write version returning `T&`. Every read-only method marked `const`.

**Flat memory layout** — a 2D matrix stored as a 1D `std::vector<T>` with row-major indexing `i * cols + j`. Foundation for cache-friendly traversal and eventual SIMD/CUDA optimization.

**LU decomposition with partial pivoting** — finding the pivot row by maximum absolute value at each step, swapping rows, tracking sign changes for the determinant. O(n³) vs the O(n!) cofactor expansion.

**Gauss-Jordan elimination** — augmenting A with an identity matrix, running forward and backward elimination passes, scaling rows to produce the inverse.

**Exception handling** — `std::invalid_argument` vs `std::runtime_error` vs `std::out_of_range` — each chosen based on whether the problem is the argument type, the mathematical state, or the index range.

---

## Numerical stability

All floating point comparisons use an epsilon threshold of `1e-9` rather than hard equality:

- `is_zero()` — checks `|element| < 1e-9` rather than `element == 0`
- `is_identity()` — checks diagonal within `1e-9` of 1, off-diagonal within `1e-9` of 0
- `is_symmetric()` — checks `|A(i,j) - A(j,i)| < 1e-9`
- Singular detection — checks `|det| < 1e-9` before inverse and solve

LU decomposition uses partial pivoting — at each elimination step, the row with the largest absolute value in the current column is swapped to the pivot position. This keeps multipliers bounded to `[-1, 1]` and prevents error amplification from near-zero pivots.

---

## Future versions

This is v1 — the intentionally naive implementation. The roadmap:

### v2 — SIMD vectorization (AVX2)

The flat row-major storage layout was chosen with this in mind. AVX2 processes 4 doubles simultaneously using 256-bit registers. Target operations for vectorization:

- `operator+`, `operator-`, `hadamard()` — pure element-wise, ideal for SIMD
- `operator*` scalar — single broadcast + multiply across the data vector
- Matrix multiply inner loop — the hot path, most impactful target

v2 will ship with a benchmark suite comparing throughput across matrix sizes (32×32, 128×128, 512×512, 1024×1024) using `std::chrono`. The performance gap between scalar and SIMD versions narrows at small sizes due to overhead and widens at large sizes where memory bandwidth dominates.

### v3 — CUDA parallelization

The analytics engine phase of this project will use this library as its foundation. Monte Carlo simulation requires generating thousands of correlated random paths — this is done via Cholesky decomposition followed by matrix-vector multiplication of the lower triangular factor with a vector of independent normals.

Cholesky is the next major algorithm to add, both for the quant use case and because it's numerically better conditioned than LU for symmetric positive-definite matrices (covariance matrices are always SPD).

CUDA targets:
- Monte Carlo path generation — embarrassingly parallel, ideal for GPU
- Large matrix multiply — tiled shared memory kernel
- Covariance matrix computation from returns data

Benchmark: scalar v1 vs SIMD v2 vs CUDA v3, measured in paths/second for the Monte Carlo use case.

### Quant integration (analytics engine)

The matrix library is the foundation layer for a backtesting analytics engine planned for August. Specific operations needed:

```cpp
// covariance matrix from returns data
Matrix<double> cov = covariance_matrix(returns);

// Cholesky factor for correlated path generation
Matrix<double> L = cov.cholesky();

// portfolio variance
double var = portfolio_variance(weights, cov);   // wᵀΣw
```

These operations sit directly on top of what is already implemented — `transpose()`, `operator*`, and the existing linear algebra stack.

---

## Connection to research

The LU decomposition and Gaussian elimination implemented here are the same numerical methods used in the finite difference PDE solver published in *American Journal of Physics* (Vol. 93, 2025) for thermal diffusion. The Black-Scholes PDE reduces to the heat equation under a change of variables — the solver that prices options and the solver that models heat conduction are structurally identical. This library is the foundation for making that connection explicit in code.

---

## Author

Yash Mohod — B.S. Computer Science + B.S. Physics, Ithaca College (2025)
