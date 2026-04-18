#pragma once
#include <vector>
#include <iostream>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <iomanip>

template <typename T>
class Matrix{

  public:

    // Constructors
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols,T fill);
    Matrix(size_t rows, size_t cols, std::initializer_list<T> values);

    // Static factories
    static Matrix<T> identity(size_t n);
    static Matrix<T> ones(size_t rows, size_t cols);

    // Rule of three
    Matrix(const Matrix<T>& other);
    Matrix<T>& operator=(const Matrix<T>& other);
    ~Matrix();

    // Element access
    T& operator()(size_t i, size_t j);
    const T& operator()(size_t i, size_t j) const;

    // Shape inspection
    size_t rows() const;
    size_t cols() const;
    size_t size() const;
    std::pair<size_t, size_t> shape() const;
    bool is_square() const;
    bool is_zero() const;
    bool is_identity() const;
    bool is_symmetric() const;

    // Transpose
    Matrix<T> transpose() const;

    // Arithemetic operator
    Matrix<T> operator+(const Matrix<T>& other) const;
    Matrix<T> operator-(const Matrix<T>& other) const;
    Matrix<T> operator*(const Matrix<T>& other) const;
    Matrix<T> operator+(T scalar) const;
    Matrix<T> operator-(T scalar) const;
    Matrix<T> operator*(T scalar) const;
    Matrix<T> operator/(T scalar) const;
    Matrix<T> operator-() const;

    Matrix<T>& operator+=(const Matrix<T>& other);
    Matrix<T>& operator-=(const Matrix<T>& other);
    Matrix<T>& operator*=(const Matrix<T>& other);
    Matrix<T>& operator+=(T scalar);
    Matrix<T>& operator-=(T scalar);
    Matrix<T>& operator*=(T scalar);
    Matrix<T>& operator/=(T scalar);


    // element-wise ops 
    Matrix<T> hadamard(const Matrix<T>& other) const;
    Matrix<T> apply(std::function<T(T)> func) const;

    // row/col extraction
    Matrix<T> row(size_t i) const;
    Matrix<T> col(size_t j) const;

    // linear algebra
    T trace() const;
    T det() const;
    T norm() const;
    Matrix<T> inverse() const;
    Matrix<T> solve(const Matrix<T>& b) const;
    Matrix<T> pow(size_t exp) const;

    // output
    void print() const;
    void print_shape() const;

  private:
    size_t rows_;
    size_t cols_;
    std::vector<T> data_;

    // private helper
    size_t idx(size_t i, size_t j) const;

};

// idx
template <typename T>
size_t Matrix<T>::idx(size_t i, size_t j)const{
  return i*cols_ + j;
}

// CONSTRUCTORS

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols)
  : rows_(rows),cols_(cols),data_(rows*cols, T{}){ 
    if (rows == 0 || cols == 0)
      throw std::invalid_argument("dimentions must be > 0");
}

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols,T fill)
  :rows_(rows),cols_(cols),data_(rows*cols, fill){
    if (rows == 0 || cols == 0)
      throw std::invalid_argument("dimentions must be > 0");
}
template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, std::initializer_list<T> values)
  :rows_(rows),cols_(cols),data_(values){
    if(rows == 0 || cols == 0)
      throw std::invalid_argument("dimentions must be > 0");
    
    if(values.size() != rows*cols)
      throw std::invalid_argument("initializer list size does not match dimensions");
}

// STATIC FACTORIES

template <typename T>
Matrix<T> Matrix<T>::identity(size_t n){
  Matrix<T> result(n,n);
  for(size_t i = 0; i < n; i++)
    result(i,i) =T{1};
  return result;
}

template <typename T>
Matrix<T> Matrix<T>::ones(size_t rows, size_t cols){
  return Matrix<T>(rows,cols,T{1});
}

// RULE OF THREE

// copy constructor
template <typename T>
Matrix<T>::Matrix(const Matrix<T>& other)
  :rows_(other.rows_),cols_(other.cols_),data_(other.data_){}

// copy assignment
template <typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other){
  if (this == &other) return *this;
  rows_ = other.rows_;
  cols_ = other.cols_;
  data_ = other.data_;
  return *this;
}

// destructor
template <typename T>
Matrix<T>::~Matrix(){}


// ELEMENT ACCESS

template <typename T>
T& Matrix<T>::operator()(size_t i, size_t j){
  if(i >= rows_ || j >= cols_ )
    throw std::out_of_range("index out of range");
  return data_[idx(i,j)];
}

template <typename T>
const T& Matrix<T>::operator()(size_t i, size_t j) const{
  if(i >= rows_ || j >= cols_)
    throw std::out_of_range("index out of range");
  return data_[idx(i,j)];
}


// SHAPE INSPECTION

template <typename T>
size_t Matrix<T>::rows() const{
  return rows_;
}

template <typename T>
size_t Matrix<T>::cols() const{
  return cols_;
}

template <typename T>
size_t Matrix<T>::size() const{
  return rows_*cols_;
}

template <typename T>
std::pair<size_t, size_t> Matrix<T>::shape() const{
  return std::pair(rows_,cols_);
}

template <typename T>
bool Matrix<T>::is_square() const{
  return rows_ == cols_;
}

template <typename T>
bool Matrix<T>::is_zero() const{
  for(size_t x =0; x < data_.size();x++){
    if(data_[x] > 1e-9)
      return false;
  }
  return true;
}

template <typename T>
bool Matrix<T>::is_identity() const{
  if(!is_square()) return false;
for(size_t row = 0; row < rows_;row++){
    for(size_t col = 0; col < cols_ ; col++){
      if(col == row){
        if(std::abs(data_[idx(row,col)] - 1) > 1e-9)
          return false;
      }else{
        if(std::abs(data_[idx(row,col)])> 1e-9)
          return false; 
      }
    }
  }
  return true;
}

template <typename T>
bool Matrix<T>::is_symmetric() const{
  if(!is_square()) return false;
  for(size_t row = 0; row < rows_;row++){
    for(size_t col = row; col < cols_ ; col++){
      if(std::abs(data_[idx(row,col)] - data_[idx(col,row)])> 1e-9)
        return false;
    }
  }
  return true;
}

// Transpose

template <typename T>
Matrix<T> Matrix<T>::transpose() const{
  Matrix<T> result(cols_, rows_);
  for (size_t row = 0; row < rows_; row++)
    for (size_t col = 0; col < cols_; col++)
      result(col, row) = data_[idx(row, col)];
  return result;
}

// Arithemetic operator

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const{
  if(other.rows_ != rows_ || other.cols_ != cols_){
    throw std::invalid_argument("matrix dimensions do not match");
  }
  Matrix<T> a(rows_,cols_);
  for(size_t row =0 ; row < rows_ ; row++){
    for(size_t col =0 ; col < cols_ ; col++){
      a(row,col) = data_[idx(row,col)] + other.data_[idx(row,col)];
    }
  }
  return a;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const{
  if(other.rows_ != rows_ || other.cols_ != cols_){
    throw std::invalid_argument("matrix dimensions do not match");
  }
  Matrix<T> a(rows_,cols_);
  for(size_t row =0 ; row < rows_ ; row++){
    for(size_t col =0 ; col < cols_ ; col++){
      a(row,col) = data_[idx(row,col)] - other.data_[idx(row,col)];
    }
  }
  return a;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const{
  if(cols_ != other.rows_){
    throw std::invalid_argument("matrix dimensions incompatible for multiplication");
  }
  
  Matrix<T> a(rows_,other.cols_);
  
  for(size_t i = 0; i < rows_; i++){
    for(size_t j = 0; j < other.cols_; j++){
      for(size_t k = 0; k < cols_; k++){
        a(i,j) += data_[idx(i,k)] * other.data_[other.idx(k,j)];
      }
    }
  }
  return a;
}

template <typename T>
Matrix<T> Matrix<T>::operator+(T scalar) const{
  Matrix<T> result(*this);   
  for (size_t i = 0; i < data_.size(); i++)
    result.data_[i] += scalar;
  return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(T scalar) const{
  Matrix<T> result(*this);
  for (size_t i = 0; i < data_.size(); i++)
    result.data_[i] -= scalar;
  return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(T scalar) const{
  Matrix<T> result(*this);
  for (size_t i = 0; i < data_.size(); i++)
    result.data_[i] *= scalar;
  return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator/(T scalar) const{
  if(std::abs(scalar) < 1e-9)
    throw std::invalid_argument("dvision by zero");
  Matrix<T> result(*this);
  for(size_t i = 0; i < data_.size(); i++)
    result.data_[i] /= scalar;
  return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator-() const{
  Matrix<T> result(*this);
  for (size_t i = 0; i < data_.size(); i++)
    result.data_[i] *= -1 ;
  return result;
}

template <typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& other){
  *this = *this + other;
  return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& other){
  *this = *this - other;
  return *this;
}

template  <typename T>
Matrix<T>& Matrix<T>::operator*=(const Matrix<T>& other){
  *this = *this * other;
  return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator+=(T scalar){
  *this = *this + scalar;
  return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator-=(T scalar){
  *this = *this - scalar;
  return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator*=(T scalar){
  *this = *this * scalar;
  return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator/=(T scalar){
  *this = *this / scalar;
  return *this;
}


// element-wise ops 
template <typename T>
Matrix<T> Matrix<T>::hadamard(const Matrix<T>& other) const{
  if(other.rows_ != rows_ || other.cols_ != cols_){
    throw std::invalid_argument("matrix dimensions do not match");
  }
 
  Matrix<T> result(*this);
  for(size_t i = 0; i < result.data_.size(); i++)
    result.data_[i] *= other.data_[i];
  return result; 
}

template <typename T>
Matrix<T> Matrix<T>::apply(std::function<T(T)> func) const{
  
  Matrix<T> result(*this);
  for(size_t i=0; i < result.data_.size();i++)
    result.data_[i] = func(result.data_[i]);
  return result;
}

// row/col extraction
template <typename T>
Matrix<T> Matrix<T>::row(size_t i) const{
  if(i >= rows_)
    throw std::out_of_range("row index out of range");
  Matrix<T> result(1,cols_);
  for(size_t col=0; col < cols_;col++)
    result.data_[col] = data_[idx(i,col)];
  return result;
}

template <typename T>
Matrix<T> Matrix<T>::col(size_t j) const{
  if(j >= cols_)
    throw std::out_of_range("col index out of range");
  Matrix<T> result(rows_,1);
  for(size_t row=0;row<rows_;row++)
    result.data_[row] = data_[idx(row,j)];
  return result;
}

// linear algebra
template <typename T>
T Matrix<T>::trace() const{
  if(!is_square())
    throw std::invalid_argument("matrix is not a square");
  T result = T{};
  for(size_t i=0 ; i < rows_; i++)
    result += data_[idx(i,i)];
  return result;
}

template <typename T>
T Matrix<T>::det() const{
  if(!is_square())
    throw std::invalid_argument("matrix is not a square");
  
  Matrix<T> a(*this);

  int sign = 1;
 
  for(size_t col = 0; col < cols_-1; col++){
    // find max in col
    T cur_max = std::abs(a(col,col));
    size_t idx_max = col;
    for(size_t row = col; row < rows_; row++){
      if(cur_max < std::abs(a.data_[idx(row,col)])){
        cur_max =std::abs(a(row,col));
        idx_max =row; 
      }
    }
    if (std::abs(cur_max) < 1e-9)
    return T{};   // return zero

    // posible swap
    if(idx_max != col){
      sign *= -1;
      for(size_t colS =0; colS<cols_; colS++ ){
        T tmp = a(col,colS);
        a(col,colS) = a(idx_max,colS);
        a(idx_max,colS) = tmp;
      }
    }
    // piviot
    for(size_t row = col+1; row<rows_; row++){
      T multiplier = a(row,col)/a(col,col);
      for(size_t colP=col; colP< cols_; colP++){
        a(row,colP) -= multiplier*a(col,colP);
      }
    } 
  }
  
  T result = T{1};
  for(size_t i = 0; i < rows_ ; i++)
    result *= a.data_[idx(i,i)];
  return result*sign;
}

template <typename T>
T Matrix<T>::norm() const{
  T result = T{};
  for(auto i : data_)
    result += i*i;
  return std::sqrt(result);
}

template <typename T>
Matrix<T> Matrix<T>::inverse() const{
  if(!is_square())
    throw std::invalid_argument("matrix is not a square");
  if (std::abs(det()) < 1e-9)
    throw std::runtime_error("matrix is singular, inverse does not exist"); 
  
  Matrix<T> a(*this);
  Matrix<T> aug = Matrix<T>::identity(rows_);
 
  for(size_t col = 0; col < cols_; col++){
    // find max in col
    T cur_max = std::abs(a(col,col));
    size_t idx_max = col;
    for(size_t row = col; row < rows_; row++){
      if(cur_max < std::abs(a.data_[idx(row,col)])){
        cur_max =std::abs(a(row,col));
        idx_max =row; 
      }
    }

    // posible swap
    if(idx_max != col){
      for(size_t colS =0; colS<cols_; colS++ ){
        T tmp = a(col,colS);
        a(col,colS) = a(idx_max,colS);
        a(idx_max,colS) = tmp;
        
        T tmpA = aug(col,colS);
        aug(col,colS) = aug(idx_max,colS);
        aug(idx_max,colS) = tmpA;
      }
    }
    // piviot
    for(size_t row = col+1; row<rows_; row++){
      T multiplier = a(row,col)/a(col,col);
      for(size_t colP=col; colP< cols_; colP++){
        a(row,colP) -= multiplier*a(col,colP);
        aug(row,colP) -= multiplier*aug(col,colP);
      }
    } 
  }
  
  for(size_t col = cols_; col-- > 0;){

    // piviot
    for(size_t row = 0; row < col; row++){
      T multiplier = a(row,col)/a(col,col);
      for(size_t colP=0; colP< cols_; colP++){
        a(row,colP) -= multiplier*a(col,colP);
        aug(row,colP) -= multiplier*aug(col,colP);
      }
    } 
  }
  
  for(size_t i =0; i < rows_; i++){
    for(size_t j =0; j < cols_; j++){
      aug(i,j) /= a(i,i);
    }
  }
  
  return aug; 
}

template <typename T>
Matrix<T> Matrix<T>::solve(const Matrix<T>& b) const {
    if (cols_ != b.rows())
        throw std::invalid_argument("incompatible dimensions for solve");
    if (std::abs(det()) < 1e-9)
        throw std::runtime_error("matrix is singular, system has no unique solution");
    return inverse() * b;
}

template <typename T>
Matrix<T> Matrix<T>::pow(size_t exp) const {
    if (!is_square())
        throw std::invalid_argument("matrix must be square for pow");
    Matrix<T> result = Matrix<T>::identity(rows_);  // start with identity
    Matrix<T> base(*this);
    for (size_t i = 0; i < exp; i++)
        result = result * base;
    return result;
}

// output
template <typename T>
void Matrix<T>::print() const {
    for (size_t i = 0; i < rows_; i++) {
        for (size_t j = 0; j < cols_; j++)
            std::cout << std::setw(10) << data_[idx(i, j)];
        std::cout << "\n";
    }
}

template <typename T>
void Matrix<T>::print_shape() const {
    std::cout << "Matrix(" << rows_ << "x" << cols_ << ")\n";
}

template <typename T>
Matrix<T> operator*(T scalar, const Matrix<T>& mat) {
    return mat * scalar;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat) {
    mat.print();
    return os;
}

































