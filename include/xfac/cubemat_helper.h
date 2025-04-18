#ifndef CUBEMAT_HELPER_H
#define CUBEMAT_HELPER_H

#define ARMA_DONT_USE_OPENMP
#include<armadillo>

namespace xfac {

using std::vector;

/// For a cube C(I, J, L), where I,J,L are the set of indices (0,1..), return the cube C(I, J, L(slice_index))
/// such that the last dimension has only one dummy index
template<class T>
arma::Cube<T> cube_eval(arma::Cube<T> const& M, int slice_index)
{
    arma::Mat<T> data=M.slice(slice_index);
    return arma::cube(data.memptr(), M.n_rows, M.n_cols, 1, true); //dummy index 1
}

template<class T>
arma::Mat<T> cubeToMat(arma::Cube<T> const& A, int cube_pos)
{
    if (cube_pos==0) { // reshape a cube as a matrix B(jk,i)=A(i,j,k)
        arma::Mat<T> c(A.n_cols * A.n_slices, A.n_rows);
        for(auto i=0u; i<A.n_rows; i++){
            arma::Mat<T> ajk = A.row(i);
            c.col(i) = arma::Col<T>(ajk.memptr(), A.n_cols * A.n_slices);
        }
        return c;
    } else if (cube_pos==1) { // reshape a cube as a matrix B(ik,j)=A(i,j,k)
        arma::Mat<T> c(A.n_rows * A.n_slices, A.n_cols);
        for(auto j=0u; j<A.n_cols; j++){
            arma::Mat<T> aik = A.col(j);
            c.col(j) = arma::Col<T>(aik.memptr(), A.n_rows * A.n_slices);
        }
        return c;
    } else if (cube_pos==2) { // reshape a cube as a matrix B(ij,k)=A(i,j,k)
        return arma::Mat<T>(const_cast<T*>(A.memptr()), A.n_rows*A.n_cols, A.n_slices, false);
    } else {
        throw std::invalid_argument("cube_pos must be 0, 1 or 2");
    }
}

template<class T>
arma::Mat<T> cubeToMat2(arma::Cube<T> const& A, int cube_pos)
{
    if (cube_pos==0) { // reshape a cube as a matrix B(ik,j)=A(i,j,k)
        arma::Mat<T> c(A.n_rows * A.n_slices, A.n_cols);
        for(auto j=0u; j<A.n_cols; j++){
            arma::Mat<T> aik = A.col(j);
            c.col(j) = arma::Col<T>(aik.memptr(), A.n_rows * A.n_slices);
        }
        return c;
    } else if (cube_pos==1) { // reshape a cube as a matrix B(jk,i)=A(i,j,k)
        arma::Mat<T> c(A.n_cols * A.n_slices, A.n_rows);
        for(auto i=0u; i<A.n_rows; i++){
            arma::Mat<T> ajk = A.row(i);
            c.col(i) = arma::Col<T>(ajk.memptr(), A.n_cols * A.n_slices);
        }
        return c;
    } else if (cube_pos==2) { // reshape a cube as a matrix B(ij,k)=A(i,j,k)
        return arma::Mat<T>(const_cast<T*>(A.memptr()), A.n_rows*A.n_cols, A.n_slices, false);
    } else {
        throw std::invalid_argument("cube_pos must be 0, 1 or 2");
    }
}



template<class T>
vector<long long unsigned int> cubeToMatPos(arma::Cube<T> const& A, int cube_pos)
{
    if (cube_pos==0) { // reshape a cube as a matrix B(jk,i)=A(i,j,k)
        return {A.n_cols, A.n_slices, A.n_rows};
    } else if (cube_pos==1) { // reshape a cube as a matrix B(ik,j)=A(i,j,k)
        return {A.n_rows, A.n_slices, A.n_cols};;
    } else if (cube_pos==2) { // reshape a cube as a matrix B(ij,k)=A(i,j,k)
        return {A.n_rows, A.n_cols, A.n_slices};;
    } else {
        throw std::invalid_argument("cube_pos must be 0, 1 or 2");
    }
}

template<class T>
vector<long long unsigned int> cubeToMat2Pos(arma::Cube<T> const& A, int cube_pos)
{
    if (cube_pos==0) { // reshape a cube as a matrix B(ik,j)=A(i,j,k)
        return {A.n_rows, A.n_slices, A.n_cols};
    } else if (cube_pos==1) { // reshape a cube as a matrix B(jk,i)=A(i,j,k)
        return {A.n_cols, A.n_slices, A.n_rows};;
    } else if (cube_pos==2) { // reshape a cube as a matrix B(ij,k)=A(i,j,k)
        return {A.n_rows, A.n_cols, A.n_slices};;
    } else {
        throw std::invalid_argument("cube_pos must be 0, 1 or 2");
    }
}


/// Swap the leg with given cube position with the leg at the end.
template<class T>
arma::Cube<T> reshape_cube(arma::Cube<T> const& a, int cube_pos)
{
    if (cube_pos==0) { // $A_{i, j, k} -> A_{k, j, i}$
        arma::Cube<T> c(a.n_slices, a.n_cols, a.n_rows);
        for(auto i=0u; i<a.n_rows; i++){
            arma::Mat<T> ajk = a.row(i);
            c.slice(i) = ajk.st();
        }
        return c;
    } else if (cube_pos==1) { // $A_{i, j, k} -> A_{i, k, j}$
        arma::Cube<T> c(a.n_rows, a.n_slices, a.n_cols);
        for(auto j=0u; j<a.n_cols; j++){
            arma::Mat<T> aik = a.col(j);
            c.slice(j) = aik;
        }
        return c;
    } else if (cube_pos==2) { // unchanged: $A_{i, j, k} -> A_{i, j, k}$
        return a;
    } else {
        throw std::invalid_argument("cube_pos must be 0, 1 or 2");
    }
}

/// Swap the leg with given cube position with the leg at the end.
template<class T>
arma::Cube<T> reshape_cube2(arma::Cube<T> const& a, int cube_pos)
{
    if (cube_pos==0) { // $A_{i, j, k} -> A_{k, i, j}$
        arma::Cube<T> c(a.n_slices, a.n_rows, a.n_cols);
        for(auto j=0u; j<a.n_cols; j++){
            arma::Mat<T> aik = a.col(j);
            c.slice(j) = aik.st();
        }
        return c;
    } else if (cube_pos==1) { // $A_{i, j, k} -> A_{i, k, j}$
        arma::Cube<T> c(a.n_rows, a.n_slices, a.n_cols);
        for(auto j=0u; j<a.n_cols; j++){
            arma::Mat<T> aik = a.col(j);
            c.slice(j) = aik;
        }
        return c;
    } else if (cube_pos==2) { // unchanged: $A_{i, j, k} -> A_{i, j, k}$
        return a;
    } else {
        throw std::invalid_argument("cube_pos must be 0, 1 or 2");
    }
}


/// Return the cube $C = A B$, $A:$ cube, $B:$ vector, where cube_pos indicates
/// which leg of the cube A to contract with B, see below for an exact definition.
/// In C, the contracted leg has only one element.
template<class T>
arma::Cube<T> cube_vec(arma::Cube<T> const& a, arma::Col<T> const& b, int cube_pos)
{
    if (cube_pos==0) { // $C_{0,i,j} = \sum_k A_{k, i, j} B_{k}$
        if (a.n_rows!=b.n_elem) throw std::invalid_argument("a.n_rows!=b.n_elem for cube_pos==0");
        arma::Cube<T> c(1, a.n_cols, a.n_slices, arma::fill::zeros);
        for(auto j=0u; j<a.n_slices; j++){
            c.slice(j) = arma::conv_to<arma::Mat<T>>::from(b.st() * a.slice(j));
        }
        return c;
    } else if (cube_pos==1) { // $C_{i,0,j} = \sum_k A_{i, k, j} B_{k}$
        if (a.n_cols!=b.n_elem) throw std::invalid_argument("a.n_cols!=b.n_elem for cube_pos==1");
        arma::Cube<T> c(a.n_rows, 1, a.n_slices, arma::fill::zeros);
        for(auto j=0u; j<a.n_slices; j++){
            c.slice(j) = arma::conv_to<arma::Mat<T>>::from(a.slice(j) * b);
        }
        return c;
    } else if (cube_pos==2) { // $C_{i,j,0} = \sum_k A_{i, j, k} B_{k}$
        if (a.n_slices!=b.n_elem) throw std::invalid_argument("a.n_slices!=b.n_elem for cube_pos==2");
        arma::Cube<T> c(a.n_rows, a.n_cols, 1, arma::fill::zeros);
        for(auto i=0u; i<a.n_rows; i++){
            arma::Mat<T> ajk = a.row(i);
            c.row(i) = ajk * b;
        }
        return c;
    } else {
        throw std::invalid_argument("cube_pos must be 0, 1 or 2");
    }
}


/// Return the cube $C = A B$, $A:$ cube, $B:$ matrix, where cube_pos indicates
/// which leg of the cube A to contract with B, see below for an exact definition.
template<class T>
arma::Cube<T> cube_mat(arma::Cube<T> const& a, arma::Mat<T> const& b, int cube_pos)
{
    if (cube_pos==0) { // $C_{i,j,l} = \sum_k A_{k, j, l} B_{k, i}$
        if (a.n_rows!=b.n_rows) throw std::invalid_argument("a.n_rows!=b.n_rows for cube_pos==0");
        arma::Cube<T> c(b.n_cols, a.n_cols, a.n_slices, arma::fill::zeros);
        for(auto l=0u; l<a.n_slices; l++){
            c.slice(l) = arma::conv_to<arma::Mat<T>>::from(b.st() * a.slice(l));
        }
        return c;
    } else if (cube_pos==1) { // $C_{i,j,l} = \sum_k A_{i, k, l} B_{k, j}$
        if (a.n_cols!=b.n_rows) throw std::invalid_argument("a.n_cols!=b.n_rows for cube_pos==1");
        arma::Cube<T> c(a.n_rows, b.n_cols, a.n_slices, arma::fill::zeros);
        for(auto l=0u; l<a.n_slices; l++){
            c.slice(l) = arma::conv_to<arma::Mat<T>>::from(a.slice(l) * b);
        }
        return c;
    } else if (cube_pos==2) { // $C_{i,j,l} = \sum_k A_{i, j, k} B_{k, l}$
        if (a.n_slices!=b.n_rows) throw std::invalid_argument("a.n_slices!=b.n_rows for cube_pos==2");
        arma::Cube<T> c(a.n_rows, a.n_cols, b.n_cols, arma::fill::zeros);
        for(auto i=0u; i<a.n_rows; i++){
            arma::Mat<T> ajk = a.row(i);
            c.row(i) = ajk * b;
        }
        return c;
    } else {
        throw std::invalid_argument("cube_pos must be 0, 1 or 2");
    }
}


/// Return the cube $C = B A$, $A:$ cube, $B:$ matrix, where cube_pos indicates
/// which leg of the cube A to contract with B, see below for an exact definition.
template<class T>
arma::Cube<T> mat_cube(arma::Mat<T> const& b, arma::Cube<T> const& a, int cube_pos)
{
    if (cube_pos==0) { // $C_{i,j,l} = \sum_k B_{i, k} A_{k, j, l}$
        if (a.n_rows!=b.n_cols) throw std::invalid_argument("a.n_rows!=b.n_cols for cube_pos==1");
        arma::Cube<T> c(b.n_rows, a.n_cols, a.n_slices, arma::fill::zeros);
        for(auto l=0u; l<a.n_slices; l++){
            c.slice(l) = arma::conv_to<arma::Mat<T>>::from(b * a.slice(l));
        }
        return c;
    } else if (cube_pos==1) { // $C_{i,j,l} = \sum_k B_{j, k} A_{i, k, l}$
        if (a.n_cols!=b.n_cols) throw std::invalid_argument("a.n_cols!=b.n_cols for cube_pos==1");
        arma::Cube<T> c(a.n_rows, b.n_rows, a.n_slices, arma::fill::zeros);
        for(auto l=0u; l<a.n_slices; l++){
            c.slice(l) = arma::conv_to<arma::Mat<T>>::from(a.slice(l) * b.st());
        }
        return c;
    } else if (cube_pos==2) { // $C_{i,j,l} = \sum_k B_{l, k} A_{i, j, k}$
        if (a.n_slices!=b.n_cols) throw std::invalid_argument("a.n_slices!=b.n_cols for cube_pos==1");
        arma::Cube<T> c(a.n_rows, a.n_cols, b.n_rows, arma::fill::zeros);
        for(auto i=0u; i<a.n_rows; i++){
            arma::Mat<T> ajk = a.row(i);
            c.row(i) = ajk * b.st();
        }
        return c;
    } else {
        throw std::invalid_argument("cube_pos must be 0, 1 or 2");
    }
}

}

#endif // CUBEMAT_HELPER_H
