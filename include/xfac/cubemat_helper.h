#ifndef CUBEMAT_HELPER_H
#define CUBEMAT_HELPER_H

#define ARMA_DONT_USE_OPENMP
#include <armadillo>
#include <algorithm>

namespace xfac {

using std::vector;


/// reshape a cube as a matrix B(i,jk)=A(i,j,k)
template<class eT>
arma::Mat<eT> cube_as_matrix1(arma::Cube<eT> const& A) { return arma::Mat<eT>(const_cast<eT*>(A.memptr()), A.n_rows, A.n_cols*A.n_slices, false); }

/// reshape a cube as a matrix B(ij,k)=A(i,j,k)
template<class eT>
arma::Mat<eT> cube_as_matrix2(arma::Cube<eT> const& A) { return arma::Mat<eT>(const_cast<eT*>(A.memptr()), A.n_rows*A.n_cols, A.n_slices, false); }


/// For a cube C(I, J, L), where I,J,L are the set of indices (0,1..), return the cube C(I, J, L(slice_index))
/// such that the last dimension has only one dummy index
template<class T>
arma::Cube<T> cube_eval(arma::Cube<T> const& M, int slice_index)
{
    arma::Mat<T> data=M.slice(slice_index);
    return arma::Cube<T>(data.memptr(), M.n_rows, M.n_cols, 1, true); //dummy index 1
}



template<class T>
arma::Mat<T> cubeToMat_R(arma::Cube<T> const& A, int cube_pos)
{
    // the packed index is in column-major order
    if (cube_pos==0) { // reshape a cube as a matrix B(jk,i)=A(i,j,k)
        arma::Mat<T> c(A.n_cols * A.n_slices, A.n_rows);
        for(auto i=0u; i<A.n_rows; i++){
            arma::Mat<T> ajk = A.row_as_mat(i).st();
            c.col(i) = arma::Col<T>(ajk.memptr(), A.n_cols * A.n_slices);
        }
        return c;
    } else if (cube_pos==1) { // reshape a cube as a matrix B(ik,j)=A(i,j,k)
        arma::Mat<T> c(A.n_rows * A.n_slices, A.n_cols);
        for(auto j=0u; j<A.n_cols; j++){
            arma::Mat<T> aik = A.col_as_mat(j);
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
arma::Mat<T> cubeToMat_L(arma::Cube<T> const& A, int cube_pos)
{
    // the packed index is in column-major order
    if (cube_pos==0) { // reshape a cube as a matrix B(i,jk)=A(i,j,k)
        arma::Mat<T> c(A.n_rows, A.n_cols * A.n_slices);
        for(auto i=0u; i<A.n_rows; i++){
            arma::Mat<T> ajk = A.row_as_mat(i).st();
            c.row(i) = arma::Row<T>(ajk.memptr(), A.n_cols * A.n_slices);
        }
        return c;
    } else if (cube_pos==1) { // reshape a cube as a matrix B(j,ik)=A(i,j,k)
        arma::Mat<T> c(A.n_cols, A.n_rows * A.n_slices);
        for(auto j=0u; j<A.n_cols; j++){
            arma::Mat<T> aik = A.col_as_mat(j);
            c.row(j) = arma::Row<T>(aik.memptr(), A.n_rows * A.n_slices);
        }
        return c;
    } else if (cube_pos==2) { // reshape a cube as a matrix B(k,ij)=A(i,j,k)
        arma::Mat<T> c(A.n_slices, A.n_rows * A.n_cols);
        for(auto k=0u; k<A.n_slices; k++){
            arma::Mat<T> aij = A.slice(k);
            c.row(k) = arma::Row<T>(aij.memptr(), A.n_rows * A.n_cols);
        }
        return c;
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
            arma::Mat<T> ajk = a.row_as_mat(i).st();
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
            arma::Mat<T> ajk = a.row_as_mat(i).st();
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
        if (a.n_rows!=b.n_cols) throw std::invalid_argument("a.n_rows!=b.n_cols for cube_pos==0");
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
        if (a.n_slices!=b.n_cols) throw std::invalid_argument("a.n_slices!=b.n_cols for cube_pos==2");
        arma::Cube<T> c(a.n_rows, a.n_cols, b.n_rows, arma::fill::zeros);
        for(auto i=0u; i<a.n_rows; i++){
            arma::Mat<T> ajk = a.row_as_mat(i).st();
            c.row(i) = ajk * b.st();
        }
        return c;
    } else {
        throw std::invalid_argument("cube_pos must be 0, 1 or 2");
    }
}

template<class T>
arma::Cube<T> cube_reorder(arma::Cube<T> const& A, std::string_view order)
{
    using std::array;
    array<int,3> idx;
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](size_t i1, size_t i2) {return order[i1] < order[i2];});
    if (idx==array {0,1,2}) return A;
    auto reorder=[&idx](std::array<size_t,3> const& a){
        std::decay<decltype(a)>::type b;
        for(int i=0; i<a.size(); i++) b[idx[i]]=a[i];
        return b;
    };
    auto B_shape=reorder({A.n_rows,A.n_cols,A.n_slices});
    auto B=std::make_from_tuple<arma::Cube<T>>( B_shape );

    for(auto i=0u; i<A.n_rows; i++)
        for(auto j=0u; j<A.n_cols; j++)
            for(auto k=0u; k<A.n_slices; k++) {
                auto B_index=reorder({i,j,k});
                B(B_index[0],B_index[1],B_index[2])=A(i,j,k);
            }
    return B;
}

template<class T>
arma::Cube<T> cube_swap_indices(arma::Cube<T> const& A, int from, int to)
{
    if (from==to) return A;
    std::string order="ijk";
    std::swap(order.at(from),order.at(to));
    return cube_reorder(A,order);
}

/// contract the two cubes along the indices pos, something like L(As,BS)=N(A,a,s)*M(B,a,S)
template<class T>
arma::Mat<T> cube_cube(arma::Cube<T> const& A, arma::Cube<T> const& B, int pos)
{
    if (pos==0) return cube_as_matrix1(A).st()*cube_as_matrix1(B);
    else if (pos==2) return cube_as_matrix2(A)*cube_as_matrix2(B).st();
    else return cube_cube(cube_swap_indices(A,0,1),cube_swap_indices(B,0,1),0);
}


template<class T>
arma::Cube<T> matToCube_R(arma::Mat<T> const& A, vector<unsigned long int> const shape, int cube_pos)
{
    // the packed index is in column-major order
    if (cube_pos==0) { // reshape a matrix as a cube B(jk,i)->A(i,j,k)
        auto c = arma::Cube<T>(A.memptr(), shape.at(1), shape.at(2), shape.at(0));
        return cube_reorder<T>(c, "kij");
    } else if (cube_pos==1) { // reshape a matrix as a cube B(ik,j)->A(i,j,k)
        auto c = arma::Cube<T>(A.memptr(), shape.at(0), shape.at(2), shape.at(1));
        return cube_reorder<T>(c, "ikj");
    } else if (cube_pos==2) { // reshape a cube as a matrix B(ij,k)->A(i,j,k)
        return arma::Cube<T>(A.memptr(), shape.at(0), shape.at(1), shape.at(2));
    } else {
        throw std::invalid_argument("cube_pos must be 0, 1 or 2");
    }
}

template<class T>
arma::Cube<T> matToCube_L(arma::Mat<T> const& A, vector<unsigned long int> const shape, int cube_pos)
{
    // the packed index is in column-major order
    if (cube_pos==0) { // reshape a matrix as a cube B(i,jk)->A(i,j,k)
        return arma::Cube<T>(A.memptr(), shape.at(0), shape.at(1), shape.at(2));
    } else if (cube_pos==1) { // reshape a matrix as a cube B(j,ik)->A(i,j,k)
        auto c = arma::Cube<T>(A.memptr(), shape.at(1), shape.at(0), shape.at(2));
        return cube_reorder<T>(c, "jik");
    } else if (cube_pos==2) { // reshape a cube as a matrix B(k,ij)->A(i,j,k)
        auto c = arma::Cube<T>(A.memptr(), shape.at(2), shape.at(0), shape.at(1));
        return cube_reorder<T>(c, "jki");
    } else {
        throw std::invalid_argument("cube_pos must be 0, 1 or 2");
    }
}

}

#endif // CUBEMAT_HELPER_H
