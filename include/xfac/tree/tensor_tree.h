#ifndef TENSOR_TREE_H
#define TENSOR_TREE_H


#include"xfac/tree/tree.h"

#include<vector>
#include<array>
#define ARMA_DONT_USE_OPENMP
#include<armadillo>

namespace xfac {

using std::vector;
using std::function;
using std::array;

template<class T>
arma::Cube<T> cube_eval(arma::Cube<T> const& M, int slice_index)
{
    return arma::reshape(M.slice(slice_index), M.n_rows, M.n_slices, 1); //dummy index 1
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


/// stores a tensor train, i.e., a list of cubes.
template<class T>
struct TensorTree {
    TopologyTree tree;       ///< a tree that stores which nodes have physical leg
    vector< arma::Cube<T> >  M;   ///< list of 3-leg tensors

    TensorTree()=default;
    TensorTree(TopologyTree const& tree_) : tree(tree_), M(tree_.size()) {}

    /// evaluate the tensor train at a given multi index.
    T eval(vector<int> const& id) const
    {
        if (id.size()!=M.size()) throw std::invalid_argument("TensorTrain::() id.size()!=size()");
        auto prod=M; // a copy
        for(auto k:tree.nodes)
            prod[k]=cube_eval(M[k],id[k]);
        for(auto [from,to]:tree.leavesToRoot()) {
            arma::Col<T> v=arma::vectorise(prod[from]);
            int pos=tree.neigh.at(to).pos(from);
            prod[to]=cube_vec(prod[to],v,pos);
        }
        return arma::vectorise(prod[0])(0);
    }

    /// evaluate the tensor train at a given multi index. Same as eval()
    T operator()(vector<int> const& id) const { return eval(id); }
};


}// end namespace xfac

#endif // TENSOR_TREE_H
