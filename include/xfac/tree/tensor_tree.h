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

/// contract the cube_pos index of the cube with the vector
template<class T>
arma::Cube<T> cube_vec(arma::Cube<T> const& M, arma::Col<T> v, int cube_pos)
{
    if (cube_pos==2) {
        arma::Mat<T> N=arma::reshape(M, M.n_rows*M.n_cols, M.n_slices, false);
        arma::Col<T> y=N*v;
        return arma::reshape(y, M.n_rows, M.n_cols, 1);
    }
    else if (cube_pos==0)
    {
        arma::Mat<T> N=arma::reshape(M, M.n_rows, M.n_cols*M.n_slices, false);//dummy index 1
        arma::Col<T> y=v.as_row()*N;
        return arma::reshape(y, 1, M.n_cols, M.n_slices);
    }
    else // cube_pos==1
    {
        if (M.n_cols!=v.size()) throw std::invalid_argument("M.n_cols!=v.size() for cube_pos==1");
        arma::Mat<T> N(M.n_rows,M.n_slices, arma::fill::zeros);
        for(auto j=0u; j<M.n_cols; j++)
            N +=M.col(j)*v.at(j);
        return arma::reshape(N, M.n_rows, 1, M.n_slices);
    }
    return arma::reshape(M.slice(cube_pos), M.n_rows, M.n_slices, 1);
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
