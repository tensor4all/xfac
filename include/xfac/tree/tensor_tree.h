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
template<class T>
arma::Cube<T> cube_vec(arma::Cube<T> const& M, arma::Col<T> v, int cube_pos)
{
    return arma::reshape(M.slice(cube_pos), M.n_rows, M.n_slices, 1); //dummy index 1
}


/// stores a tensor train, i.e., a list of cubes.
template<class T>
struct TensorTree {
    vector< arma::Cube<T> >  M;   ///< list of 3-leg tensors
    OrderedTree tree;       ///< a tree that stores which nodes have physical leg

    TensorTree()=default;
    TensorTree(size_t len) : M(len) {}  //TODO: initialize the tree!

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

private:
    static function<T(vector<vector<int>>)> tensorFun(function<T(vector<double>)> f, grid::QuanticsTree grid) {
        return [=](vector<vector<int>> const& id) { return f(grid.id_to_coord(id)); };
    }

};


}// end namespace xfac

#endif // TENSOR_TREE_H
