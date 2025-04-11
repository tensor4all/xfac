#ifndef TENSOR_TREE_H
#define TENSOR_TREE_H


#include"xfac/tree/tree.h"
#include"xfac/cubemat_helper.h"

#include<vector>
#define ARMA_DONT_USE_OPENMP
#include<armadillo>

namespace xfac {

using std::vector;

/// A class to store a tensor tree and to evaluate it.
template<class T>
struct TensorTree {
    TopologyTree tree;       ///< a tree that stores which nodes have physical leg
    vector< arma::Cube<T> >  M;   ///< list of 3-leg tensors

    TensorTree()=default;
    TensorTree(TopologyTree const& tree_) : tree(tree_), M(tree_.size()) {}

    /// evaluate the tensor tree at a given multi index.
    T eval(vector<int> const& id) const
    {
        if (id.size()!=tree.nodes.size()) throw std::invalid_argument("TensorTree::() id.size()!=tree.nodes.size()");
        auto prod=M; // a copy
        for(auto k:tree.nodes)
            prod[k]=cube_eval(M[k],id[k]);
        for(auto [from,to]:tree.leavesToRoot()) {
            arma::Col<T> v=arma::vectorise(prod[from]);
            int pos=tree.neigh.at(to).pos(from);
            prod[to]=cube_vec(prod[to],v,pos);
        }
        return arma::conv_to<arma::Col<T>>::from(arma::vectorise(prod[tree.root]))(0);
    }

    /// evaluate the tensor tree at a given multi index. Same as eval()
    T operator()(vector<int> const& id) const { return eval(id); }

    /// compute the sum of the tensor tree
    T sum() const;

};

/// A class to sum over the tensor tree.
template<class T>
class TTree_sum {
public:

    TopologyTree tree;
    int neigh, root;
    arma::Row<T> L;  ///< left product at the root site
    std::vector<arma::Cube<T>> R;  ///< right product from the leaf sites

    TTree_sum(){}
    TTree_sum(TensorTree<T> const& tt)
        : tree{tt.tree}
        , R{tt.M}
    {
        auto path = tree.leavesToRoot();
        std::tie(neigh, root) = path.back();
        path.pop_back();
        L = sumRoot(root, neigh, tt.M[root]);
        for(auto node: tree.nodes)
            R[node] = sumPhysicalLeg(node, tt.M[node]);
        for(auto [from, to] : path)
            R[to] = sumLeaveToRoot(from, to);

    }

    /// sum up the the tensors from the root node at *from* to a neighbouring node *to*
    arma::Row<T> sumRoot(int from, int to, arma::Cube<T> const& M)
    {
        auto LM = cubeToMat(M, tree.neigh.at(from).pos(to));
        auto w = arma::Row<T>(LM.n_rows, arma::fill::ones);
        return w * LM;
    }

    /// sum up the local set of the physical node, it corresponds always to the last tensor leg
    arma::Cube<T> sumPhysicalLeg(int node, arma::Cube<T> const& M)
    {
        auto w = arma::Col<T>(M.n_slices, arma::fill::ones);
        return cube_vec(M, w, 2);
    }

    /// sum up the the tensors from node *from* to node *to* in the direction leave to root.
    arma::Cube<T> sumLeaveToRoot(int from, int to)
    {
        arma::Col<T> v = arma::vectorise(R[from]);
        return cube_vec(R[to], v, tree.neigh.at(to).pos(from));
    }

    T value() const { arma::Col<T> Rv = R.at(neigh); return arma::dot(L, Rv); }

};

template<class T>
T TensorTree<T>::sum() const { return TTree_sum<T>(*this).value(); }

}// end namespace xfac

#endif // TENSOR_TREE_H
