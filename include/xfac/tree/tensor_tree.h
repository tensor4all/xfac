#ifndef TENSOR_TREE_H
#define TENSOR_TREE_H


#include"xfac/tree/tree.h"
#include"xfac/cubemat_helper.h"
#include "xfac/matrix/mat_decomp.h"

#include<vector>
#include<array>
#define ARMA_DONT_USE_OPENMP
#include<armadillo>

namespace xfac {

using std::vector;
using std::function;
using std::array;

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

    /// compute the overlap with another tensor tree <this|tt>
    T overlap(const TensorTree<T>& tt) const
    {
        if (M.empty() || tt.M.empty()) return 0;
        if (tree != tt.tree) throw std::invalid_argument("tt1.overlap(tt2) with different tree");
        auto prod = vector<arma::Mat<T>> (M.size());
        for(auto p:tree.leaves()) {// initialize the leaves: L(A,B) = tt.M(A,a,s) * M(B,a,s)
            int pos=tree.neigh.at(p).to_int().begin()->second;
            auto Mc=cube_swap_indices(M[p],0,pos);
            auto Nc=cube_swap_indices(tt.M[p],0,pos);
            prod[p]=cube_as_matrix1(Nc) * cube_as_matrix1(Mc).t();
        }

        auto neigh=tree.neigh;
        for(auto [from,to]:tree.leavesToRoot()) {
            int pos=neigh.at(to).pos(from);
            if (tree.nodes.contains(to)) { // physical node: like mps
                auto Mc=cube_swap_indices(M[to],1,pos);
                auto Nc=cube_swap_indices(tt.M[to],1,pos);
                // L(A,B)=L(a,b)*Nc(A,a,s)*Mc(B,b,s)
                arma::Cube<T> LN=mat_cube(prod[from].st().eval(),Nc,1);
                prod[to]=cube_as_matrix1(LN) * cube_as_matrix1(Mc).t();
            }
            else if (neigh[to].size()==3){ // virtual node: 1st visit
                // L(A,s,B,S)=L(a,b)*N(A,a,s)*M(B,b,S)
                arma::Cube<T> LN=mat_cube(prod[from].st().eval(),tt.M[to], pos);  //LN(A,b,s)
                prod[to]=cube_cube(LN,arma::conj(M[to]),pos);
            }
            else if (neigh[to].size()==2){ // vitual node: 2nd visit
                if (pos==0) {
                    // L(s,S)=Lf(a,b)*Lt(a,s,b,S)
                    auto nS=prod[to].n_cols/prod[from].n_cols;
                    auto ns=prod[to].n_rows/prod[from].n_rows;
                    arma::Cube<T> Lt(prod[to].memptr(), prod[to].n_rows, prod[from].n_cols, nS); // Lt(as,b,S)
                    Lt=cube_swap_indices(Lt,0,1); // Lt(b,as,S)
                    arma::Mat<T> Ltm(Lt.memptr(), Lt.n_rows*prod[from].n_rows, ns*nS); // Lt(ba,sS)
                    arma::Mat<T> Lf=arma::reshape(prod[from].st(),1,prod[from].size());
                    prod[to]=arma::reshape(Lf*Ltm, ns,nS);
                }
                else {
                    // L(s,S)=Lf(a,b)*Lt(sa,Sb)
                    auto nS=prod[to].n_cols/prod[from].n_cols;
                    auto ns=prod[to].n_rows/prod[from].n_rows;
                    arma::Cube<T> Lt(prod[to].memptr(), prod[to].n_rows,  nS, prod[from].n_cols); // Lt(sa,S,b)
                    Lt=cube_swap_indices(Lt,0,1); // Lt(S,sa,b)
                    arma::Mat<T> Ltm(Lt.memptr(), ns*nS, prod[from].size()); // Lt(Ss,ab)
                    arma::Mat<T> Lf=arma::reshape(prod[from],1,prod[from].size());
                    prod[to]=arma::reshape(Lf*Ltm.st(), ns,nS).st();
                }
            }
            else if (neigh[to].size()==1) { // virtual node: root
                // L=L(a,b)*L(a,b)
                T value=arma::dot(prod[from],prod[to]);
                prod[to]=arma::Mat<T>(1,1, arma::fill::value(value));
            }
            else throw std::runtime_error("tree::overlap unexpected number of neighbors");

            neigh[to].remove(from);
            prod[from].clear();
        }
        return prod[tree.root](0,0);
    }

    T norm2() const { return overlap(*this); }

    /// compress the bond between site *from* and site *to*
    void compress_bond(int from, int to, function<array<arma::Mat<T>,2>(arma::Mat<T>,bool)> mat_decomp)
    {
        auto ab = mat_decomp(cubeToMat_R(M.at(from), tree.neigh.at(from).pos(to)), true);
        arma::Mat<T> &M1=ab[0];
        arma::Mat<T> M2= ab[1] * cubeToMat_L(M.at(to), tree.neigh.at(to).pos(from));

        vector<unsigned long int> shape_f{M.at(from).n_rows, M.at(from).n_cols, M.at(from).n_slices};
        shape_f.at(tree.neigh.at(from).pos(to)) = M1.n_cols;
        M.at(from) = matToCube_R(M1, shape_f, tree.neigh.at(from).pos(to));

        vector<unsigned long int> shape_t{M.at(to).n_rows, M.at(to).n_cols, M.at(to).n_slices};
        shape_t.at(tree.neigh.at(to).pos(from)) = M2.n_rows;
        M.at(to) = matToCube_L(M2, shape_t, tree.neigh.at(to).pos(from));
    }

    /// compress along the full tree
    void compress_tree(function<array<arma::Mat<T>,2>(arma::Mat<T>,bool)> mat_decomp){
        for(auto [from, to]:tree.rootToLeaves()) compress_bond(from, to, mat_decomp);
    }

    void compressSVD(double reltol=1e-12, int maxBondDim=0) { compress_tree(MatQR<T> {}); compress_tree(MatSVDFixedTol<T> {reltol,maxBondDim}); }
    void compressLU(double reltol=1e-12, int maxBondDim=0)  { compress_tree(MatRRLUFixedTol<T> {}); compress_tree(MatRRLUFixedTol<T> {reltol, maxBondDim}); }
    void compressCI(double reltol=1e-12, int maxBondDim=0)  { compress_tree(MatCURFixedTol<T> {}); compress_tree(MatCURFixedTol<T> {reltol, maxBondDim}); }

    void save(std::ostream &out) const
    {
        tree.save(out);
        for (const arma::Cube<T>& Mi:M) { Mi.save(out,arma::arma_ascii); out<<std::endl; }
    }
    void save(std::string fileName) const { std::ofstream out(fileName); save(out); }

    static TensorTree<T> load(std::ifstream& in)
    {
        TopologyTree tree=TopologyTree::load(in);
        TensorTree<T> tt(tree);
        for(arma::Cube<T>& Mi:tt.M)
            Mi.load(in,arma::arma_ascii);
        return tt;
    }

    static TensorTree<T> load(std::string fileName)
    {
        std::ifstream in(fileName);
        if (in.fail()) throw std::runtime_error("TensorTrain::load fails to load file: " + fileName);
        return load(in);
    }
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
        auto LM = cubeToMat_R(M, tree.neigh.at(from).pos(to));
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
