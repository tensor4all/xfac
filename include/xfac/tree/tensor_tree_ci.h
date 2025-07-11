#ifndef TENSOR_TREE_CI_H
#define TENSOR_TREE_CI_H

#include "xfac/index_set.h"
#include "xfac/tree/tensor_tree.h"
#include "xfac/tensor/tensor_function.h"
#include "xfac/cubemat_helper.h"

namespace xfac {

/// Parameters of the TensorCI2 algorithm.
struct TensorCIParam {
    int bondDim=30;                         ///< the max bond dimension of the tensor train
    double reltol=1e-12;                    ///< CI will stop when pivotError < reltol*max(pivotError)
    vector<int> pivot1;                     ///< the first pivot (optional)
    bool fullPiv=false;                     ///< whether to search in the full Pi matrix to find a new pivot
    int nRookIter=3;                        ///< number of rook pivoting iterations, used when fullPiv=false
    vector<vector<double>> weight;          ///< TODO: weight for the tt sum. When not empty, it activates the ENV learning
    function<bool(vector<int>)> cond;       ///< TODO: cond(x)=false when x should not be a pivot
    bool useCachedFunction=true;            ///< whether the internal caching should be used to avoid function call to the same point.

    operator ARRLUParam() const { return {.reltol=reltol, .bondDim=bondDim, .fullPiv=fullPiv, .nRookIter=nRookIter}; }
};

/// This class is responsible for building a cross interpolation (CI) of an input tensor.
/// The main output is the tensor tree tt (an effective separation of variables),
/// which allows many cheap computations, i.e. the integral.
template<class T>
struct TensorTreeCI {
    TopologyTree tree;
    TensorFunction<T> f;                                    ///< the tensor function f:(a1,a2,...,an)->T
    TensorFunction<unsigned int> fCond;                     ///< the condition function f:(a1,a2,...,an)->uint which returns 1 if param.cond=true, else 0
    TensorCIParam param;                                   ///< parameters of the algorithm
    vector<double> pivotError;                              ///< max pivot error for each rank
    vector< IndexSet<MultiIndex> > localSet;    ///< collection of MultiIndex for each site: left, site, and right set of multiindex
    std::map<std::pair<int,int>, IndexSet<MultiIndex>> Iset;
    TensorTree<T> tt;                                      ///< main output of the Tensor CI
    std::map<std::pair<int,int>, arma::Mat<T>> P;      ///< the pivot matrix in LU form for each bond
    int cIter=0;                                            ///< counter of iterations. Used for sweeping only.


    /// constructs a rank-1 TensorCI2 from a function f:(a1,a2,...,an)->eT  where the index ai is in [0,localDim[i]).
    TensorTreeCI(function<T(vector<int>)> const& f_, TopologyTree tree_, vector<int> localDim, TensorCIParam param_={})
        : tree(tree_)
        , f {f_, param_.useCachedFunction, false}
        , param(param_)
        , pivotError(1)
        , localSet {localDim.size()}
        , tt {tree}
    {
        if (localDim.size() != tree.nodes.size()) throw std::invalid_argument("tree and localDim are incompatible");
        if (param.pivot1.empty())
            param.pivot1.resize(localDim.size(), 0);
        T fpiv = f.f(param.pivot1);
        pivotError[0]=std::abs(fpiv);
        if (pivotError[0]==0.0)
            throw std::invalid_argument("Not expecting f(pivot1)=0. Provide a better first pivot in the param");

        // fill localSet
        for(auto p=0u; p<len(); p++){
            for(auto i=0; i<localDim[p]; i++){
                MultiIndex lset(len(), 0);
                lset[p] = i;
                localSet[p].push_back(lset);
            }
        }
        if (param.cond){
            if (!param.cond(param.pivot1))
                throw std::invalid_argument("First pivot does not fulfill the condition function, expecting cond(f(pivot1))=true.");
            fCond = TensorFunction<unsigned int>{param.cond, false, false};
        }
        addPivotsAllBonds(enrich_pivot1());
    }

    /// Return the 3-leg T-tensor T_l, where l is the node index.
    /// Convention:
    /// For a physical node: [T_l]_{i j \sigma} \equiv F_{i \oplus j \oplus (\sigma)}
    /// for i \elem \mathcal{Iset}_0 and j \elem \mathcal{Iset}_1, where the subscript
    /// k on \mathcal{Iset}_k labels the neighbours of a given node in consecutive order.
    /// For an artificial node: [T_l]_{i j m} \equiv F_{i \oplus j \oplus m} with
    /// i \elem \mathcal{Iset}_0, j \elem \mathcal{Iset}_1 and m \elem \mathcal{Iset}_2.
    arma::Cube<T> get_T3(int node) const {

        vector<MultiIndex> Ip;
        vector<int> shape;
        for (auto neighbour : tree.neigh.at(node).from_int()) {
            Ip = add(Ip, Iset.at({neighbour, node}));
            shape.push_back(Iset.at({neighbour, node}).size());
        }
        if (shape.size()==1) shape.push_back(1);// leaf

        vector<MultiIndex> Jp;
        if (tree.nodes.contains(node)) {
            Jp = add(Jp,localSet.at(node));
            shape.push_back(localSet.at(node).size());
        } else {
            Jp.push_back(MultiIndex(tree.nodes.size(), 0));
        }

        if(shape.size()!=3) throw std::runtime_error("tensor degree not 3 at get_T3");

        arma::Mat<T> data = f.eval(Ip, Jp);
        return arma::Cube<T>(data.memptr(), shape[0], shape[1], shape[2], true);
    }


    // build the TP^{-1} tensor at node *from*
    arma::Cube<T> get_TP1(int from, int to) const
    {
        auto M = get_T3(from);
        arma::Mat<T> t3mat = cubeToMat_R(M, tree.neigh.at(from).pos(to));
        arma::Mat<T> TP1 = compute_CU_on_rows(t3mat, P.at({from, to}));
        vector<long unsigned int> shape_f{M.n_rows, M.n_cols, M.n_slices};
        shape_f[tree.neigh.at(from).pos(to)] = P.at({from, to}).n_cols;
        return matToCube_R<T>(TP1, shape_f, tree.neigh.at(from).pos(to));
    }

    /// returns the number of physical legs of the tensor tree
    size_t len() const { return localSet.size(); }

    /// makes nIter half sweeps. The dmrg_type can be 0,1,2
    void iterate(int nIter=1, int dmrg_type=2)
    {
        for(auto i=0; i<nIter; i++) {
            for(auto [from,to]:tree.rootToLeaves()) { updatePivotAt(from, to, dmrg_type); }
            cIter++;
        }
    }

    void dmrg0_updatePivotAt(int from, int to)
    {
        auto ci = (param.cond) ? CURDecomp<T> { f.eval(Iset[{from, to}], Iset[{to, from}]), fCond.eval(Iset[{from, to}], Iset[{to, from}]), true, param.reltol, param.bondDim }
                               : CURDecomp<T> { f.eval(Iset[{from, to}], Iset[{to, from}]), true, param.reltol, param.bondDim };

        Iset[{from, to}] = Iset[{from, to}].at(ci.row_pivots());
        Iset[{to, from}] = Iset[{to, from}].at(ci.col_pivots());
        P[{from, to}] = ci.PivotMatrixTri();
        tt.M.at(from) = get_TP1(from,to);
        tt.M.at(to)=get_T3(to);
    }

    /// update the pivots on bond from node *from* to node *to* using the Pi matrix.
    void dmrg2_updatePivotAt(int from, int to)
    {
        IndexSet<MultiIndex> I = set_union(I0[{from, to}], kronecker(from, to));
        IndexSet<MultiIndex> J = set_union(I0[{to, from}], kronecker(to, from));

        auto ci = (param.cond) ? CURDecomp<T> { f.matfun(I, J), fCond.matfun(I, J), I.pos(Iset[{from, to}]), J.pos(Iset[{to, from}]), true, param }
                               : CURDecomp<T> { f.matfun(I, J), I.pos(Iset[{from, to}]), J.pos(Iset[{to, from}]), true, param };

        Iset[{from, to}]=I.at(ci.row_pivots());
        Iset[{to, from}]=J.at(ci.col_pivots());
        P[{from,to}]=ci.PivotMatrixTri();
        tt.M.at(from) = get_TP1(from,to);
        tt.M.at(to)=get_T3(to);
        collectPivotError(from, to, ci.pivotErrors());
    }

    /// add global pivots. The tt is not super stable anymore. For that call makeCanonical() afterward.
    void addPivotsAllBonds(vector<vector<int>> const& pivots)
    {
        for (auto [from,to]:tree.leavesToRoot()) addPivotsAt(pivots, from, to);
        iterate(1,0);
    }

    /// add these pivots at a given bond b. The tt is not touched.
    void addPivotsAt(vector<vector<int>> const& pivots, int from, int to)
    {
        auto [nodes0, nodes1]=tree.split(from, to);
        MultiIndex pvec0(tree.nodes.size(), 0);
        MultiIndex pvec1(tree.nodes.size(), 0);
        for (const auto& pivot : pivots) {
            for (auto node: nodes0) pvec0[node] = pivot[node];
            for (auto node: nodes1) pvec1[node] = pivot[node];
            Iset[{from,to}].push_back(pvec0);
            Iset[{to,from}].push_back(pvec1);
        }
    }

protected:

    /// add all pair around virtual indices, to avoid rank-1 problem
    vector<vector<int>> enrich_pivot1() const
    {
        vector<int> nvnodes; // neighbor to a virtual
        for(auto i=0u; i<tree.neigh.size(); i++) {
            if(tree.nodes.contains(i)) continue;
            for(auto j:tree.neigh.at(i).from_int())
                if (tree.nodes.contains(j)) nvnodes.push_back(j);
        }
        vector<vector<int>> pivots={param.pivot1};
        for(auto i:nvnodes)
            for(auto j:nvnodes)
                if (i!=j)
                    for(const MultiIndex& xi:localSet[i].from_int())
                        for(const MultiIndex& xj:localSet[j].from_int()) {
                            MultiIndex mi {param.pivot1.begin(), param.pivot1.end()};
                            add_inplace(mi,xi);
                            add_inplace(mi,xj);
                            pivots.push_back({mi.begin(),mi.end()});
                        }
        return pivots;
    }

    /// update the pivots at bond b, the dmrg can be 0,1,2.
    void updatePivotAt(int from, int to, int dmrg=2)
    {
        for(auto const& id:Iset[{from, to}].from_int()) I0[{from, to}].push_back(id); // update pivot history
        for(auto const& id:Iset[{to, from}].from_int()) I0[{to, from}].push_back(id);

        switch (dmrg) {
        case 0: dmrg0_updatePivotAt(from,to); break;
        //case 1: dmrg1_updatePivotAt(from, to); break;
        case 2: dmrg2_updatePivotAt(from,to); break;
        }
    }

    /// return $\mathcal{I}_{from}$
    IndexSet<MultiIndex> kronecker(int from, int to) const
    /*
     *  $\mathcal{I}_{from} = (\sum_{i \in N} \mathcal{Iset}_{i, from}) \oplus \mathcal{localSet}_{from}$,
     *  where $N$ are the direct neighbours of the site *from*, except one specific neighbours which we label as *to*.
     *  The sum has to be understood in a $\oplus$ sense.
     */
    {
        IndexSet<MultiIndex> Is;
        for (auto neighbour : tree.neigh.at(from).from_int())
            if (neighbour != to) Is = add(Is, Iset.at({neighbour, from}));
        if (tree.nodes.contains(from))
            return add(Is, localSet.at(from));
        return Is;
    }

    void collectPivotError(int from, int to, vector<double> const& pe)
    {
        pivotErrorAll[{from,to}]=pe;
        if (pe.size()>pivotError.size()) pivotError.resize(pe.size(), 0);
        for(auto i=0u; i<pe.size(); i++)
            if (pe[i]>pivotError[i])
                pivotError[i]=pe[i];
    }

private:
    std::map<std::pair<int,int>, vector<double>> pivotErrorAll;           ///< The pivot error list for each bonds
    std::map<std::pair<int,int>, IndexSet<MultiIndex> > I0;               ///< Historical lists of accepted pivots for each site

};

} // end namespace xfac

#endif // TENSOR_TREE_CI_H
