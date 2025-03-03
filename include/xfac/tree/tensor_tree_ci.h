#ifndef TENSOR_TREE_CI_H
#define TENSOR_TREE_CI_H

#include "xfac/index_set.h"
#include "xfac/tree/tensor_tree.h"
#include "xfac/tensor/tensor_function.h"

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
    OrderedTree tree;
    TensorFunction<T> f;                                    ///< the tensor function f:(a1,a2,...,an)->T
    TensorCIParam param;                                   ///< parameters of the algorithm
    vector<double> pivotError;                              ///< max pivot error for each rank
    vector< IndexSet<MultiIndex> > localSet;    ///< collection of MultiIndex for each site: left, site, and right set of multiindex
    std::map<std::pair<int,int>, IndexSet<MultiIndex>> Iset, Jset;
    TensorTree<T> tt;                                      ///< main output of the Tensor CI
    std::map<std::pair<int,int>, < arma::Mat<T> >> P;      ///< the pivot matrix in LU form for each bond
    int cIter=0;                                            ///< counter of iterations. Used for sweeping only.
    int center=0;                                           ///< current position of the CI center


    /// constructs a rank-1 TensorCI2 from a function f:(a1,a2,...,an)->eT  where the index ai is in [0,localDim[i]).
    TensorTreeCI(TensorFunction<T> const& f_, OrderedTree tree_, vector<int> localDim, TensorCIParam param_={})
        : tree(tree_)
        , f {f_}
        , param(param_)
        , pivotError(1)
        , localSet {localDim.size()}
        , tt {localDim.size()}
        , pivotErrorAll {localDim.size()-1}
    {
        if (localDim.size() != tree.nodes.size()) throw std::invalid_argument("tree and localDim are incompatible");
        if (param.pivot1.empty())
            param.pivot1.resize(localDim.size(), 0);
        pivotError[0]=std::abs(f.f(param.pivot1));
        if (pivotError[0]==0.0)
            throw std::invalid_argument("Not expecting f(pivot1)=0. Provide a better first pivot in the param");

        // fill localSet
        for(auto p=0u; p<tree.nodes.size(); p++)
            for(auto i=0; i<localDim[p]; i++)
                localSet[p].push_back({char32_t(i)});

        //fill Iset, Jset
        for (auto [from,to]:tree.leavesToRoot()) {
            auto [nodes0,nodes1]=tree.split(from,to);
            for (auto node: nodes0) Iset[{from,to}].push_back(param.pivot1[node]);
            for (auto node: nodes1) Jset[{from,to}].push_back(param.pivot1[node]);
        }

        //iterate(1,0); // just to define tt
    }

    /// makes nIter half sweeps. The dmrg_type can be 0,1,2
    void iterate(int nIter=1, int dmrg_type=2)
    {
        for(auto i=0; i<nIter; i++) {
            if (cIter%2==0)
                for(auto [from,to]:tree.leavesToRoot()) { center=to; updatePivotAt(from, to, dmrg_type); }
            else
                for(auto [from,to]:tree.rootToLeaves()) { center=to; updatePivotAt(from, to, dmrg_type); }
            cIter++;
        }
    }

protected:

    /// update the pivots at bond b, the dmrg can be 0,1,2.
    void updatePivotAt(int from, int to, int dmrg=2)
    {
        switch (dmrg) {
        case 0: dmrg0_updatePivotAt(from,to); break;
        // case 1: dmrg1_updatePivotAt(b); break;
        case 2: dmrg2_updatePivotAt(from,to); break;
        }
    }

    /// update the pivots at bond b using the Pi matrix.
    void dmrg2_updatePivotAt(int from, int to)
    {
        // Iset[b]   -> Iset[{from,to}]
        // Iset[b+1] -> Iset[{to,from}]
        // Jset[b]   -> Jset[{from,to}]
        // Jset[b+1] -> Jset[{to,from}]
        // localSet[b] -> localSet[from]
        // localSet[b+1] -> localSet[to]

        IndexSet<MultiIndex> Ib=kron(Iset[{from,to}],localSet[from])
        IndexSet<MultiIndex> Jb=kron(localSet[to],Jset[{to,from}])) ;
        auto p1=param;
        //        p1.bondDim=std::min(p1.bondDim, (int)Iset[b+1].size()*2);           // limit the rank increase to duplication only
        auto ci=CURDecomp<T> { f.matfun(Ib,Jb), Ib.pos(Iset[{to, from}]), Jb.pos(Jset[{from, to}]), from<center, p1 };
        Iset[{to, from}]=Ib.at(ci.row_pivots());
        Jset[{from, to}]=Jb.at(ci.col_pivots());
        P[{from,to}]=ci.PivotMatrixTri();
        set_site_tensor(from, to);
        set_site_tensor(to, from);
        collectPivotError(from, to, ci.pivotErrors());
    }

/*
    /// update the pivots at bond b using the P matrix
    void dmrg0_updatePivotAt(int from, int to)
    {
        auto ci=CURDecomp<T> { f.eval(Iset[b+1],Jset[b]), b<center, param.reltol, param.bondDim };
        Iset[b+1]=Iset[b+1].at(ci.row_pivots());
        Jset[b]=Jset[b].at(ci.col_pivots());
        P[b]=ci.PivotMatrixTri();
        set_site_tensor(b);
        set_site_tensor(b+1);
        collectPivotError(b, ci.pivotErrors());
    }
*/
    void set_site_tensor(int from, int to)
    {
        set_site_tensor(from, to, f.eval(kron(Iset[{from,to}],localSet[from]), Jset[{from, to}]));
        if (from<center)
            set_site_tensor(from, to, compute_CU_on_rows(cube_as_matrix2(tt.M[from]), P[{from,to}]));
        else if (from>center)
            set_site_tensor(to, from, compute_UR_on_cols(cube_as_matrix1(tt.M[to]), P.at({to,from})));
    }

    void set_site_tensor(int from, int to, arma::Mat<T> const& M) { tt.M[from]=arma::Cube<T>(M.memptr(), Iset[{from,to}].size(), localSet[from].size(), Jset[{from,to}].size());  }

    void collectPivotError(int from, int to, vector<double> const& pe)
    {
        pivotErrorAll[{from,to}]=pe;
        if (pe.size()>pivotError.size()) pivotError.resize(pe.size(), 0);
        for(auto i=0u; i<pe.size(); i++)
            if (pe[i]>pivotError[i])
                pivotError[i]=pe[i];
    }

private:
    std::map<std::pair<int,int>, <vector<double>> pivotErrorAll;           ///< The pivot error list for each bonds


};

} // end namespace xfac

#endif // TENSOR_TREE_CI_H
