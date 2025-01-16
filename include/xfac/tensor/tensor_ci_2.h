#ifndef TENSOR_CI2_H
#define TENSOR_CI2_H


#include "xfac/index_set.h"
#include "xfac/tensor/tensor_train.h"
#include "xfac/tensor/tensor_function.h"

#include <bitset>

namespace xfac {

/// Parameters of the TensorCI2 algorithm.
struct TensorCI2Param {
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
/// The main output is the tensor train tt (an effective separation of variables),
/// which allows many cheap computations, i.e. the integral.
template<class T>
struct TensorCI2 {
    TensorFunction<T> f;                                    ///< the tensor function f:(a1,a2,...,an)->T
    TensorCI2Param param;                                   ///< parameters of the algorithm
    vector<double> pivotError;                              ///< max pivot error for each rank
    vector< IndexSet<MultiIndex> > Iset, localSet, Jset;    ///< collection of MultiIndex for each site: left, site, and right set of multiindex
    TensorTrain<T> tt;                                      ///< main output of the Tensor CI
    vector< arma::Mat<T> > P;                               ///< the pivot matrix in LU form for each bond
    int cIter=0;                                            ///< counter of iterations. Used for sweeping only.
    int center=0;                                           ///< current position of the CI center


    /// constructs a rank-1 TensorCI2 from a function f:(a1,a2,...,an)->eT  where the index ai is in [0,localDim[i]).
    TensorCI2(TensorFunction<T> const& f_, vector<int> localDim, TensorCI2Param param_={})
        : f {f_}
        , param(param_)
        , pivotError(1)
        , Iset {localDim.size()}
        , localSet {localDim.size()}
        , Jset {localDim.size()}
        , tt {localDim.size()}
        , P {localDim.size()}
        , pivotErrorAll {localDim.size()-1}
        , I0 {localDim.size()}
        , J0 {localDim.size()}
    {
        if (param.pivot1.empty())
            param.pivot1.resize(localDim.size(), 0);
        pivotError[0]=std::abs(f.f(param.pivot1));
        if (pivotError[0]==0.0)
            throw std::invalid_argument("Not expecting f(pivot1)=0. Provide a better first pivot in the param");

        // fill localSet, Iset Jset
        for(auto p=0u; p<len(); p++)
            for(auto i=0; i<localDim[p]; i++)
                localSet[p].push_back({char32_t(i)});

        //add param.pivot1;
        for(auto b=0u; b<len(); b++) {
            Iset[b].push_back({param.pivot1.begin(), param.pivot1.begin()+b});
            Jset[b].push_back({param.pivot1.begin()+b+1, param.pivot1.end()});
        }
        iterate(1,0); // just to define tt
    }

    /// constructs a rank-1 TensorCI2 from a function f:(a1,a2,...,an)->eT  where the index ai is in [0,localDim[i]).
    TensorCI2(function<T(vector<int>)> f_, vector<int> localDim, TensorCI2Param param_={})
        : TensorCI2(TensorFunction<T> {f_, param_.useCachedFunction}, localDim, param_)
    {}

    /// constructs a TensorCI2 from a given tensor train, and using the provided function f.
    TensorCI2(TensorFunction<T> const& f_, TensorTrain<T> tt_, TensorCI2Param param_={})
        : f {f_}
        , param(param_)
        , pivotError(1)
        , Iset {tt_.M.size()}
        , localSet {tt_.M.size()}
        , Jset {tt_.M.size()}
        , tt {tt_.M.size()}
        , P {tt_.M.size()}
        , pivotErrorAll {tt_.M.size()-1}
        , I0 {tt_.M.size()}
        , J0 {tt_.M.size()}
    {
        // fill localSet, Iset Jset
        for(auto p=0u; p<len(); p++)
            for(auto i=0; i<tt_.M[p].n_cols; i++)
                localSet[p].push_back({char32_t(i)});
        Iset[0].push_back({});
        Jset[len()-1].push_back({});
        // generate all pivots
        tt_.compressCI(param.reltol, param.bondDim);
        // left-to-right
        for(auto b=0u; b+1<tt_.M.size(); b++) {
            IndexSet<MultiIndex> Ib= kron(Iset[b],localSet[b]);
            auto ci=CURDecomp<T> {cube_as_matrix2(tt_.M[b]), true, 0, param.bondDim};
            arma::Mat<T> M1=ci.left();
            arma::Mat<T> M2=ci.right()* cube_as_matrix1(tt_.M[b+1]);
            tt_.M[b]=arma::Cube<T>(M1.memptr(), tt_.M[b].n_rows, tt_.M[b].n_cols, M1.n_cols);
            tt_.M[b+1]=arma::Cube<T>(M2.memptr(), M2.n_rows, tt_.M[b+1].n_cols, tt_.M[b+1].n_slices);
            Iset[b+1]= Ib.at(ci.row_pivots());
        }
        // right-to-left
        for(int b=tt_.M.size()-2; b>=0; b--) {
            IndexSet<MultiIndex> Jb= kron(localSet[b+1],Jset[b+1]);
            auto ci=CURDecomp<T> (cube_as_matrix1(tt_.M[b+1]), false, 0, param.bondDim);
            arma::Mat<T> M1=cube_as_matrix2(tt_.M[b])*ci.left();
            arma::Mat<T> M2=ci.right();
            tt_.M[b]=arma::Cube<T>(M1.memptr(), tt_.M[b].n_rows, tt_.M[b].n_cols, M1.n_cols);
            tt_.M[b+1]=arma::Cube<T>(M2.memptr(), M2.n_rows, tt_.M[b+1].n_cols, tt_.M[b+1].n_slices);
            Iset[b+1]= Iset[b+1].at(ci.row_pivots());
            Jset[b]= Jb.at(ci.col_pivots());
        }
        iterate(1,0); // just to define tt, while reevaluating the original f in the pivots.
    }

    /// constructs a TensorCI2 from a given tensor train, and using the provided function f.
    TensorCI2(function<T(vector<int>)> f_, TensorTrain<T> tt_, TensorCI2Param param_={})
        : TensorCI2(TensorFunction<T> {f_, param_.useCachedFunction}, tt_, param_)
    {}

    /// constructs a TensorCI2 from a given tensor train. It takes the tt as a true function f.
    TensorCI2(TensorTrain<T> const& tt_, TensorCI2Param param_={}) : TensorCI2(tt_, tt_, param_) {}

    /// add global pivots. The tt is not super stable anymore. For that call makeCanonical() afterward.
    void addPivotsAllBonds(vector<vector<int>> const& pivots)
    {
        for(auto b=0u; b<len()-1; b++) addPivotsAt(pivots,b);
        iterate(1,0);
    }

    /// add these pivots at a given bond b. The tt is not touched.
    void addPivotsAt(vector<vector<int>> const& pivots, int b)
    {
        for (const auto& pg : pivots) {
            Iset[b+1].push_back({pg.begin(), pg.begin()+b+1});
            Jset[b].push_back({pg.begin()+b+1, pg.end()});
        }
    }

    /// add all the pivots from another ci at the correct partitions. The tt is not super stable anymore. For that call makeCanonical() afterward.
    template<class Tci>
    void addPivots(Tci const& ci)
    {
        cIter=ci.cIter;
        for(auto b=0u; b<ci.len()-1; b++)
            addPivotsAt(ci.getPivotsAt(b), b);
        iterate(1,0);
    }

    /// generate a fully nested tci and well-conditioned tt
    void makeCanonical()
    {
        iterate(2,1); // full dmrg1 sweep

        {// dmrg1 without throwing any pivot
            auto reltol=param.reltol;
            param.reltol=0;
            iterate(1,1);
            param.reltol=reltol;
        }
    }

    /// makes nIter half sweeps. The dmrg_type can be 0,1,2
    void iterate(int nIter=1, int dmrg_type=2)
    {
        for(auto i=0; i<nIter; i++) {
            if (cIter%2==0)
                for(auto b=0u; b<len()-1; b++) { center=b+1; updatePivotAt(b, dmrg_type); }
            else
                for(int b=len()-2; b>=0; b--) { center=b; updatePivotAt(b, dmrg_type); }
            cIter++;
        }
    }

    /// returns the length of the tensor
    size_t len() const { return localSet.size(); }

    /// computes the max error |f-tt| if the tensor is smaller than max_n_Eval.
    double trueError(size_t max_n_eval=1e6) const { return tt.trueError(f.f, max_n_eval); }

    /// returns the pivots a given bond b
    vector<vector<int>> getPivotsAt(int b) const
    {
        vector<vector<int>> pivots;
        for(auto r=0u; r< Iset[b+1].size(); r++) {
            MultiIndex ij=Iset[b+1].at(r)+Jset[b].at(r);
            pivots.push_back({ij.cbegin(),ij.cend()});
        }
        return pivots;
    }

    // return true if each bond is low rank or it is saturated
    bool isDone() const
    {
        double ref=*std::max_element(pivotError.begin(), pivotError.end())*param.reltol;
        for(auto b=0; b<len()-1; b++) {
            int rankMax=std::min(Iset[b].size()*localSet[b].size(),
                                 localSet[b+1].size()*Jset[b+1].size());
            rankMax=std::min(rankMax, param.bondDim);
            if (Iset[b+1].size() < rankMax &&                               // not saturated
                    pivotErrorAll[b].back() > ref)   // not low rank
                return false;
        }
        return true;
    }

protected:
    /// update the pivots at bond b, the dmrg can be 0,1,2.
    void updatePivotAt(int b, int dmrg=2)
    {
        for(auto const& id:Iset[b+1].from_int()) I0[b+1].push_back(id); // update pivot history
        for(auto const& id:Jset[b].from_int()) J0[b].push_back(id);
        switch (dmrg) {
        case 0: dmrg0_updatePivotAt(b); break;
        case 1: dmrg1_updatePivotAt(b); break;
        case 2: dmrg2_updatePivotAt(b); break;
        }
    }

    /// update the pivots at bond b using the Pi matrix.
    void dmrg2_updatePivotAt(int b)
    {        
        IndexSet<MultiIndex> Ib=set_union(I0[b+1], kron(Iset[b],localSet[b]));
        IndexSet<MultiIndex> Jb=set_union(J0[b], kron(localSet[b+1],Jset[b+1])) ;
        auto p1=param;
//        p1.bondDim=std::min(p1.bondDim, (int)Iset[b+1].size()*2);           // limit the rank increase to duplication only
        auto ci=CURDecomp<T> { f.matfun(Ib,Jb), Ib.pos(Iset[b+1]), Jb.pos(Jset[b]), b<center, p1 };
        Iset[b+1]=Ib.at(ci.row_pivots());
        Jset[b]=Jb.at(ci.col_pivots());
        P[b]=ci.PivotMatrixTri();
        set_site_tensor(b);
        set_site_tensor(b+1);
        collectPivotError(b, ci.pivotErrors());
    }

    /// update the pivots at bond b using the T tensor
    void dmrg1_updatePivotAt(int b)
    {
        bool isLeft=b<center;
        IndexSet<MultiIndex> Ib= isLeft ? kron(Iset[b],localSet[b]) : Iset[b+1].from_int() ;
        IndexSet<MultiIndex> Jb= isLeft ? Jset[b].from_int() : kron(localSet[b+1],Jset[b+1]);
        auto ci=CURDecomp<T> { f.eval(Ib,Jb), isLeft, param.reltol, param.bondDim };
        Iset[b+1]= Ib.at(ci.row_pivots());
        Jset[b]= Jb.at(ci.col_pivots());
        P[b]=ci.PivotMatrixTri();
        if (isLeft) {
            set_site_tensor(b, ci.left());
            set_site_tensor(b+1);
        }
        else {
            set_site_tensor(b);
            set_site_tensor(b+1, ci.right());
        }
        collectPivotError(b, ci.pivotErrors());
    }

    /// update the pivots at bond b using the P matrix
    void dmrg0_updatePivotAt(int b)
    {
        auto ci=CURDecomp<T> { f.eval(Iset[b+1],Jset[b]), b<center, param.reltol, param.bondDim };
        Iset[b+1]=Iset[b+1].at(ci.row_pivots());
        Jset[b]=Jset[b].at(ci.col_pivots());
        P[b]=ci.PivotMatrixTri();
        set_site_tensor(b);
        set_site_tensor(b+1);
        collectPivotError(b, ci.pivotErrors());
    }

    void set_site_tensor(int b)
    {
        set_site_tensor(b, f.eval(kron(Iset[b],localSet[b]), Jset[b]));
        if (b<center)
            set_site_tensor(b, compute_CU_on_rows(cube_as_matrix2(tt.M[b]), P[b]));
        else if (b>center)
            set_site_tensor(b, compute_UR_on_cols(cube_as_matrix1(tt.M[b]),P.at(b-1)));
    }

    void set_site_tensor(int b, arma::Mat<T> const& M) { tt.M[b]=arma::Cube<T>(M.memptr(), Iset[b].size(), localSet[b].size(), Jset[b].size());  }

    void collectPivotError(int b, vector<double> const& pe)
    {
        pivotErrorAll[b]=pe;
        if (pe.size()>pivotError.size()) pivotError.resize(pe.size(), 0);
        for(auto i=0u; i<pe.size(); i++)
            if (pe[i]>pivotError[i])
                pivotError[i]=pe[i];
    }    

    vector<vector<double>> pivotErrorAll;           ///< The pivot error list for each bonds
    vector< IndexSet<MultiIndex> > I0, J0;          ///< Historical lists of accepted pivots for each site
};


//-------------------------------------------------------------------------------------------  CTensorCI2 -----------------------------------------


/// This class is a TensorCI2 built form an arbitrary function f(u1,u2,...un)
/// where ui can be anything as long as the user provides a list of possible values for each one.
/// Apart from the TensorCI2 including its (discrete) tensor train, the main output is the continuous tensor train: an effective separation of variables.
template<class T,class Index>
class CTensorCI2: public TensorCI2<T> {
public:
    function<T(vector<Index>)> fc;      ///< the original function
    vector< vector<Index> > xi;         ///< the original grid
    using TensorCI2<T>::P;
    using TensorCI2<T>::center;


    /// constructs a rank-1 CTensorCI2 from a function f:(u1,u2,...,un)->T  where ui is in xi[i].
    CTensorCI2(function<T(vector<Index>)> f_, vector<vector<Index>> const& xi_, TensorCI2Param par={})
        : TensorCI2<T> {tensorFun(f_, xi_), readDims(xi_), par}
        , fc(f_),xi(xi_)
    {}

    /// returns the underline continuous tensor train
    CTensorTrain<T,Index> get_CTensorTrain() const
    {
        CTensorTrain<T,Index> tt;
        for(int p=0; p<center; p++)
            tt.M.push_back( [this,p](Index x){return get_TP1_at(p,x);} );
        tt.M.push_back( [this](Index x){return get_T_at(center,x);} );
        for(int p=center+1; p<this->len(); p++)
            tt.M.push_back( [this,p](Index x){return get_P1T_at(p,x);} );
        return tt;
    }

    /// returns the matrix obtained when evaluating the tensor T[p] at value x, i.e.,
    /// A(i,j)=fc( xi(Ip[i]) + x + xi(Jp[j]) ) where xi() converts to grid point a given MultiIndex and
    /// Ip, Jp are the pivot indices at bond p, as defined in TensorCI2.
    arma::Mat<T> get_T_at(size_t p, Index x) const {
        auto Ip=to_MultiIndexG<Index>(this->Iset.at(p), xi);
        auto Jp=to_MultiIndexG<Index>(this->Jset.at(p), {xi.begin()+p+1, xi.end()} );  // because Jset[p] start at pos p+1
        vector<Index> ixj(this->len());
        arma::Mat<T> T2(Ip.size(), Jp.size());
        for(auto i=0u; i<T2.n_rows; i++) {
            std::copy(Ip[i].begin(),Ip[i].end(),ixj.begin());
            ixj[Ip[i].size()]=x;
            for(auto j=0u; j<T2.n_cols; j++) {
                // auto ixj=Ip[i]+x+Jp[j];
                std::copy(Jp[j].begin(),Jp[j].end(),ixj.begin()+Ip[i].size()+1);
                T2(i,j)=fc(ixj);
            }
        }
        return T2;
    }

    /// return the value of get_T_at(p,x)*P[p]^-1 where P[p] is the pivot matrix at bond p.
    arma::Mat<T> get_TP1_at(size_t p, Index x) const { return compute_CU_on_rows(get_T_at(p,x), P.at(p)); }

    /// return the value of get_T_at(p,x)*P[p]^-1 where P[p] is the pivot matrix at bond p.
    arma::Mat<T> get_P1T_at(size_t p, Index x) const { return compute_UR_on_cols(get_T_at(p,x), this->P.at(p-1)); }

private:
    static vector<int> readDims(vector<vector<Index>> const& xi)
    {
        vector<int> dims(xi.size());
        for(auto i=0u; i<xi.size(); i++)
            dims[i]=xi[i].size();
        return dims;
    }

    static function<T(vector<int>)> tensorFun(function<T(vector<Index>)> f, vector<vector<Index>> const& xi)
    {
        return [=](vector<int> const& id) {
            MultiIndex mi {id.begin(),id.end()};
            auto xs=to_MultiIndexG<Index>(mi, xi);
            return f({xs.begin(),xs.end()});
        };
    }

};


//-------------------------------------------------------------------------------------------  QTensorCI -----------------------------------------


/// This class is a TensorCI2 built form a function f(x1,x2,...,xn) where the binary digits of the each xi are used to buid a tensor of length n*nBit
/// Apart from the TensorCI2 including its tensor train, the main output is the quantics tensor train: a cheap representation of the function that can be saved/loaded to file
template<class T>
class QTensorCI: public TensorCI2<T> {
public:
    grid::Quantics grid;

    /// constructs a rank-1 QTensorCI from a function f:(u1,u2,...,un)->T  and the given quantics grid
    QTensorCI(function<T(vector<double>)> f_, grid::Quantics grid_, TensorCI2Param par={})
        : TensorCI2<T> {tensorFun(f_,grid_), grid_.tensorDims(), par}
        , grid {grid_}
    {}

    /// constructs a rank-1 QTensorCI from a function f:(u1)->T and the given quantics grid. Specialization for the 1d case.
    QTensorCI(function<T(double)> f_, grid::Quantics grid_, TensorCI2Param par={})
        : TensorCI2<T> {tensorFun(f_,grid_), grid_.tensorDims(), par}
        , grid {grid_}
    {}

    /// constructs a QTensorCI from a function f:(u1,u2,...,un)->T, a tensor train and the given quantics grid
    QTensorCI(function<T(vector<double>)> f_, grid::Quantics grid_, TensorTrain<T> tt_, TensorCI2Param par={})
        : TensorCI2<T> {tensorFun(f_,grid_), tt_, par}
        , grid {grid_}
    {}

    /// returns the underline quantics tensor train
    QTensorTrain<T> get_qtt() const { return {this->tt, grid}; }

    /// add the pivots to all bonds of the tci. Each point is translated to the corresponding tensor index
    void addPivotPoints(vector<vector<double>> const& xpivots)
    {
        vector<vector<int>> pivots;
        for(auto const& xv : xpivots) pivots.push_back(grid.coord_to_id(xv));
        this->addPivotsAllBonds(pivots);
    }

    /// add the pivots to all bonds of the tci. Each point is translated to the corresponding tensor index. Specialization for the 1d case.
    void addPivotValues(vector<double> const& xpivots)
    {
        if (grid.dim!=1) throw std::invalid_argument("QuanticsTCI:addPivotValues() is only for 1d functions");
        vector<vector<int>> pivots;
        for(auto const& xv : xpivots) pivots.push_back(grid.coord_to_id({xv}));
        this->addPivotsAllBonds(pivots);
    }

private:
    static function<T(vector<int>)> tensorFun(function<T(vector<double>)> f, grid::Quantics grid) {
        return [=](vector<int> const& id) { return f(grid.id_to_coord(id)); };
    }

    static function<T(vector<int>)> tensorFun(function<T(double)> f, grid::Quantics grid) {
        if (grid.dim!=1) throw std::invalid_argument("QuanticsTCI grid.dim should be 1 for a function<T(double)>");
        return [=](vector<int> const& id) { return f(grid.id_to_coord(id)[0]); };
    }

};

} // end namespace xfac


#endif // TENSOR_CI2_H
