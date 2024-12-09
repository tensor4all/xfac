#ifndef TENSOR_CI_H
#define TENSOR_CI_H

#include "xfac/index_set.h"
#include "xfac/matrix/matrix_interface.h"
#include "xfac/matrix/cross_data.h"
#include "xfac/matrix/pivot_finder.h"
#include "tensor_train.h"
#include "tensor_function.h"

#include <memory>

//----------------------------- The class TensorCI1 ------------------

namespace xfac {

using std::unique_ptr;

/// Parameters of the TensorCI1 algorithm.
struct TensorCI1Param {
    int nIter=0;                            ///< the initialization will call TensorCI1::Iterate(nIter)
    double reltol=1e-12;                    ///< CI will stop when pivotError < reltol*max(pivotError)
    vector<int> pivot1;                     ///< the first pivot (optional)
    bool fullPiv=false;                     ///< whether to search in the full Pi matrix to find a new pivot
    int nRookIter=5;                        ///< number of rook pivoting iterations, used when fullPiv=false
    vector<vector<double>> weight;          ///< weight for the tt sum. When not empty, it activates the ENV learning
    function<bool(vector<int>)> cond;       ///< cond(x)=false when x should not be a pivot
    bool useCachedFunction=true;            ///< unused: whether the internal caching should be used to avoid function call to the same point.

    operator PivotFinderParam() const { return {.fullPiv=fullPiv, .nRookIter=nRookIter}; }
};


/// This class is responsible for building a cross interpolation (CI) of an input tensor.
/// The main output is the tensor train tt (an effective separation of variables),
/// which allows many cheap computations, i.e. the integral.
template<class eT>
struct TensorCI1 {
    TensorFunction<eT> f;                                   ///< the tensor function f:(a1,a2,...,an)->eT
    TensorCI1Param param;                                    ///< parameters of the algorithm
    vector<double> pivotError;                              ///< max pivot error for each iteration
    vector< vector<MultiIndex> > Iset, localSet, Jset;      ///< collection of MultiIndex for each site: left, site, and right set of multi index
    vector< arma::Cube<eT> > T3;                            ///< the T tensor at each site: T(i,a,j)=f(i+k+j) where i:Iset, a:localSet, j:Jset
    vector< arma::Mat<eT> > P;                              ///< the pivot matrix at each site p: P(i,j)=f(i+j) where i:Iset[p+1], j:Jset[p]
    int cIter=0;                                            ///< counter of iterations, roughly equals to rank(). Used for sweeping only.


    TensorCI1(){}

    /// constructs a rank-1 TCI from a function f:(a1,a2,...,an)->eT  where the index ai is in [0,localDim[i]).
    TensorCI1(function<eT(vector<int>)> f_, vector<int> localDim, TensorCI1Param par_={})
        : f {f_}
        , param {par_}
        , pivotError(1,1.0)
        , Iset {localDim.size()}
        , localSet {localDim.size()}
        , Jset {localDim.size()}
        , T3 {localDim.size()}
        , P {localDim.size()}
        , pivotErrorLastIter(localDim.size()-1,1.0)
    {
        if (param.pivot1.empty())
            param.pivot1.resize(localDim.size(), 0);
        pivotError[0]=std::abs(f_(param.pivot1));
        if (pivotError[0]==0.0)
            throw std::invalid_argument("Not expecting f(pivot1)=0. Provide a better first pivot in the param");

        // fill localSet, Iset Jset
        for(int p=0; p<len(); p++)
        {
            for(MultiIndex::value_type i=0; i<localDim[p]; i++) localSet[p].push_back({i});
            Iset[p].push_back({param.pivot1.begin(), param.pivot1.begin()+p});
            Jset[p].push_back({param.pivot1.begin()+p+1, param.pivot1.end()});
        }

        // fill Pi tensors and cross data for each bond
        for(int p=0; p<len()-1; p++) {
            Pi_mat.push_back( buildPiAt(p) );
            if (param.cond) Pi_bool.push_back( buildPiBoolAt(p) );
            cross.push_back( {Pi_mat.back()->n_rows, Pi_mat.back()->n_cols} );
        }

        // fill the T tensors and P matrices
        for(int p=0; p<len()-1; p++)
        {
            matrix_t& Pi= PiAt(p);
            cross[p].addPivot(Pi.Iset.pos(Iset[p+1][0]), Pi.Jset.pos(Jset[p][0]), Pi);
            if (p==0)
                T3[p]=arma::Cube<eT>(cross[p].C.memptr(), Iset[p].size(), localSet[p].size(), Jset[p].size() );
            T3[p+1]=arma::Cube<eT>(cross[p].R.memptr(), Iset[p+1].size(), localSet[p+1].size(), Jset[p+1].size() );
            P[p]=cross[p].pivotMat();
        }
        P[len()-1]=arma::Mat<eT>(1,1, arma::fill::eye);

        if (!param.weight.empty())
            tt_sum=TT_sum<eT>(get_TensorTrain(0), param.weight );
        iterate(param.nIter);
    }

    /// constructs a TensorCI1 from a another TCI, while setting the new param.
    template<class Tci>
    TensorCI1(Tci const& tci, TensorCI1Param param_)
        : f {tci.f}
        , param {param_}
        , pivotError(tci.pivotError)
        , Iset {tci.Iset.begin(), tci.Iset.end()}
        , localSet {tci.localSet.begin(), tci.localSet.end()}
        , Jset {tci.Jset.begin(), tci.Jset.end()}
        , T3 {tci.localSet.size()}
        , P {tci.localSet.size()}
        , pivotErrorLastIter(tci.len()-1,1.0)
    {
        cIter=tci.pivotError.size();
        try {
            // fill the T tensors and P matrices
            for(int p=0; p<len(); p++)
            {
                IndexSet<MultiIndex> Ib=kron(Iset[p],localSet[p]);
                auto M=f.eval(Ib, Jset[p]);
                T3[p]=arma::Cube<eT>(M.memptr(), Iset[p].size(), localSet[p].size(), Jset[p].size());
                P[p]= (p==len()-1) ? arma::Mat<eT>(1,1, arma::fill::eye)
                                   : M.rows(arma::conv_to<arma::uvec>::from(Ib.pos(Iset[p+1])));
            }

            // fill Pi tensors and cross data for each bond
            for(int p=0; p<len()-1; p++) {
                Pi_mat.push_back( buildPiAt(p) );
                if (param.cond) Pi_bool.push_back( buildPiBoolAt(p) );
                cross.push_back( {PiAt(p).Iset.pos(Iset[p+1]), PiAt(p).Jset.pos(Jset[p]),
                                  cube_as_matrix2(T3[p]), cube_as_matrix1(T3[p+1])} );
            }
        }
        catch (...) {
            throw std::runtime_error("constructor TensorCI1(tci) failed. Try tci.makeCanonical() before");
        }

        if (!param.weight.empty())
            tt_sum=TT_sum<eT>(get_TensorTrain(0), param.weight );
    }

    /// copy constructor
    TensorCI1(TensorCI1<eT> const& tci) : TensorCI1(tci, tci.param) {}

    /// makes nIter half sweeps, i.e., tries to add a pivot on each bond of the tensor CI, nIter times.
    void iterate(int nIter=1)
    {
        for(int t=0;t<nIter; t++,cIter++)
        {
            if (cIter == 1) continue;
            else if (cIter%2==0)
                for(auto p=0; p<len()-1; p++)
                    addPivotAt(p);
            else
                for(int p=len()-2; p>=0; p--) addPivotAt(p);
            pivotError.push_back( *std::max_element(pivotErrorLastIter.begin(), pivotErrorLastIter.end()) );
        }
    }

    /// Tries to add a pivot at bond p
    void addPivotAt(int p)
    {
        PivotData pivot=buildPivotFinderAt(p)(PiAt(p), cross[p]);
        pivotErrorLastIter[p]=pivot.error;
        if (pivot.error < *std::max_element(pivotError.begin(), pivotError.end()) * param.reltol ) return;

        addPivotRowAt(p,pivot.i);
        addPivotColAt(p,pivot.j);
        updateEnvAt(p);
    }

    /// returns the length of the tensor
    int len() const { return localSet.size(); }

    /// computes the max error |f-ftci| if the tensor is smaller than max_n_eval.
    double trueError(size_t max_n_eval=1e6) const { return get_TensorTrain().trueError(f.f, max_n_eval); }

    ///get the underline tensor train with the given CI-canonical center. Negative center means center+len()
    TensorTrain<eT> get_TensorTrain(int center=-1) const
    {
        if (center<0) center+=len();
        TensorTrain<eT> tt;
        for(int p=0; p<center; p++)
            tt.M.push_back(get_TP1_at(p));
        tt.M.push_back(T3[center]);
        for(int p=center+1; p<len(); p++) tt.M.push_back(get_P1T_at(p));
        return tt;
    }

    /// computes the T*P^-1 at site p
    arma::Cube<eT> get_TP1_at(int p) const
    {
        auto TP1=mat_AB1(cube_as_matrix2(T3.at(p)), P[p]);
        return {TP1.memptr(), Iset[p].size(), localSet[p].size(), Jset[p].size()};
    }

    /// computes the P^-1*T at site p
    arma::Cube<eT> get_P1T_at(int p) const
    {
        auto P1T=mat_A1B(P.at(p-1), cube_as_matrix1(T3[p]));
        return {P1T.memptr(), Iset[p].size(), localSet[p].size(), Jset[p].size()};
    }

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

private:
    using matrix_t=IMatrix<eT,MultiIndex>;

    ///returns the Pi matrix at bond p: Pi(ia,bj)=f(i+a+b+j) where i:Iset[p], a,b:localSet, j:Jset[p+1]
    matrix_t& PiAt(int p) { return *Pi_mat.at(p); }
    const matrix_t& PiAt(int p) const { return *Pi_mat.at(p); }

    //----------------------------------------------------- build methods-----------------

    /// @param i/j is the first/last MultiIndex of the Pi 4-leg tensor.
    unique_ptr<matrix_t> buildPiAt(int p)
    {
        auto Ai=[this](MultiIndex const& ix,MultiIndex const& yj) { return f(ix+yj); };
        auto I=kron(Iset[p], localSet[p]);
        auto J=kron(localSet[p+1], Jset[p+1]);
        return make_IMatrix<eT,MultiIndex>(Ai, I, J, param.fullPiv);
    }

    /// @param i/j is the first/last MultiIndex of the Pi 4-leg tensor.
    MatDense<int,MultiIndex> buildPiBoolAt(int p)
    {
        function Ai=[this](MultiIndex const& ix,MultiIndex const& yj) {
            auto ij=ix+yj;
            return param.cond({ij.begin(), ij.end()});
        };
        auto I=kron(Iset[p], localSet[p]);
        auto J=kron(localSet[p+1], Jset[p+1]);
        return {Ai, I, J};
    }

    PivotFinder<eT> buildPivotFinderAt(int p)
    {
        PivotFinderParam par=param;
        if (!tt_sum.L.empty()) { // ENV learning
            par.weightRow=arma::abs(arma::kron(tt_sum.L[p].as_col(), arma::vec(param.weight[p])));
            par.weightCol=arma::abs(arma::kron(arma::vec(param.weight[p]), tt_sum.R[p+1].as_col()));
        }
        if (param.cond) par.fBool=Pi_bool[p]; // pivot condition
        return par;
    }

    //----------------------------------------------------- update methods-----------------

    void addPivotRowAt(int p, int pivot_i)
    {
        const matrix_t& Pi=PiAt(p);
        cross[p].addPivotRow(pivot_i, Pi);
        Pi.forgetRow(pivot_i);
        Iset[p+1].push_back(Pi.Iset.at(pivot_i));
        T3[p+1]=arma::Cube<eT>(cross[p].R.memptr(), Iset[p+1].size(), localSet[p+1].size(), Jset[p+1].size());
        P[p]=cross[p].pivotMat();
        updatePiRowsAt(p+1);
    }

    void updatePiRowsAt(int p)
    {
        if (p>=len()-1) return;
        vector<int> P=PiAt(p).setRows( kron(Iset[p], localSet[p]) );
        cross[p].setRows(cube_as_matrix2(T3[p]), P);
        if (param.cond) Pi_bool[p].setRows( kron(Iset[p], localSet[p]) );
    }

    void addPivotColAt(int p, int pivot_j)
    {
        const matrix_t& Pi=PiAt(p);
        cross[p].addPivotCol(pivot_j, Pi);
        Pi.forgetCol(pivot_j);
        Jset[p].push_back(Pi.Jset.at(pivot_j));
        T3[p]=arma::Cube<eT>(cross[p].C.memptr(), Iset[p].size(), localSet[p].size(), Jset[p].size());
        P[p]=cross[p].pivotMat();
        updatePiColsAt(p-1);
    }

    void updatePiColsAt(int p)
    {
        if (p<0) return;
        vector<int> Q=PiAt(p).setCols( kron(localSet[p+1], Jset[p+1]) );
        cross[p].setCols(cube_as_matrix1(T3[p+1]), Q);
        if (param.cond) Pi_bool[p].setCols( kron(localSet[p+1], Jset[p+1]) );
    }

    /// ENV learning
    void updateEnvAt(int p)
    {
        if (tt_sum.L.empty()) return;
        tt_sum.updateSite(p, get_TP1_at(p), true);
        tt_sum.updateSite(p+1, get_P1T_at(p+1), false);
    }

    //----------------------------------------------------- data --------------------

    vector< unique_ptr<matrix_t> >  Pi_mat;             ///< the Pi matrix at each bond: Pi(ia,bj)=f(i+a+b+j) where i:Iset, a,b:localSet, j:Jset
    vector< CrossData<eT> > cross;                      ///< the cross matrix at each bond with respect to the Pi_mat
    TT_sum<eT> tt_sum;                                  ///< The partial contration of the tt sum, used only when ENV learning is activated.
    vector< MatDense<int,MultiIndex> >  Pi_bool;        ///< The Pi matrix for par.cond
    vector<double> pivotErrorLastIter;
};



//-------------------------------------------------------------------------------------------  CTensorCI1 -----------------------------------------



/// This class is a TensorCI1 built from an arbitrary function f(u1,u2,...un)
/// where xi can be anything as long as the user provides a list of possible values for each one.
/// Apart from the TensorCI1 including its (discrete) tensor train, the main output is the continuous tensor train: an effective separation of variables.
template<class T,class Index>
class CTensorCI1: public TensorCI1<T> {
public:
    function<T(vector<Index>)> fc;          ///< the original function
    vector< vector<Index> > xi;             ///< the original grid

    /// constructs a rank-1 TCI from a function f:(u1,u2,...,un)->T  where ui is in xi[i].
    CTensorCI1(function<T(vector<Index>)> f_, vector<vector<Index>> const& xi_, TensorCI1Param par={})
        : TensorCI1<T> {tensorFun(f_, xi_), readDims(xi_), par}
        , fc(f_),xi(xi_)
    {}

    /// returns the underline continuous tensor train with the given CI-canonical center. Negative center means center+len()
    CTensorTrain<T,Index> get_CTensorTrain(int center=-1) const
    {
        if (center<0) center+=this->len();
        CTensorTrain<T,Index> tt;
        for(int p=0; p<center; p++)
            tt.M.push_back( [this,p](Index x){return get_TP1_at(p,x);} );
        tt.M.push_back( [this,center](Index x){return get_T_at(center,x);} );
        for(int p=center+1; p<this->len(); p++)
            tt.M.push_back( [this,p](Index x){return get_P1T_at(p,x);} );
        return tt;
    }

    /// returns the matrix obtained when evaluating the tensor T[p] at value x, i.e.,
    /// A(i,j)=f( xi(Ip[i]) + x + xi(Jp[j]) ) where xi() converts to grid point a given MultiIndex and
    /// Ip, Jp are the pivot indices at bond p, as defined in TensorCI1.
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
    arma::Mat<T> get_TP1_at(size_t p, Index x) const { return mat_AB1(get_T_at(p,x), this->P.at(p)); }
    /// return the value of get_T_at(p,x)*P[p]^-1 where P[p] is the pivot matrix at bond p.
    arma::Mat<T> get_P1T_at(size_t p, Index x) const { return mat_A1B(this->P.at(p-1), get_T_at(p,x)); }

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

}// end namespace xfac


#endif // TENSOR_CI_H
