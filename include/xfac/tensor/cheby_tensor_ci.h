#ifndef CHEBY_TENSOR_CI
#define CHEBY_TENSOR_CI

#include "xfac/tensor/tensor_ci.h"
#include "../extern/cheby/chebyshev.hpp"

namespace xfac {

using mgrid = std::vector<std::tuple<double, double, size_t>>;

/// Parameters of the TensorCI algorithm.
struct TensorCIParamCheby: TensorCI1Param {
    bool env=true;  ///< tci with environment
    function<bool(vector<double>)> ccond;       ///< contineous condition ccond(x)=false when x should not be a pivot
};



template<class T,class Index>
class CTensorCheby : public TensorCI1<T>
{

public:

    bool do_prepare_simplex_v = true;
    vector< vector<Index> > xi;
    vector<chebyshev::ChebyMat<T>> fii;
    vector<mgrid> grid;
    mutable std::map<int, chebyshev::ChebyMat<T>> chebyTcache;
    

    chebyshev::ChebyMat<T> fint, gint;

    CTensorCheby() = default;

    CTensorCheby(function<T(vector<int>)> f_, vector<mgrid> const& grid_, TensorCIParamCheby par={})
      : TensorCI1<T>{f_, readDims(grid_), add_cheby_env(grid_, par)}
      , xi{make_xi(grid_)}
      , grid{grid_}
      {}

    CTensorCheby(function<T(vector<int>)> f_, int n, mgrid const& grid_, TensorCIParamCheby par={})
      : CTensorCheby{f_, vector(n, grid_), par}  {}

    CTensorCheby(function<T(vector<Index>)> f_, vector<mgrid> const& grid_,  TensorCIParamCheby par={})
      : CTensorCheby{tensorFun(f_, make_xi(grid_)), grid_, par}  {}

    CTensorCheby(function<T(vector<Index>)> f_, int n, mgrid const& grid_, TensorCIParamCheby par={})
      : CTensorCheby{f_, vector(n, grid_), par}  {}

    /// returns the CI formula for xs.
    T eval(vector<Index> const& xs)
    {
        arma::Mat<T> prod(1, 1, arma::fill::eye);
        for(int p=0; p<this->len(); p++) {
            arma::Mat<T> M = p==this->len()-1 ? get_T_at(p,xs[p])
                                              : get_TP1_at(p,xs[p]);
            prod=prod*M;
        }
        return prod(0,0);
    }

    T simplex_v(Index const& x, bool integrate=true)
    {
        if (do_prepare_simplex_v) prepare_simplex_v(integrate);
        auto fl = [&](double y){return get_T_at(0, y);};
        return fint.convolute_left(fl, x)(0, 0);
    }


    arma::Mat<T> get_T_at(size_t p, Index x) {
        chebyshev::ChebyMat<T> chebyT;
        if (chebyTcache.contains(p)) {
            chebyT = chebyTcache[p];
        } else {
            std::unordered_map<double,int> xim;
            chebyT = chebyshev::ChebyMat<T>(grid[p]);
            for(size_t i=0;i<chebyT.xi.size();i++) xim[chebyT.xi[i]]=i;
            auto fk = [&](double y){ return TensorCI1<T>::T3[p].col(xim[y]); };
            chebyT.interpolate(fk);
            chebyTcache[p] = chebyT;
        }
        return chebyT.eval(x);
    }


    /// return the value of get_T_at(p,x)*P[p]^-1 where P[p] is the pivot matrix at bond p.
    arma::Mat<T> get_TP1_at(size_t p, Index x) { return mat_AB1(get_T_at(p,x), this->P[p]);}
    arma::Mat<T> get_P1T_at(size_t p, Index x) { return mat_A1B(this->P[p-1], get_T_at(p,x)); }


    void iterate(int nIter=1){
        TensorCI1<T>::iterate(nIter);
        chebyTcache.clear();
        do_prepare_simplex_v = true;
    }



private:

    void prepare_simplex_v(bool integrate)
    {
        fint = chebyshev::ChebyMat<T>(grid[this->len()-1]);
        auto fl = [&](double x){return get_P1T_at(this->len()-1, x);};
        fint.interpolate(fl);
        if (integrate)
            fint.integrate();

        for(size_t k=this->len()-2; k>0; k--) {
            auto fk = [&](double x){return get_P1T_at(k, x);};
            fint.convolute_inplace_left(fk);
        }

        do_prepare_simplex_v = false;
    }

    static vector<int> readDims(vector<mgrid> const& grids)
    {
        vector<int> dims;
        for (auto const& grid : grids)
        {
            int ndim = 0;
            for (auto const& [a, b, n] : grid)
            {
                ndim += n;
            }
            dims.push_back(ndim);
        }
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

    std::vector<std::vector<double>> make_xi(vector<mgrid> const& grid) const
    {
        std::vector<std::vector<double>> xiv;
        for (auto const& x : grid){
            xiv.push_back(chebyshev::cheby_multi_xi(x));
        }
        return xiv;
    }

    TensorCI1Param add_cheby_env(vector<mgrid> const& grid, TensorCIParamCheby const& par_) const
    {
        TensorCI1Param par{par_};

        if (par_.env && !par_.weight.empty())
            throw std::runtime_error("CTensorCheby(): par.wi and par.env mutual exclusive");

        if (par_.env)
        {
            std::vector<std::vector<double>> wi;
            for (auto const& x : grid){
                wi.push_back(chebyshev::cheby_multi_wi(x));
            }
            par.weight = wi;
        }
        if (par_.ccond){
            std::vector<std::vector<double>> xi;
            for (auto const& x : grid){
                xi.push_back(chebyshev::cheby_multi_xi(x));
            }
            par.cond = [=](std::vector<int> const& id)
            {
                std::vector<double> xc(id.size());
                for (auto i=0; i<id.size(); i++)
                    xc[i] = xi[i][id[i]];
                return par_.ccond(xc);
            };
        }
        return par;
    }

};


}// end namespace xfac

#endif // CHEBY_TENSOR_CI
