#ifndef TENSOR_FUNCTION_H
#define TENSOR_FUNCTION_H

#include "xfac/index_set.h"
#include "xfac/matrix/mat_decomp.h"

#include <unordered_map>
#include <algorithm>

namespace xfac {

using std::vector;
using std::function;


///< To store the tensor function f:(a1,a2,...,an)->T
template<class T>
struct TensorFunction {
    function<T(vector<int>)> f;
    bool useCache=false;

    TensorFunction(function<T(vector<int>)> f_, bool useCache_=false) : f(f_), useCache(useCache_) {}

    T operator()(MultiIndex const& id) const { cEval+=1; return f({id.begin(),id.end()}); }

    arma::Mat<T> evalCache(vector<MultiIndex> const& I, vector<MultiIndex> const& J) const
    {
        arma::Mat<T> values(I.size(), J.size(), arma::fill::none);
        vector<std::tuple<size_t,size_t, decltype(dat.begin())>> pos_eval;
        for(auto i=0u; i<I.size(); i++)
            for(auto j=0u; j<J.size(); j++) {
                auto [it,isNew]=dat.try_emplace(I[i]+J[j]);
                if (!isNew)
                    values(i,j)=it->second;
                else
                    pos_eval.push_back({i,j,it});
            }
        #pragma omp parallel for
        for(auto [i,j,it] : pos_eval)
            values(i,j)=it->second=f({it->first.begin(), it->first.end()});
        return values;
    }

    arma::Mat<T> eval(vector<MultiIndex> const& I, vector<MultiIndex> const& J) const
    {
        if (useCache) return evalCache(I,J);
        arma::Mat<T> values(I.size(), J.size(), arma::fill::none);
        #pragma omp parallel for collapse(2)
        for(auto i=0u; i<I.size(); i++)
            for(auto j=0u; j<J.size(); j++) {
                MultiIndex ij=I[i]+J[j];
                values(i,j)=f({ij.begin(), ij.end()});
            }
        cEval += values.size();
        return values;
    }

    MatFun<T> matfun(vector<MultiIndex> const& I, vector<MultiIndex> const& J) const
    {
        auto submat=[this,I,J](vector<int> const& I0, vector<int> const& J0) {
            vector<MultiIndex> Is(I0.size()), Js(J0.size());
            for(auto i=0u; i<Is.size(); i++) Is[i]=I[I0[i]];
            for(auto j=0u; j<Js.size(); j++) Js[j]=J[J0[j]];
            return eval(Is,Js);
        };
        return {I.size(), J.size(), submat};
    }

    void clearCache() { cEval+=dat.size(); dat.clear(); }
    size_t nEval() const { return dat.size()+cEval; }

private:
    mutable size_t cEval=0;
    mutable std::unordered_map<MultiIndex,T> dat;
};

} // end namespace xfac


#endif // TENSOR_FUNCTION_H
