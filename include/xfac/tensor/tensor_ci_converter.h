#ifndef TENSOR_CI_CONVERTER_H
#define TENSOR_CI_CONVERTER_H

#include "xfac/tensor/tensor_ci.h"
#include "xfac/tensor/tensor_ci_2.h"

namespace xfac {

inline TensorCI1Param to_tci1Param(TensorCI2Param const& x)
{
    TensorCI1Param y;
    y.reltol=x.reltol;
    y.pivot1=x.pivot1;
    y.fullPiv=x.fullPiv;
    y.nRookIter=x.nRookIter;
    y.weight=x.weight;
    y.cond=x.cond;
    y.useCachedFunction=y.useCachedFunction;
    return y;
}

inline TensorCI2Param to_tci2Param(TensorCI1Param const& x)
{
    TensorCI2Param y;
    y.reltol=x.reltol;
    y.pivot1=x.pivot1;
    y.fullPiv=x.fullPiv;
    y.nRookIter=x.nRookIter;
    y.weight=x.weight;
    y.cond=x.cond;
    y.useCachedFunction=y.useCachedFunction;
    return y;
}

namespace impl {
template<class Tci>
vector<int> readDims(Tci const& ci)
{
    vector<int> dims(ci.localSet.size());
    for(auto i=0u; i<dims.size(); i++)
        dims[i]=ci.localSet[i].size();
    return dims;
}
}

template<class T>
TensorCI1<T> to_tci1(TensorCI2<T> const& tci2)
{
    return TensorCI1<T>(tci2, to_tci1Param(tci2.param));
}

template<class T>
TensorCI2<T> to_tci2(TensorCI1<T> const& tci1, function<T(vector<int>)> g, TensorCI2Param par)
{
    TensorCI2<T> tci2(g, impl::readDims(tci1), par);
    tci2.addPivots(tci1);
    return tci2;
}

template<class T>
TensorCI2<T> to_tci2(TensorCI1<T> const& tci1, function<T(vector<int>)> g)
{
    TensorCI2Param par=to_tci2Param(tci1.param);
    par.bondDim=tci1.pivotError.size();
    return to_tci2(tci1,g,par);
}

template<class T>
TensorCI2<T> to_tci2(TensorCI1<T> const& tci1, TensorCI2Param par)
{
    TensorCI2<T> tci2(tci1.f, impl::readDims(tci1), par);
    tci2.addPivots(tci1);
    return tci2;
}

template<class T>
TensorCI2<T> to_tci2(TensorCI1<T> const& tci1)
{
    TensorCI2Param par=to_tci2Param(tci1.param);
    par.bondDim=tci1.pivotError.size();
    return to_tci2(tci1,par);
}

} // end namespace xfac


#endif // TENSOR_CI_CONVERTER_H
