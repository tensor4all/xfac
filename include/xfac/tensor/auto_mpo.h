#ifndef AUTO_MPO_H
#define AUTO_MPO_H

#include "tensor_train.h"

namespace xfac {

/**
 * This approach is applicable when the number of term is not too big.
*/
namespace autompo {

template<class T=double, int d=2>
using LocOp=typename  arma::Mat<T>::template fixed<d,d>;

template<class T=double, int d=2>
struct ProdOp: public map<int,LocOp<T,d>> {

    using map<int,LocOp<T,d>>::map;
    static LocOp<T,d> one;

    int length() const { return this->empty() ? 0 : this->crbegin()->first+1; }

    TensorTrain<T> to_tensorTrain(int length_) const
    {
        TensorTrain<T> tt;
        arma::Cube<T> onec(one.memptr(),1,one.size(),1, false);
        tt.M.resize(length_, onec);
        for(auto const& [pos,op] : (*this) )
            tt.M[pos]=arma::Cube<T>(op.memptr(), 1, op.size(), 1);
        return tt;
    }

    T overlap(const TensorTrain<T>& mps) const
    {
        vector<vector<T>> weights (mps.M.size(), {one.begin(), one.end()});
        for(auto const& [pos,op] : (*this))
            weights[pos]={op.begin(), op.end()};
        return mps.sum(weights);
    }
};

template<class T, int d>
LocOp<T,d> ProdOp<T,d>::one=LocOp<T,d>(arma::fill::eye);


template<class T, int d>
ProdOp<T,d> operator*(ProdOp<T,d> const& A, ProdOp<T,d> const& B)
{
    vector<int> iA, iB, iAB, iAb, iaB;
    for (const auto& x : A) iA.emplace_back(x.first);
    for (const auto& x : B) iB.emplace_back(x.first);
    std::set_intersection(iA.begin(), iA.end(),
                          iB.begin(), iB.end(), std::back_inserter(iAB));
    std::set_difference(iA.begin(), iA.end(),
                        iB.begin(), iB.end(), std::back_inserter(iAb));

    std::set_difference(iB.begin(), iB.end(),
                        iA.begin(), iA.end(), std::back_inserter(iaB));

    ProdOp<T,d> C;
    for(int i : iAB)
            C[i]=A.at(i)*B.at(i);
    for(int i : iAb)
            C[i]=A.at(i);
    for(int i : iaB)
            C[i]=B.at(i);
    return C;
}

template<class T, int d>
ProdOp<T,d> operator*(ProdOp<T,d> A, T c) { A.begin()->second*=c; return A; }


/// just a collection of ProdOp
template<class T=double, int d=2>
struct PolyOp {

    /// These parameters control the conversion to mps. They are also used when the size of the collection reached maxNTerm.
    int compressEvery=20;       ///<   number of terms to form one mps.
    int maxNTerm=100000;        ///<   to protect against memory overflow, when this number is reached, the terms are compressed to tt.
    double reltol=1e-9;         ///<   the relative tolerance of the matrix compression.
    bool use_svd=false;         ///<   the method for compression: CI by default.

    PolyOp<>& operator+=(ProdOp<T,d> const& A) { add(A); return *this; }
    PolyOp<>& operator+=(PolyOp<T,d> const& As)
    {
        for(const ProdOp<T,d>& A : As.ops) add(A);
        tt = tt + As.tt;
        if (!tt.M.empty()) tt.compressCI(reltol);
        tt_nTerm += As.tt_nTerm;
        return *this;
    }

    void add(ProdOp<T,d> const& A)
    {
        ops.push_back(A);
        if(ops.size() == maxNTerm) {
            tt = to_tensorTrain();
            tt_nTerm+=ops.size();
            ops={};
        }
    }

    int length() const
    {
        int length=tt.M.size();
        for(auto const& x : ops)
            length=std::max(length, x.length());
        return length;
    }

    size_t nTerm() const { return tt_nTerm+ops.size(); }

    TensorTrain<T> to_tensorTrain() const
    {
        int L=length();
        int n_mps=ops.size()/compressEvery;
        if (n_mps*compressEvery != ops.size()) n_mps++;

        vector<TensorTrain<T>> tts(n_mps);
        #pragma omp parallel for
        for(auto i=0; i< n_mps; i++) {
            for (auto ii=0; ii<compressEvery; ii++)
                if (auto pos=i*compressEvery+ii; pos<ops.size())
                    tts[i] = tts[i] + ops[pos].to_tensorTrain(L);
            if (use_svd) tts[i].compressSVD(reltol);
            else tts[i].compressCI(reltol);
        }
        if (tt.M.empty()) return sum(tts,reltol);
        auto tt2 = fix_tt_len() + sum(tts,reltol);
        tt2.compressCI(reltol);
        return tt2;
    }

    /// compute the overlap with a given mps (the physical dimensions should match)
    T overlap(const TensorTrain<T>& mps) const
    {
        T sum=0;
        for(const auto& x : ops)
            sum+=x.overlap(mps);
        sum += mps.overlap(fix_tt_len());
        return sum;
    }

private:
    TensorTrain<T> fix_tt_len() const
    {
        if (tt.M.empty()) return {};
        arma::Cube<T> onec(ProdOp<T,d>::one.memptr(),1,ProdOp<T,d>::one.size(),1, false);
        auto tt2=tt;
        tt2.M.resize(length(),onec);
        return tt2;
    }

    vector<ProdOp<T,d>> ops;
    TensorTrain<T> tt;
    size_t tt_nTerm=0;
};

template<class T, int d>
PolyOp<T,d> operator+(PolyOp<T,d> A, PolyOp<T,d> const& B) { return A+=B; }


} // end namespace autompo
} // end namespace xfac

#endif // AUTO_MPO_H
