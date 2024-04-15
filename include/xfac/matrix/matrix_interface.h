#ifndef MATRIX_INTERFACE_H
#define MATRIX_INTERFACE_H

#include<complex>
#include<vector>
#include<map>
#include<memory>
#define ARMA_DONT_USE_OPENMP
#include <armadillo>
#include "xfac/index_set.h"

namespace xfac {

using std::pair;
using std::function;
using std::vector;
using std::map;

//--------------------------------------------------- Technical point --------------------

/**
 * Here we have two diamond inheritance scheme. See for instance
 * https://stackoverflow.com/questions/2659116/how-does-virtual-inheritance-solve-the-diamond-multiple-inheritance-ambiguit
 *
 *                  IMatrix<T>                                      IMatrix<T>
 *                 /         \                                     /         \
 *          IMatrix<T,I>   MatDense<T>                      IMatrix<T,I>   MatLazy<T>
 *                 \         /                                     \         /
 *                 MatDense<T,I>                                   MatLazy<T,I>
 *
**/

//---------------------------------------------------- IMatrix ---------------------------

template<class T,class... Indices> class IMatrix;

/// Interface for a matrix that we need in our implementation.
template<class T>
class IMatrix<T>
{
public:
    size_t n_rows, n_cols;
    IMatrix()=default;
    IMatrix(size_t n_rows, size_t n_cols): n_rows(n_rows), n_cols(n_cols) {}
    virtual ~IMatrix() = default;

    /// return the values ordered by columns
    virtual vector<T> submat(vector<int> const& rows, vector<int> const& cols) const = 0;

    /// return the values of the matrix at the given list of pairs (row,column).
    virtual vector<T> eval(vector<pair<int,int>> const& ids) const = 0;

    /// Useful in case of using cache
    virtual void forgetRow(int) const {}

    /// Useful in case of using cache
    virtual void forgetCol(int) const {}
};


/// similar to IMatrix<T> but the Index can be anything: i.e double, or MultiIndex
/// whenever the possible values are provided in Iset, Jset
template<class T, class Index>
class IMatrix<T,Index>: virtual public IMatrix<T> {
public:
    function<T(Index,Index)> A;
    IndexSet<Index> Iset, Jset;
    IMatrix()=default;
    IMatrix(function<T(Index,Index)> A_, vector<Index> const& Iset_, vector<Index> const& Jset_)
        : IMatrix<T> {Iset_.size(), Jset_.size()}
        , A {A_}, Iset {Iset_}, Jset {Jset_}
    {}

    /// returns the underline matrix function (i,j)->A(Iset[i],Jset[j])
    function<T(int, int)> matFun() const
    {
        return [A = A, I = Iset.from_int(), J = Jset.from_int()](int i, int j) {
            return A(I.at(i), J.at(j));
        };
    }

    /// returns the values A(x, Jset[J])
    vector<T> evalHyb(Index x, vector<int> const& J) const
    {
        vector<T> values(J.size());
        #pragma omp parallel for
        for(auto j=0u;j<J.size();j++)
            values[j]=A(x, Jset[J[j]]);
        return values;
    }

    /// returns the values A(Iset[I], y)
    vector<T> evalHyb(vector<int> const& I, Index y) const
    {
        vector<T> values(I.size());
        #pragma omp parallel for
        for(auto i=0u;i<I.size();i++)
            values[i]=A(Iset[I[i]], y);
        return values;
    }

    /// and return the permutation that transforms the rows:  i -> P[i]
    virtual vector<int> setRows(vector<Index> const& i_set)=0;
    /// and return the permutation that transforms the columns:  j -> Q[j]
    virtual vector<int> setCols(vector<Index> const& j_set)=0;
};


//---------------------------------------------------- Dense matrix ---------------------------

template<class T,class...Indices> class MatDense;


/// A dense matrix implementing an IMatrix
template<class T>
class MatDense<T>: virtual public IMatrix<T> {
public:
    MatDense()=default;

    MatDense(arma::Mat<T> A): IMatrix<T>(A.n_rows,A.n_cols), data(std::move(A)) {}

    MatDense(function<T(int,int)> f, int n_rows, int n_cols)
        : IMatrix<T>(n_rows,n_cols), data(n_rows,n_cols,arma::fill::none)
    {
        #pragma omp parallel for collapse(2)
        for(auto j=0;j<n_cols;j++)
            for(auto i=0;i<n_rows;i++)
                data(i,j)=f(i,j);
    }

    const T& operator()(int i,int j) const {return data(i,j);}

    vector<T> submat(vector<int> const& rows, vector<int> const& cols) const override
    {
        using namespace arma;
        auto I0=conv_to<uvec>::from(rows);
        auto J0=conv_to<uvec>::from(cols);
        arma::Mat<T> sm=data.submat(I0,J0);
        return {sm.begin(), sm.end()};
    }
    vector<T> eval(vector<pair<int,int>> const& ids) const override
    {
        vector<T> values;
        values.reserve(ids.size());
        for(auto [i,j]:ids) values.push_back(data(i,j));
        return values;
    }

    /// Increase the rows of the matrix to n_rows, while reordering its old rows according to P: row i -> row P[i].
    /// The new matrix function is required
    void setRows(int n_rows, vector<int> const& P, function<T(int,int)> fnew)
    {
        arma::Mat<T> data2(n_rows, this->n_cols, arma::fill::none);
        data2.rows(arma::conv_to<arma::uvec>::from(P))=data;
        vector<int> Pc=set_diff(n_rows,P);  // complement
        #pragma omp parallel for collapse(2)
        for(auto j=0u; j<this->n_cols; j++)
            for(auto i:Pc)
                data2(i,j)=fnew(i,j);
        data=data2;
        this->n_rows=n_rows;
    }

    /// Increase the cols of the matrix to n_cols, while reordering its old cols according to Q: col j -> col Q[j].
    /// The new matrix function is required
    void setCols(int n_cols, vector<int> const& Q, function<T(int,int)> fnew)
    {
        arma::Mat<T> data2(this->n_rows, n_cols, arma::fill::none);
        data2.cols(arma::conv_to<arma::uvec>::from(Q))=data;
        vector<int> Qc=set_diff(n_cols,Q);  // complement
        #pragma omp parallel for collapse(2)
        for(auto j:Qc)
            for(auto i=0u; i<this->n_rows; i++)
                data2(i,j)=fnew(i,j);
        data=data2;
        this->n_cols=n_cols;
    }

private:
    arma::Mat<T> data;
};

/// similar to MatDense but the Index can be anything: i.e double, or MultiIndex
/// whenever the possible values are provided in Iset, Jset
template<class T, class Index>
class MatDense<T,Index>: public IMatrix<T,Index>, public MatDense<T> {
public:
    MatDense()=default;
    MatDense(function<T(Index,Index)> A_, vector<Index> const& i_set, vector<Index> const& j_set)
        : IMatrix<T>(i_set.size(), j_set.size())
        , IMatrix<T,Index>(A_, i_set, j_set)
        , MatDense<T>(IMatrix<T,Index>::matFun(), IMatrix<T>::n_rows, IMatrix<T>::n_cols)
    {}

    vector<int> setRows(vector<Index> const& i_set) override
    {
        IndexSet<Index> I(i_set);
        vector<int> pos=I.pos(this->Iset.from_int());
        IMatrix<T,Index>::Iset=I;
        MatDense<T>::setRows(i_set.size(), pos, IMatrix<T,Index>::matFun());
        return pos;
    }

    vector<int> setCols(vector<Index> const& j_set) override
    {
        IndexSet<Index> J(j_set);
        vector<int> pos=J.pos(this->Jset.from_int());
        IMatrix<T,Index>::Jset=J;
        MatDense<T>::setCols(j_set.size(), pos, IMatrix<T,Index>::matFun());
        return pos;
    }
};

//---------------------------------------------------- Lazy matrix ---------------------------

template<class T,class...Indices> class MatLazy;

/// A lazy matrix is defined through a function f:(int,int)->T
/// It only computes/stores the requested values.
template<class T>
class MatLazy<T>: virtual public IMatrix<T> {
public:
    function<T(int,int)> f;

    MatLazy()=default;
    MatLazy(function<T(int,int)> f, int n_rows, int n_cols)
        : IMatrix<T>(n_rows,n_cols), f(f) {}

    vector<T> eval(vector<pair<int,int>> const& ids) const override
    {
        vector<T> values(ids.size());
        vector<size_t> pos_eval;
        for(auto c=0u; c<ids.size(); c++)
            if (auto it=data.find(ids[c]); it!=data.end())
                values[c]=it->second;
            else pos_eval.push_back(c);
        #pragma omp parallel for
        for(auto c:pos_eval)
            values[c]=f(ids[c].first, ids[c].second);
        for(auto c:pos_eval)
            data[ids[c]]=values[c];
        return values;
    }

    vector<T> submat(vector<int> const& rows, vector<int> const& cols) const override
    {
        vector<pair<int,int>> ids;
        ids.reserve(rows.size()*cols.size());
        for(auto j:cols)
            for(auto i:rows)
                ids.emplace_back(i,j);
        return eval(ids);
    }

    void forgetRow(int i0) const override
    {
        for(auto j=0u; j < this->n_cols; j++)
            data.erase({i0,j});
    }

    void forgetCol(int j0) const override
    {
        for(auto i=0u; i < this->n_rows; i++)
            data.erase({i,j0});
    }

    /// Increase the rows of the matrix to n_rows, while reordering its old rows according to P: row i -> row P[i].
    /// The new matrix function is required
    void setRows(int n_rows, vector<int> const& P, function<T(int,int)> fnew)
    {
        map<pair<int,int>,T> data2;
        for(auto [id,value]:data) {
            auto [i,j]=id;
            data2[{P[i],j}]=value;
        }
        data=data2;
        this->n_rows=n_rows;
        f=fnew;
    }

    /// Increase the cols of the matrix to n_cols, while reordering its old cols according to Q: col j -> col Q[j].
    /// The new matrix function is required
    void setCols(int n_cols, vector<int> const& Q, function<T(int,int)> fnew)
    {
        map<pair<int,int>,T> data2;
        for(auto [id,value]:data) {
            auto [i,j]=id;
            data2[{i,Q[j]}]=value;
        }
        data=data2;
        this->n_cols=n_cols;
        f=fnew;
    }

private:
    mutable map<pair<int,int>,T> data;
};

/// similar to MatLazy but the Index can be anything: i.e double, or MultiIndex
/// whenever the possible values are provided in Iset, Jset
template<class T, class Index>
class MatLazy<T,Index>: public IMatrix<T,Index>, public MatLazy<T> {
public:
    MatLazy()=default;

    MatLazy(function<T(Index,Index)> A_, vector<Index> const& Iset_, vector<Index> const& Jset_)
        : IMatrix<T>(Iset_.size(), Jset_.size())
        , IMatrix<T,Index>(A_, Iset_, Jset_)
        , MatLazy<T>(IMatrix<T,Index>::matFun(), IMatrix<T>::n_rows, IMatrix<T>::n_cols)
    {}

    vector<int> setRows(vector<Index> const& i_set) override
    {
        IndexSet<Index> I(i_set);
        vector<int> pos=I.pos(this->Iset.from_int());
        IMatrix<T,Index>::Iset=I;
        MatLazy<T>::setRows(i_set.size(), pos, IMatrix<T,Index>::matFun());
        return pos;
    }

    vector<int> setCols(vector<Index> const& j_set) override
    {
        IndexSet<Index> J(j_set);
        vector<int> pos=J.pos(this->Jset.from_int());
        IMatrix<T,Index>::Jset=J;
        MatLazy<T>::setCols(j_set.size(), pos, IMatrix<T,Index>::matFun());
        return pos;
    }
};



//--------------------------------------------------------- Factory functions for IMatrix ------------------

///@{ Factory functions for the matrix
template<class T>
std::unique_ptr<IMatrix<T>> make_IMatrix(function<T(int,int)> f, int n_rows, int n_cols, bool is_full)
{
    if (is_full)
        return std::make_unique<MatDense<T>>(f,n_rows,n_cols);
    else
        return std::make_unique<MatLazy<T>>(f,n_rows,n_cols);
}

template<class T, class Index>
std::unique_ptr<IMatrix<T,Index>> make_IMatrix(function<T(Index,Index)> f, vector<Index> const& Iset, vector<Index> const& Jset, bool is_full)
{
    if (is_full)
        return std::make_unique<MatDense<T,Index>>(f,Iset,Jset);
    else
        return std::make_unique<MatLazy<T,Index>>(f,Iset,Jset);
}

///@}



} // end namespace xfac

#endif // MATRIX_INTERFACE_H
