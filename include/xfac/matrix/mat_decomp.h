#ifndef MAT_DECOMP_H
#define MAT_DECOMP_H

#include "xfac/index_set.h"
#include<memory>
#define ARMA_DONT_USE_OPENMP
#include<armadillo>

namespace xfac {

using std::vector;

template<class Decomp>
struct MatDecompFixedTol
{
    using T=typename Decomp::value_type;

    double tol;
    int rankMax;
    MatDecompFixedTol(double tol_=1e-14, int rankMax_=0) : tol(tol_), rankMax(rankMax_) {}

    std::array<arma::Mat<T>,2> operator()(arma::Mat<T> const& M, bool leftOrthogonal)
    {
        Decomp s {M, leftOrthogonal, tol, rankMax};
        return { s.left(), s.right() };
    }
};

//--------------------------------------- QR decomposition -----------------------

template<class T>
struct MatQR {
    std::array<arma::Mat<T>,2> operator()(arma::Mat<T> const& A, bool leftOrthogonal) { return leftOrthogonal ? mat_qr(A) : mat_qr_t(A); }

private:
    static std::array<arma::Mat<T>,2> mat_qr(arma::Mat<T> const& A)
    {
        arma::Mat<T> Q,R;
        arma::qr_econ(Q, R, A);
        return {Q, R};
    }
    static std::array<arma::Mat<T>,2> mat_qr_t(arma::Mat<T> const& A)
    {
        arma::Mat<T> Q,R;
        arma::qr_econ(Q, R, A.t());
        return {R.t(), Q.t()};
    }
};


//--------------------------------------- SVD decomposition -----------------------

template<class T>
struct SVDDecomp {
    using value_type=T;

    arma::Mat<T> U,V;
    arma::vec s;
    bool leftOrthogonal=true;

    SVDDecomp(arma::Mat<T> const& M, bool leftOrthogonal_=true, double reltol=1e-12, int rankMax=0)
        : leftOrthogonal(leftOrthogonal_)
    {
        arma::svd_econ(U,s,V,M,"both","std");
        int n=findnValues(s, reltol);
        if (rankMax>0 && rankMax<n) n=rankMax;
        s.resize(n);
        U.resize(U.n_rows,n);
        V.resize(V.n_rows,n);
//        if (arma::norm(U*arma::diagmat(s)*V.t()-M) > reltol*arma::norm(M))
//            std::cerr<<" problem with my SVD!!! \n";
    }

    arma::Mat<T> left() const { if (leftOrthogonal) return U; return U*arma::diagmat(s); }
    arma::Mat<T> right() const { if (leftOrthogonal) return arma::diagmat(s)*V.t(); return V.t(); }

private:
    static int findnValues(arma::vec const& s, double reltol)
    {
        double tol2=reltol*reltol;
        double norm2=pow(arma::norm(s,2),2);
        double sum=0;
        int n=s.size();
        for(int i=s.size()-1; i>=0; i--) {
            sum += s[i]*s[i];
            if (sum>tol2*norm2) { n=i+1; break; }
        }
        return n;
    }
};

template<class T>
struct MatSVDFixedTol: public MatDecompFixedTol<SVDDecomp<T>> {
    using MatDecompFixedTol<SVDDecomp<T>>::MatDecompFixedTol;
};

//--------------------------------------- rank-revealing LU decomposition -----------------------

template<class T>
struct RRLUDecomp {
    using value_type=T;

    vector<int> Iset, Jset; ///< permutation of rows and columns
    arma::Mat<T> L, U;
    bool leftOrthogonal=true;
    int npivot=0;
    double error=0;

    RRLUDecomp()=default;

    RRLUDecomp(arma::Mat<T> const& A, bool leftOrthogonal_=true, double reltol=0, int rankMax=0)
        : leftOrthogonal(leftOrthogonal_)
        , Iset (iota(A.n_rows))
        , Jset (iota(A.n_cols))
    { calculate(A, reltol, rankMax); }

    void calculate(arma::Mat<T> A, double reltol, int rankMax)
    {
        npivot= std::min(A.n_rows, A.n_cols);
        if (reltol==0) reltol=npivot*std::abs(std::numeric_limits<T>::epsilon());
        double max_error=0;
        if (rankMax>0 && rankMax<npivot) npivot=rankMax;
        for(auto k=0u; k<npivot; k++) {
            // find pivot
            auto p=arma::abs( A(arma::span(k,A.n_rows-1), arma::span(k,A.n_cols-1)) ).index_max();
            auto i0=p%(A.n_rows-k)+k;      // was relative to the corner (k,k)
            auto j0=p/(A.n_rows-k)+k;
            if (double err=std::abs(A(i0,j0));
                k>0 &&  err < reltol*max_error )  { npivot=k; break; }
            else max_error=std::max(max_error, err);
            // move it to position k,k
            std::swap(Iset[k],Iset[i0]);
            std::swap(Jset[k],Jset[j0]);
            A.swap_rows(k,i0);
            A.swap_cols(k,j0);
            // update the error, i.e. make Gaussian elimination
            auto rows=arma::span(k+1,A.n_rows-1);
            auto cols=arma::span(k+1,A.n_cols-1);
            if (k+1<A.n_rows && leftOrthogonal)
                A(rows,k) *= 1.0/A(k,k);
            else if (k+1<A.n_cols && !leftOrthogonal)
                A(k, cols) *= 1.0/A(k,k);
            if (k+1<npivot)
                A(rows,cols) -= A(rows,k)*A(k,cols) ;
        }
        if (npivot<std::min(A.n_rows, A.n_cols))
            error=arma::abs( A(arma::span(npivot,A.n_rows-1), arma::span(npivot,A.n_cols-1)) ).max();
        readLU(A);
    }

    vector<int> row_pivots() const { return {Iset.begin(), Iset.begin()+npivot}; }
    vector<int> col_pivots() const { return {Jset.begin(), Jset.begin()+npivot}; }
    arma::Mat<T> PivotMatrixTri() const
    {
        arma::Mat<T> P=U.submat(0,0,npivot-1,npivot-1);
        for(auto j=0u; j<P.n_cols; j++)
            for(size_t i=j+leftOrthogonal; i<P.n_rows; i++)
              P(i,j)=L(i,j);
        return P;
    }

    vector<double> pivotErrors() const
    {
        vector<double> out(npivot);
        auto diag=leftOrthogonal ? U.diag() : L.diag();
        for(int i=0; i<npivot; i++)
            out[i]=std::abs( diag(i));
        if (npivot<std::min(L.n_rows, U.n_cols)) out.push_back(error);
        return out;
    }

    /// return the matrix L with permuted rows. left()*right() gives the rank=npivot reconstructed matrix.
    arma::Mat<T> left() const  { return L.rows(arma::conv_to<arma::uvec>::from(inversePermutation(Iset))); }

    /// return the matrix U with permuted columns. left()*right() gives the rank=npivot reconstructed matrix.
    arma::Mat<T> right() const { return U.cols(arma::conv_to<arma::uvec>::from(inversePermutation(Jset))); }

protected:
    void readLU(arma::Mat<T> const& lu) {
        L=arma::Mat<T>(lu.n_rows, npivot, arma::fill::eye);
        for(auto j=0u; j<L.n_cols; j++)
            for(size_t i=j+leftOrthogonal; i<L.n_rows; i++)
              L(i,j)=lu(i,j);

        U=arma::Mat<T>(npivot, lu.n_cols, arma::fill::eye);
        for(auto j=0u; j<U.n_cols; j++) {
            auto m=std::min(j+leftOrthogonal, std::uint32_t(U.n_rows));
            for(size_t i=0u; i<m; i++)
              U(i,j)=lu(i,j);
        }
    }
};

/// Given the pivot matrix P already in LU form, update the C columns according to P.
template<class T>
void apply_LU_on_cols(arma::Mat<T>& C, arma::Mat<T> const& P, bool leftOrthogonal)
{
    if (P.n_rows != C.n_rows)
        throw std::invalid_argument("apply_LU_on_Cols: incompatible matrices");
    for(auto k=0u; k<P.n_rows; k++) {
        auto rows=arma::span(k+1,C.n_rows-1);
        if (!leftOrthogonal)
            C.row(k) *= 1.0/P(k,k);
        if (k+1<P.n_rows)
            C.rows(rows) -= P(rows,k)*C.row(k) ;
    }
}

/// Given the pivot matrix P already in LU form, update the R rows according to P.
template<class T>
void apply_LU_on_rows(arma::Mat<T>& R, arma::Mat<T> const& P, bool leftOrthogonal)
{
    if (P.n_cols != R.n_cols)
        throw std::invalid_argument("apply_LU_on_Rows: incompatible matrices");
    for(auto k=0u; k<P.n_rows; k++) {
        auto cols=arma::span(k+1,R.n_cols-1);
        if (leftOrthogonal)
            R.col(k) *= 1.0/P(k,k);
        if (k+1<P.n_rows)
            R.cols(cols) -= R.col(k)*P(k,cols) ;
    }
}

template<class T>
struct MatRRLUFixedTol: public MatDecompFixedTol<RRLUDecomp<T>> {
    using MatDecompFixedTol<RRLUDecomp<T>>::MatDecompFixedTol;
};


//--------------------------------------- alternate rank-revealing LU decomposition -----------------------

template<class T>
struct MatFun {
    size_t n_rows;
    size_t n_cols;
    std::function<arma::Mat<T>(vector<int>, vector<int>)> submat;
};

struct ARRLUParam {
    double reltol=0;
    int bondDim=0;
    bool fullPiv=false;
    int nRookIter=3;
};

template<class T>
struct ARRLUDecomp: public RRLUDecomp<T> {
    using RRLUDecomp<T>::Iset;
    using RRLUDecomp<T>::Jset;
    using RRLUDecomp<T>::npivot;
    using RRLUDecomp<T>::L;
    using RRLUDecomp<T>::U;
    using RRLUDecomp<T>::RRLUDecomp;

    ARRLUDecomp(MatFun<T> fA, vector<int> I0, vector<int> J0, bool leftOrthogonal_=true, ARRLUParam param={})
    {
        RRLUDecomp<T>::leftOrthogonal=leftOrthogonal_;
        RRLUDecomp<T>::Iset=iota(fA.n_rows);
        RRLUDecomp<T>::Jset=iota(fA.n_cols);

        if (param.fullPiv) { this->calculate(fA.submat(Iset,Jset), param.reltol, param.bondDim); return; }

        int rankMax=std::min(fA.n_rows,fA.n_cols);
        if (param.bondDim>0 && param.bondDim<rankMax) rankMax=param.bondDim;
        bool is_low_rank=false;
        do {
            // take new random rows or cols trying to duplicate the rank
            if (!leftOrthogonal_)
                for(auto x : take_n_random(set_diff(fA.n_rows, I0), std::max(1ul, I0.size())) ) I0.push_back(x);
            else
                for(auto x : take_n_random(set_diff(fA.n_cols, J0), std::max(1ul, J0.size())) ) J0.push_back(x);

            // iterate
            for(int k=0; k<param.nRookIter; k++)
            {
                arma::Mat<T> A= (k%2==leftOrthogonal_) ? fA.submat(I0, Jset) : fA.submat(Iset,J0);
                this->calculate(A, param.reltol, param.bondDim);
                auto I1= this->row_pivots();
                auto J1= this->col_pivots();
                if(I1.size()<std::min(A.n_rows,A.n_cols)) is_low_rank=true;
                if (I0==I1 && J0==J1) break; // rook condition
                I0=I1;
                J0=J1;
            }
        }  while(I0.size()<rankMax && !is_low_rank);

        if (!is_low_rank) this->error= std::abs((leftOrthogonal_ ? U.diag() : L.diag())(npivot-1));

        if (L.n_rows<fA.n_rows) { // complete the L
            auto L2=fA.submat(vector(Iset.begin()+npivot, Iset.end()),
                              vector(Jset.begin(), Jset.begin()+npivot));
            apply_LU_on_rows(L2, this->PivotMatrixTri(), leftOrthogonal_);
            L=arma::join_vert(L,L2);
        }
        if (this->U.n_cols<fA.n_cols) { // complete the U
            auto U2=fA.submat(vector(Iset.begin(), Iset.begin()+npivot),
                         vector(Jset.begin()+npivot, Jset.end()));
            apply_LU_on_cols(U2, this->PivotMatrixTri(), leftOrthogonal_);
            U=arma::join_horiz(U,U2);
        }
    }
};



//--------------------------------------- Cross interpolation or CUR decomposition -----------------------

template<class T>
struct CURDecomp: public ARRLUDecomp<T> {
    using value_type=T;

    using ARRLUDecomp<T>::Iset;
    using ARRLUDecomp<T>::Jset;
    using ARRLUDecomp<T>::L;
    using ARRLUDecomp<T>::U;
    using ARRLUDecomp<T>::npivot;
    using ARRLUDecomp<T>::leftOrthogonal;
    using ARRLUDecomp<T>::row_pivots;
    using ARRLUDecomp<T>::col_pivots;

    const arma::Mat<T> C, R;

    CURDecomp(arma::Mat<T> const& A_, bool leftOrthogonal_=true, double reltol=1e-14, int rankMax=0)
        : ARRLUDecomp<T>(A_,leftOrthogonal_,reltol,rankMax)
        , C(A_.cols(arma::conv_to<arma::uvec>::from(col_pivots())))
        , R(A_.rows(arma::conv_to<arma::uvec>::from(row_pivots())))
    {}

    CURDecomp(MatFun<T> fA, vector<int> const& I0, vector<int> const& J0, bool leftOrthogonal=true, ARRLUParam param={})
        : ARRLUDecomp<T>(fA, I0, J0, leftOrthogonal, param)
        , C(fA.submat(iota(fA.n_rows), col_pivots()))
        , R(fA.submat(row_pivots(), iota(fA.n_cols)))
    {}

    arma::Mat<T> left() const { return leftOrthogonal ? CU() : C ; }

    arma::Mat<T> right() const { return leftOrthogonal ? R : UR() ; }

    arma::Mat<T> CU() const
    {
        arma::Mat<T> cu(C.n_rows, npivot, arma::fill::eye);
        if (npivot<C.n_rows) {
            auto L1=L.head_rows(npivot);
            auto L2=L.tail_rows(C.n_rows-npivot);
            cu.tail_rows(C.n_rows-npivot)=arma::solve(arma::trimatu(L1.t()), L2.t()).t();
        }
        return cu.rows(arma::conv_to<arma::uvec>::from(inversePermutation(Iset)));
    }

    arma::Mat<T> UR() const
    {
        arma::Mat<T> ur(npivot, R.n_cols, arma::fill::eye);
        if (npivot<R.n_cols) {
            auto U1=U.head_cols(npivot);
            auto U2=U.tail_cols(R.n_cols-npivot);
            ur.tail_cols(R.n_cols-npivot)=arma::solve(arma::trimatu(U1), U2);
        }
        return ur.cols(arma::conv_to<arma::uvec>::from(inversePermutation(Jset)));
    }
};


/// Given the pivot matrix P already in LU form, compute UR according to P.
template<class T>
arma::Mat<T> compute_UR_on_cols(arma::Mat<T>  C, arma::Mat<T> const& P)
{
    apply_LU_on_cols(C,P,false);
    arma::Mat<T> Pu=P;
    Pu.diag().fill(1);
    return arma::solve(arma::trimatu(Pu), C);
}


/// Given the pivot matrix P already in LU form, update the R rows according to P.
template<class T>
arma::Mat<T> compute_CU_on_rows(arma::Mat<T> R, arma::Mat<T> const& P)
{
    apply_LU_on_rows(R,P,true);
    arma::Mat<T> Pu=P.t();
    Pu.diag().fill(1);
    return arma::solve(arma::trimatu(Pu), R.t()).t();
}

template<class T>
struct MatCURFixedTol: public MatDecompFixedTol<CURDecomp<T>> {
    using MatDecompFixedTol<CURDecomp<T>>::MatDecompFixedTol;
};



} //end namespace xfac


#endif // MAT_DECOMP_H
