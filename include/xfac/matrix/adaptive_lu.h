#ifndef ADAPTIVE_LU_H
#define ADAPTIVE_LU_H

#include<memory>
#define ARMA_DONT_USE_OPENMP
#include<armadillo>

namespace xfac {

using std::vector;

/**
 * @brief The AdaptiveLU class builds a rank-r LU decomposition of a matrix M by adding pivots 1 by 1
 *  The decomposition reads
 *  $ M ~ L matdiag(D) U
 *  Interestingly, both the rows  and columns of M can be increased and permuted at any time.
 *  All the operation scales as O(r^2).
 */
template<class T>
struct AdaptiveLU {

    size_t n_rows, n_cols;
    vector<int> Iset, Jset; ///< selected rows and columns

    arma::Mat<T> L, U;
    arma::Col<T> D;

    AdaptiveLU(size_t n_rows_, size_t n_cols_): n_rows(n_rows_), n_cols(n_cols_){}

    size_t npivot() const { return D.size(); }

    /// Gaussian elimination with pivoting on rows
    void addPivotRow(int i, arma::Row<T> const& row)
    {
        Iset.push_back(i);
        auto k=npivot();
        U = arma::join_cols(U, row);
        for(auto l=0u; l<k; l++)
            U.row(k) -= U.row(l) * ( L(Iset[k],l) * D(l) );
    }

    /// Gaussian elimination with pivoting on columns
    void addPivotCol(int j, arma::Col<T> const& col)
    {
        Jset.push_back(j);
        auto k=npivot();
        L = arma::join_rows(L,col);
        for(auto l=0u;l<k;l++)
            L.col(k) -= L.col(l) * ( U(l,Jset[k]) * D(l) );

        D.resize(L.n_cols);
        D(k)=1.0/L(Iset[k],k);
    }

    /// Increase the rows of the matrix according to C_,
    /// while reordering its old rows according to P: row i -> row P[i].
    void setRows(arma::Mat<T> const& C, vector<int> const& P)
    {
        n_rows=C.n_rows;
        for(auto& i : Iset) i=P.at(i);
        auto r=L.n_cols;
        L=[&](){
            arma::Mat<T> uMat2(C.n_rows, r, arma::fill::none);
            uMat2.rows(arma::conv_to<arma::uvec>::from(P))=L;
            return uMat2;
        }();
        auto Pc=arma::conv_to<arma::uvec>::from(set_diff(C.n_rows,P));
        for(auto k=0u; k<r; k++) {
            L(Pc, arma::uvec({k}))=C(Pc, arma::uvec({k}));
            for(auto l=0u;l<k;l++)
                L(Pc, arma::uvec({k})) -= L(Pc, arma::uvec({l})) * ( U(l,Jset[k]) * D(l) );
        }
    }

    /// Increase the cols of the matrix  according to R_,
    /// while reordering its old cols according to Q: col j -> col Q[j].
    void setCols(arma::Mat<T> const& R,vector<int> const& Q)
    {
        n_cols=R.n_cols;
        for(auto& j : Jset) j=Q.at(j);
        auto r=U.n_rows;
        U=[&](){
            arma::Mat<T> vMat2(r, R.n_cols, arma::fill::none);
            vMat2.cols(arma::conv_to<arma::uvec>::from(Q))=U;
            return vMat2;
        }();
        auto Qc=arma::conv_to<arma::uvec>::from(set_diff(R.n_cols,Q));
        for(auto k=0u; k<r; k++) {
            U(arma::uvec({k}), Qc)=R(arma::uvec({k}), Qc);
            for(auto l=0u;l<k;l++)
                U(arma::uvec({k}), Qc) -= U(arma::uvec({l}), Qc) * ( L(Iset[k],l) * D(l) );
        }
    }

};

}



#endif // ADAPTIVE_LU_H
