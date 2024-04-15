#ifndef CROSS_DATA_H
#define CROSS_DATA_H

#include<memory>
#define ARMA_DONT_USE_OPENMP
#include<armadillo>

#include "matrix_interface.h"
#include "xfac/index_set.h"
#include "xfac/matrix/adaptive_lu.h"

namespace xfac {

using std::vector;

//------------------------------------------------------- stable product by inverse matrix -------

/// compute matrix A*B^-1 in a stable way
template<class T>
arma::Mat<T> mat_AB1(arma::Mat<T> const& A, arma::Mat<T> const& B)
{
    arma::Mat<T> AB=join_cols(A,B), Q,R;

    arma::qr_econ(Q, R, AB);
    arma::Mat<T> Qa=Q.rows(0,A.n_rows-1);
    arma::Mat<T> Qb=Q.rows(A.n_rows,Q.n_rows-1);
    return Qa*inv(Qb);
}

/// compute matrix A^-1*B in a stable way
template<class T>
arma::Mat<T> mat_A1B(arma::Mat<T> const& A, arma::Mat<T> const& B)
{
    arma::Mat<T> BA=join_cols(B.st(),A.st()), Q,R;
    arma::qr_econ(Q, R, BA);
    arma::Mat<T> Qb=Q.rows(0,B.n_cols-1);
    arma::Mat<T> Qa=Q.rows(B.n_cols,Q.n_rows-1);
    return inv(Qa).st()*Qb.st();
}


//-------------------------------------------------------- CrossData class ----------------------------

/// This class store a cross data from a big matrix
/// and can compute the cross interpolation associated to this data:
/// Aapprox=C*P^-1*R
template<class T>
class CrossData {
public:
    arma::Mat<T> C, R; ///< column and row submatrices

    AdaptiveLU<T> lu;

    /// Constructs an empty cross data
    CrossData(size_t n_rows_, size_t n_cols_): lu(n_rows_,n_cols_) {}

    /// Constructor from a list of rows/columns indices (I/J), and the corresponding submatrices: C=A(:,J), R=A(I,:) where A is the big matrix.
    template<class Container>
    CrossData(Container const& I, Container const& J,
              arma::Mat<T> const& C_, arma::Mat<T> const& R_)
        : C(C_), R(R_), lu(C_.n_rows, R.n_cols)
    {
        for(auto k=0u; k<I.size(); k++) {
            lu.addPivotRow(I[k], R_.row(k));
            lu.addPivotCol(J[k], C_.col(k));
        }
    }

    arma::Mat<T> pivotMat() const { return C.rows(arma::conv_to<arma::uvec>::from(lu.Iset)); }
    const arma::Mat<T>& leftMat() const { return cache.LD.empty() ? cache.LD=lu.L*arma::diagmat(lu.D) : cache.LD; }
    const arma::Mat<T>& rightMat() const { return lu.U; }
    const vector<int>& availRows() const { return cache.I_avail.empty() ? cache.I_avail=set_diff(lu.n_rows, lu.Iset) : cache.I_avail; }
    const vector<int>& availCols() const { return cache.J_avail.empty() ? cache.J_avail=set_diff(lu.n_cols, lu.Jset) : cache.J_avail; }

    size_t rank() const { return lu.npivot(); }
    T firstPivotValue() const { return C.empty() ? 1.0 : C(lu.Iset.at(0),0); }

    ///@{compute the cross formula at given rows/columns
    T eval(int i,int j) const { return C.empty() ? 0 : arma::dot(leftMat().row(i), rightMat().col(j)); }
    vector<T> eval(vector<pair<int,int>> const& ids) const
    {
        vector<T> values(ids.size());
        if (C.empty()) return values;
        for(auto c=0u;c<values.size();c++) {
            auto [i,j]=ids[c];
            values[c]=eval(i,j);
        }
        return values;
    }
    vector<T> row(int i) const { return arma::conv_to<vector<T>>::from( leftMat().row(i)*rightMat() ); }
    vector<T> col(int j) const { return arma::conv_to<vector<T>>::from( leftMat()*rightMat().col(j) ); }
    vector<T> submat(vector<int> const& rows, vector<int> const& cols) const
    {
        if (C.empty()) return vector<T>(rows.size()*cols.size(),0);
        auto I0=arma::conv_to<arma::uvec>::from(rows);
        auto J0=arma::conv_to<arma::uvec>::from(cols);
        return arma::conv_to<vector<T>>::from( leftMat().rows(I0)*rightMat().cols(J0) );
    }    
    arma::Mat<T> mat() const { return leftMat()*rightMat(); }

    /// return the update after adding last pivot (rank-1 update)
    arma::Mat<T> matDiff() const { return lu.L.tail_cols(1) * lu.U.tail_rows(1) * lu.D.back(); }
    ///@}

    /// Update the cross data with a new pivot at row i, column j of the matrix A.
    void addPivot(int i, int j, IMatrix<T> const& A)
    {
        addPivotRow(i,A);
        addPivotCol(j,A);
    }

    /// Update the cross by adding the row i of the matrix A. This is for developers only.
    void addPivotRow(int i, IMatrix<T> const& A)
    {
        arma::Mat<T> row(1,A.n_cols);
        for(auto j=0u;j<lu.Jset.size(); j++) // copy the data from C.row(i)
            row[lu.Jset[j]]=C(i,j);
        auto Ri=A.submat({i}, availCols()); // get the rest from A.row(i)
        for(auto j=0u; j<availCols().size(); j++)
            row[availCols()[j]]=Ri[j];
        R=arma::join_cols(R,row);
        lu.addPivotRow(i, row);
        cache={};
    }

    /// Update the cross by adding the col j of the matrix A. This is for developers only.
    void addPivotCol(int j, IMatrix<T> const& A)
    {
        arma::Mat<T> col(A.n_rows,1);
        for(auto i=0u;i<lu.Iset.size(); i++) // copy the data from R.col(j)
            col[lu.Iset[i]]=R(i,j);
        auto Cj=A.submat(availRows(),{j}); // get the rest from A.col(j)
        for(auto i=0u; i<availRows().size(); i++)
            col[availRows()[i]]=Cj[i];
        C=arma::join_rows(C, col);
        lu.addPivotCol(j, col);
        cache={};
    }

    /// Increase the rows of the matrix according to C_,
    /// while reordering its old rows according to P: row i -> row P[i].
    void setRows(arma::Mat<T> const& C_, vector<int> const& P)
    {
        C=C_;
        lu.setRows(C_,P);
        cache={};
    }

    /// Increase the cols of the matrix  according to R_,
    /// while reordering its old cols according to Q: col j -> col Q[j].
    void setCols(arma::Mat<T> const& R_, vector<int> const& Q)
    {
        R=R_;
        lu.setCols(R_,Q);
        cache={};
    }

private:
    struct Cache {
        vector<int> I_avail, J_avail;
        arma::Mat<T> LD;
    };
    mutable Cache cache;
};


}// end namespace xfac

#endif // CROSS_DATA_H
