#ifndef PIVOT_FINDER_H
#define PIVOT_FINDER_H

#include"cross_data.h"
#include"matrix_interface.h"

namespace xfac {



//------------------------------------- Functor implementing a PivotFinder ----------

struct PivotData {int i,j; double error;};

struct PivotFinderParam {
    bool fullPiv=false;
    int nRookStart=1, nRookIter=5; // for alternate search
    std::function<bool(int,int)> fBool; // fBool(i,j)=0 when i,j should not be a pivot
    arma::vec weightRow, weightCol;
};

template<class T>
struct PivotFinder {

    PivotFinderParam p;

    PivotFinder(PivotFinderParam p={}): p(std::move(p)) {}

    /// propose a pivot trying to maximize the local error of the cross interpolation ci with respect to A.
    PivotData operator()(IMatrix<T> const& A, CrossData<T> const& ci) const
    {
        auto I0=ci.availRows();
        auto J0=ci.availCols();
        PivotData pivotBest={-1, -1, 0};
        if (I0.empty() || J0.empty()) return pivotBest;
        if (p.fullPiv) return findIn(A, ci, ci.availRows(), ci.availCols());

        // otherwise do the alternate search
        vector<int> Jsample=availColsSample(ci);
        if (Jsample.empty()) return pivotBest;

        for(int t=0;t<p.nRookStart;t++)
        {
            PivotData pivot = {-1, Jsample[rand()%Jsample.size()], -1};
            for(int k=0;k<p.nRookIter;k++)
            {
                int i0=pivot.i;
                pivot=findIn(A, ci, I0, vector({pivot.j})); // col search
                if (i0==pivot.i) break; // rook condition
                int j0=pivot.j;
                pivot=findIn(A, ci, vector({pivot.i}), J0); // row search
                if (j0==pivot.j) break; // rook condition
            }
            if (t==0 || pivot.error>pivotBest.error) pivotBest=pivot;
        }
        return pivotBest;
    }

    /// compute the local error of the cross interpolation ci with respect to A at (i,j).
    double error(IMatrix<T> const& A, CrossData<T> const& ci, int i, int j) const
    {
        return localError(i,j,A.submat({i},{j})[0],ci.eval(i,j));
    }

private:

    double localError(int i,int j, T const& Aij,T const& Aij_approx) const
    {
        T err=Aij_approx-Aij;
        if (p.weightRow.empty() || p.weightCol.empty()) return std::abs(err);
        return std::abs(err* p.weightRow[i]* p.weightCol[j]);
    }

    /// Keeps the indices that satisfy the condition fBool (if any).
    vector<pair<int,int>> filter(vector<int> const& I0, vector<int> const& J0) const
    {
        vector<pair<int,int>> ids;
        for(auto i:I0)
            for(auto j:J0)
                if(!p.fBool || p.fBool(i,j))
                    ids.push_back({i,j});
        return ids;
    }

    /// Find the pivot that maximize the localError of the cross interpolation ci in the submatrix A(I0,J0).
    PivotData findIn(IMatrix<T> const& A, CrossData<T> const& ci, vector<int> const& I0, vector<int> const& J0) const
    {
        vector<pair<int,int>> ids=filter(I0, J0);
        vector<T> data_fu=A.eval(ids);
        vector<T> data_ci=ci.eval(ids);
        PivotData pivot={-1,-1,-1.0};
        for(auto c=0u;c<data_fu.size();c++)
        {
            auto [i,j]=ids[c];
            double err=localError(i, j, data_fu[c], data_ci[c]);
            if (c==0 || err>pivot.error) pivot={i,j,err};
        }
        return pivot;
    }

    /// generate by sampling a list of possible columns
    vector<int> availColsSample(CrossData<T> const& ci) const
    {
        if (!p.fBool) return ci.availCols();
        vector<int> J0;
        for(auto i:ci.availRows())
            for(auto j:ci.availCols())
                if (p.fBool(i,j)) J0.push_back(j);
        return J0;
    }
};

}// end namepsace xfac

#endif // PIVOT_FINDER_H
