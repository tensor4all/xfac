#ifndef TENSOR_TRAIN_H
#define TENSOR_TRAIN_H

#include "xfac/grid.h"
#include "xfac/index_set.h"
#include "xfac/matrix/mat_decomp.h"

#include<vector>
#include<array>
#define ARMA_DONT_USE_OPENMP
#include<armadillo>

namespace xfac {

using std::vector;
using std::function;
using std::array;


/// reshape a cube as a matrix B(i,jk)=A(i,j,k)
template<class eT>
arma::Mat<eT> cube_as_matrix1(arma::Cube<eT> const& A) { return arma::Mat<eT>(const_cast<eT*>(A.memptr()), A.n_rows, A.n_cols*A.n_slices, false); }

/// reshape a cube as a matrix B(ij,k)=A(i,j,k)
template<class eT>
arma::Mat<eT> cube_as_matrix2(arma::Cube<eT> const& A) { return arma::Mat<eT>(const_cast<eT*>(A.memptr()), A.n_rows*A.n_cols, A.n_slices, false); }


/// stores a tensor train, i.e., a list of cubes.
template<class T>
struct TensorTrain {
    vector< arma::Cube<T> >  M;   ///< list of 3-leg tensors

    TensorTrain()=default;
    TensorTrain(size_t len) : M(len) {}

    /// evaluate the tensor train at a given multi index.
    T eval(vector<int> const& id) const
    {
        if (id.size()!=M.size()) throw std::invalid_argument("TensorTrain::() id.size()!=size()");
        arma::Mat<T> prod(1,1,arma::fill::eye);
        for(auto k=0u; k<M.size(); k++)
            prod=prod* arma::Mat<T>{ M[k].col(id[k]) };
        return prod.eval()(0,0);
    }

    /// evaluate the tensor train at a given multi index. Same as eval()
    T operator()(vector<int> const& id) const { return eval(id); }

    /// compute the weighted sum of the tensor train
    T sum(const vector<vector<double>>& weight) const;

    /// compute the plane sum of the tensor train
    T sum1() const
    {
        vector<vector<double>> weight;
        for(auto const& Mi : M)
            weight.push_back(vector(Mi.n_cols,1.0));
        return sum(weight);
    }

    /// compute the overlap with another tensor train
    T overlap(const TensorTrain<T>& tt) const
    {
        if (M.empty() || tt.M.empty()) return 0;
        if (M.size() != tt.M.size())
            throw std::invalid_argument("tt1.overlap(tt2) with different lengths");
        arma::Mat<T> L(1,1, arma::fill::eye);
        for(auto p=0u; p<M.size(); p++) {  // L(A,B) = L(a,b)*N(a,s,A)*M(b,s,B)
            arma::Mat<T> LN=L.t()*cube_as_matrix1(tt.M.at(p));
            auto LNm=arma::Mat<T>(LN.memptr(), LN.n_rows*tt.M[p].n_cols, tt.M[p].n_slices, false);
            L=LNm.t()*cube_as_matrix2(M[p]);
        }
        return L(0,0);
    }

    T norm2() const { return overlap(*this); }

    void compressSVD(double reltol=1e-12, int maxBondDim=0) { right_to_left(MatQR<T> {}); sweep(MatSVDFixedTol<T> {reltol,maxBondDim}); }
    void compressLU(double reltol=1e-12, int maxBondDim=0)  { right_to_left(MatRRLUFixedTol<T> {}); sweep(MatRRLUFixedTol<T> {reltol, maxBondDim}); }
    void compressCI(double reltol=1e-12, int maxBondDim=0)  { right_to_left(MatCURFixedTol<T> {}); sweep(MatCURFixedTol<T> {reltol, maxBondDim}); }

    /// computes the max error |f-tt| if the tensor is smaller than max_n_Eval.
    double trueError(function<T(vector<int>)> f, size_t max_n_eval=1e6) const
    {
        size_t prod=1;
        vector<int> dims(M.size());
        for(size_t k=0; k<dims.size(); k++) {
            dims[k] = M[k].n_cols;
            prod *= dims[k];
            if (prod>max_n_eval) return -1;
        }
        double e=0;
        vector<int> x;
        for(size_t i=0;i<prod;i++) {
            auto idv=to_tensorIndex(i, dims);
            double error=std::abs(eval(idv)-f({idv.begin(),idv.end()}));
            if (error>e) {x=idv; e=error;}
        }
#ifndef NDEBUG
        for(auto xi:x) std::cout<<xi<<",";
        std::cout<<"-->";
#endif
        return e;
    }

    void save(std::ostream &out) const
    {
        out<<M.size()<<std::endl;
        for (const arma::Cube<T>& Mi:M) { Mi.save(out,arma::arma_ascii); out<<std::endl; }
    }
    void save(std::string fileName) const { std::ofstream out(fileName); save(out); }

    static TensorTrain<T> load(std::ifstream& in)
    {
        int L;
        in>>L;
        TensorTrain<T> tt;
        tt.M.resize(L);
        for(arma::Cube<T>& Mi:tt.M)
            Mi.load(in,arma::arma_ascii);
        return tt;
    }
    static TensorTrain<T> load(std::string fileName) { std::ifstream in(fileName); return load(in); }

    /// for developers
    void sweep(function<array<arma::Mat<T>,2>(arma::Mat<T>,bool)> mat_decomp)
    {
        left_to_right(mat_decomp);
        right_to_left(mat_decomp);
    }

    /// for developers
    void left_to_right(function<array<arma::Mat<T>,2>(arma::Mat<T>,bool)> mat_decomp)
    {
        for(auto i=0u; i+1<M.size(); i++) {
            auto ab=mat_decomp(cube_as_matrix2(M[i]), true);
            arma::Mat<T> &M1=ab[0];
            arma::Mat<T> M2=ab[1]* cube_as_matrix1(M[i+1]);
            M[i]=arma::Cube<T>(M1.memptr(), M[i].n_rows, M[i].n_cols, M1.n_cols);
            M[i+1]=arma::Cube<T>(M2.memptr(), M2.n_rows, M[i+1].n_cols, M[i+1].n_slices);
        }
    }

    /// for developers
    void right_to_left(function<array<arma::Mat<T>,2>(arma::Mat<T>,bool)> mat_decomp)
    {
        for(int i=M.size()-1; i>0; i--) {
            auto ab=mat_decomp(cube_as_matrix1(M[i]), false);
            arma::Mat<T> M1=cube_as_matrix2(M[i-1])*ab[0];
            arma::Mat<T> &M2=ab[1];
            M[i-1]=arma::Cube<T>(M1.memptr(), M[i-1].n_rows, M[i-1].n_cols, M1.n_cols);
            M[i]=arma::Cube<T>(M2.memptr(), M2.n_rows, M[i].n_cols, M[i].n_slices);
        }
    }
};

template<class T>
TensorTrain<T> operator+(TensorTrain<T> const& tt1, TensorTrain<T> const& tt2)
{
    if (tt1.M.empty()) return tt2;
    if (tt2.M.empty()) return tt1;
    if (tt1.M.size() != tt2.M.size())
        throw std::invalid_argument("tt1+tt2 with different lengths");
    auto kron_add=[&](arma::Cube<T> const& A, arma::Cube<T> const& B, int i)
    {
        if (A.n_cols != B.n_cols)
            throw std::invalid_argument("kron_add(A,B) with A.n_cols != B.n_cols");
        arma::Cube<T> C;
        if (i==0) {
            C=arma::Cube<T>(A.n_rows, A.n_cols, A.n_slices+B.n_slices, arma::fill::zeros);
            C(0,0,A.n_slices,arma::size(B))=B;
        }
        else if (i==tt1.M.size()-1) {
            C=arma::Cube<T>(A.n_rows+B.n_rows, A.n_cols, B.n_slices, arma::fill::zeros);
            C(A.n_rows,0,0,arma::size(B))=B;
        }
        else {
            C=arma::Cube<T>(A.n_rows+B.n_rows, A.n_cols, A.n_slices+B.n_slices, arma::fill::zeros);
            C(A.n_rows,0,A.n_slices,arma::size(B))=B;
        }
        C(0,0,0,arma::size(A))=A;        
        return C;
    };

    TensorTrain<T> tt;
    for(auto i=0; i<tt1.M.size(); i++) {
        tt.M.push_back( kron_add(tt1.M[i], tt2.M[i], i) );
    }
    return tt;
}

/// compute the sum of many tensor trains, while compressing along the tree.
template<class T>
TensorTrain<T> sum(vector<TensorTrain<T>> v,double reltol=1e-12, int maxBondDim=0, bool use_svd=false)
{
    if (v.empty()) return {};
    int step=1;
    while (v.size()>step)
    {
        #pragma omp parallel for
        for(int i=0;i<v.size();i+=2*step)
            if (i+step<v.size()) {
                v[i]=v[i]+v[i+step];
                if (use_svd) v[i].compressSVD(reltol, maxBondDim);
                else v[i].compressCI(reltol, maxBondDim);
            }
        step+=step;
    }
    return v.at(0);
}


/// stores a continuous tensor train, i.e., a list of 3-leg tensors where one of the legs is an Index type (i.e. a double).
template<class T, class Index>
struct CTensorTrain {
    vector< std::function<arma::Mat<T>(Index)> >  M;   ///< list of 3-leg tensors

    /// evaluate the c-tensor train at a given multi index.
    T eval(vector<Index> const& xs) const
    {
        if (xs.size()!=M.size()) throw std::invalid_argument("TensorTrain::() id.size()!=size()");
        arma::Mat<T> prod(1,1,arma::fill::eye);
        for(auto k=0u; k<M.size(); k++)
            prod=prod* M[k](xs[k]);
        return prod.eval()(0,0);
    }

    /// return the TensorTrain generated by sampling the CTensorTrain in a grid xi.
    TensorTrain<T> getTensorTrain(vector<vector<Index>> const&) const
    {
        return {}; // TODO
    }
};


/// store a quantics tensor train, ie a tensor train and a quantics grid. This struct is able to save/load a multidimensional function R^n -> T
template<class T>
struct QTensorTrain {
    TensorTrain<T> tt;
    grid::Quantics grid;

    /// evaluate the qtt at a given point in R^n
    T eval(vector<double> const& xs) const { return tt.eval(grid.coord_to_id(xs)); }

    /// compute the integral in the hypercube [a,b]^n where [a,b] are the bounds of the grid
    T integral() const { return tt.sum1()*grid.deltaVolume; }

    void save(std::ostream &out) const
    {
        tt.save(out);
        out<<std::endl;
        grid.save(out);
    }
    void save(std::string fileName) const { std::ofstream out(fileName); save(out); }

    static QTensorTrain<T> load(std::ifstream& in) { return {TensorTrain<T>::load(in), grid::Quantics::load(in)}; }
    static QTensorTrain<T> load(std::string fileName) { std::ifstream in(fileName); return load(in); }
};


/// Manage the weighted sum of a tensor train.
///             w0     w1     w2
///             |      |      |
///             M0 --- M1 --- M2
template<class T>
class TT_sum {
    vector< vector<double> > w; ///< the weights at each site
public:
    vector< arma::Row<T> > L;  ///< accumulated left product up to before a given site
    vector< arma::Col<T> > R;  ///< accumulated right product up to after a given site

    TT_sum(){}
    TT_sum(TensorTrain<T> const& tt, vector< vector<double> > const& weight)
        : w(weight), L(tt.M.size()), R(tt.M.size())
    {
        R.back()=arma::Col<T>(1,arma::fill::ones);
        L.front()=arma::Row<T>(1,arma::fill::ones);
        for(auto s=0u; s<tt.M.size()-1; s++)
            updateSite(s, tt.M[s], true);
        for(int s=tt.M.size()-1; s>0; s--)
            updateSite(s, tt.M[s], false);
    }

    TT_sum(TensorTrain<T> const& tt, vector<double> const& weight)
        : TT_sum(tt, vector(tt.M.size(), weight)) {}


    T value() const { return arma::dot(L[1], R[0]); }

    /// update the left or right product given the new cube M at site s.
    void updateSite(size_t s, arma::Cube<T> const& M, bool updateLeft) //TODO: this is invalidating the value()
    {
        if (!updateLeft && s>0) { // R[s-1]=M(i,j,k)*R[s](k)*w[s](j)
            arma::Mat<T> MRv=cube_as_matrix2(M)*R[s];
            auto MR=arma::Mat<T>(MRv.memptr(), M.n_rows, M.n_cols, false);
            auto ws=arma::colvec(w[s].data(), w[s].size(), false );
            R[s-1]=MR*ws;
        }
        if (updateLeft && s<L.size()-1) {// L[s+1]=w[s](j)*L[s](i)*M(i,j,k)
            arma::Mat<T> LMv=L[s]*cube_as_matrix1(M);
            auto LM=arma::Mat<T>(LMv.memptr(), M.n_cols, M.n_slices, false);
            auto ws=arma::rowvec(w[s].data(), w[s].size(), false );
            L[s+1]=ws*LM;
        }
    }
};

template<class T>
T TensorTrain<T>::sum(const vector<vector<double> > &weight) const { return TT_sum<T>(*this,weight).value(); }

}// end namespace xfac

#endif // TENSOR_TRAIN_H
