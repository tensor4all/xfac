#ifndef GRID_H
#define GRID_H

#include <cassert>
#include <math.h>
#include<vector>
#include<string>

#include<iomanip>
#include<bitset>
#include <algorithm>

namespace xfac {
namespace grid {

using std::vector;
using std::pair;
using uint = unsigned int;

template <typename T>
std::vector<T> linspace(T a, T b, size_t n) {
    if (n==0) return {};
    if (n==1) return {a};
    T h = (b - a) / static_cast<T>(n-1);
    std::vector<T> xs(n);
    for(size_t i=0;i<n;i++)
        xs[i]=a+i*h;
    return xs;
}

template <typename T>
std::vector<T> logspace(T a, T b, size_t n) {
    auto xs=linspace<T>(log(a),log(b),n);
    for(auto& x:xs) x=exp(x);
    return xs;
}

/// Gauss Kronrod quadrature as defined at boost::math::quadrature::gauss_kronrod<double, 15>
inline auto QuadratureGK15(double a=0, double b=1)
{
    static const vector<double> abscissa={0, 0.2077849550078985, 0.4058451513773972, 0.5860872354676911, 0.7415311855993945, 0.8648644233597691, 0.9491079123427585, 0.9914553711208126};
    static const vector<double> weights={0.2094821410847278, 0.2044329400752989, 0.1903505780647854, 0.1690047266392679, 0.1406532597155259, 0.1047900103222502, 0.06309209262997856, 0.02293532201052922};
    int nq=2*abscissa.size()-1;
    std::vector<double> xi(nq), weight(nq);
    double factor=0.5*(b-a);
    for(uint i=0;i<abscissa.size();i++)
    {
        xi[nq/2+i]=factor*(abscissa[i]+1)+a;
        weight[nq/2+i]=weight[nq/2-i]=weights[i]*factor;
        xi[nq/2-i]=factor*(-abscissa[i]+1)+a;
    }
    return make_pair(xi,weight);
}


/// This class represents a quantics grid.
/// A quantics grid is an uniform grid with $2^R$ points in [a,b):
/// $x_i=a+i\Delta$,
/// where $\Delta = (b -a)/2^R $ and $i=0,1,...,2^R-1 $.
/// The function $f$ is mapped to a tensor with $R$ legs in the following way. The binary digits $\{\sigma_i\}$ of $i$ are used as indices of the tensor $F$ defined by:
/// $ F(\{\sigma_i\})=f(x_i) $
struct Quantics {
    double a=0, b=1;         ///< the start and end of the interval for all the variables
    int nBit=10;             ///< number of bit for each variable
    int dim=1;               ///< dimension of the hypercube
    bool fused=false;        ///< whether to combine the bits of the same "scale"

    double deltaX;
    double deltaVolume;
    int tensorLen;
    int tensorLocDim;

    Quantics(double a_=0, double b_=1, int nBit_=10, int dim_=1, bool fused_=false)
        : a(a_)
        , b(b_)
        , nBit(nBit_)
        , dim(dim_)
        , fused(fused_)
        , deltaX( (b-a)/(1ull<<nBit) )
        , deltaVolume( pow(deltaX,dim) )
        , tensorLen( fused ? nBit : nBit*dim )
        , tensorLocDim( fused ? 1<<dim : 2 )
    { assert(nBit<64); }


    vector<int> tensorDims() const { return vector(tensorLen, tensorLocDim); }

    vector<int> coord_to_id(vector<double> const& us) const
    {
        assert(dim==us.size());
        vector<int> id(tensorLen,0);
        for(auto i=0u; i<us.size(); i++) {
            std::bitset<64> bi=(us[i]-a)/deltaX;
            for(auto d=0; d<nBit; d++)
                if (fused) id[d] |= (bi[d]<<i);
                else id[i+d*dim]=bi[d];
        }
        return id;
    }

    vector<double> id_to_coord(vector<int> const& id) const
    {
        assert(tensorLen==id.size());
        #ifndef NDEBUG
        if (std::any_of(id.begin(), id.end(), [](int x) { return x < 0 || x > 1;}))
            throw std::invalid_argument("Quantics::id_to_coord(): bitvector contains invalid entries");
        #endif
        vector<double> us(dim);
        for(auto i=0; i<dim; i++) {
            std::bitset<64> bi;
            for(auto d=0; d<nBit; d++)
                if (fused) bi[d]=id[d] & (1<<i);
                else bi[d]=id[i+d*dim];
            us[i]=a+deltaX*bi.to_ullong();
        }
        return us;
    }

    void save(std::ostream &out) const { out<<std::setprecision(18)<<a<<" "<<b<<" "<<nBit<<" "<<dim<<" "<<fused<<std::endl; }

    static Quantics load(std::istream& in)
    {
        Quantics g0;
        in>>g0.a>>g0.b>>g0.nBit>>g0.dim>>g0.fused;
        return Quantics(g0.a,g0.b,g0.nBit,g0.dim,g0.fused);
    }

};

} // end namespace grid
} // end namespace xfac

#endif // GRID_H
