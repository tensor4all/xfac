#ifndef GRID_H
#define GRID_H

#include <cassert>
#include <math.h>
#include<vector>
#include<string>

#include<iomanip>
#include<bitset>

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

inline auto QuadratureGK15(double a=0, double b=1)
{
    // can be obtained at boost::math::quadrature::gauss_kronrod<double, 15>
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

struct Quantics {
    double a=0, b=1;
    int nBit=10;
    int dim=1;
    bool pack=false;

    double deltaX;
    double deltaVolume;
    int tensorLen;
    int tensorLocDim;

    Quantics(double a_=0, double b_=1, int nBit_=10, int dim_=1, bool pack_=false)
        : a(a_)
        , b(b_)
        , nBit(nBit_)
        , dim(dim_)
        , pack(pack_)
        , deltaX( (b-a)/(1ull<<nBit) )
        , deltaVolume( pow(deltaX,dim) )
        , tensorLen( pack ? nBit : nBit*dim )
        , tensorLocDim( pack ? 1<<dim : 2 )
    { assert(nBit<64); }


    vector<int> tensorDims() const { return vector(tensorLen, tensorLocDim); }

    vector<int> coord_to_id(vector<double> const& us) const
    {
        assert(dim==us.size());
        vector<int> id(tensorLen,0);
        for(auto i=0u; i<us.size(); i++) {
            std::bitset<64> bi=(us[i]-a)/deltaX;
            for(auto d=0; d<nBit; d++)
                if (pack) id[d] |= (bi[d]<<i);
                else id[i+d*dim]=bi[d];
        }
        return id;
    }

    vector<double> id_to_coord(vector<int> const& id) const
    {
        assert(tensorLen==id.size());
        vector<double> us(dim);
        for(auto i=0; i<dim; i++) {
            std::bitset<64> bi;
            for(auto d=0; d<nBit; d++)
                if (pack) bi[d]=id[d] & (1<<i);
                else bi[d]=id[i+d*dim];
            us[i]=a+deltaX*bi.to_ullong();
        }
        return us;
    }

    void save(std::ostream &out) const { out<<a<<" "<<b<<" "<<nBit<<" "<<dim<<" "<<pack<<std::endl; }

    static Quantics load(std::istream& in)
    {
        Quantics g0;
        in>>g0.a>>g0.b>>g0.nBit>>g0.dim>>g0.pack;
        return Quantics(g0.a,g0.b,g0.nBit,g0.dim,g0.pack);
    }

};

} // end namespace grid
} // end namespace xfac

#endif // GRID_H
