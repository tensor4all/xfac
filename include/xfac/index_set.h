#ifndef INDEX_SET_H
#define INDEX_SET_H

#include <algorithm>
#include <vector>
#include <map>
#include <string>

namespace xfac {

using std::vector;
using std::map;

inline vector<int> iota(int n)
{
    vector<int> all(n);
    for(int i=0;i<n;i++) all[i]=i;
    return all;
}

inline vector<int> set_diff(int n, vector<int> Iset)
{
    vector<int> all=iota(n);
    std::sort(Iset.begin(),Iset.end());
    vector<int> diff;
    diff.reserve(n-Iset.size());
    std::set_difference(all.begin(),all.end(),
                        Iset.begin(),Iset.end(),back_inserter(diff));
    return diff;
}

inline vector<int> take_n_random(vector<int> A, int n)
{
    vector<int> out;
    while (!A.empty() && out.size()<n) {
        int pos=rand()%A.size();
        out.push_back(A[pos]);
        A.erase(A.begin()+pos);
    }
    return out;
}

inline vector<int> inversePermutation(vector<int> const& Iset)
{
    vector<int> Ip(Iset.size());
    for(auto i=0u; i<Iset.size(); i++) Ip.at(Iset[i])=i;
    return Ip;
}


template<class Index=int>
class IndexSet {
    map<Index,int> to_int_data;
    vector<Index> from_int_data;
public:

    IndexSet(){}
    IndexSet(vector<Index> const& ids) { for(const Index& id:ids) push_back(id); }

    size_t size() const { return from_int_data.size(); }
    void clear() { to_int_data.clear(); from_int_data.clear(); }
    int pos(Index const& id) const { return to_int_data.at(id); }
    vector<int> pos(vector<Index> const& ids) const
    {
        vector<int> p;
        p.reserve(ids.size());
        for(Index const& id:ids)
            p.push_back(pos(id));
        return p;
    }
    const vector<Index>& from_int() const { return from_int_data; }
    operator const vector<Index>& () const { return from_int_data; }
    const map<Index,int>& to_int() const { return to_int_data; }
    const Index& at(int i) const { return from_int_data.at(i); }
    vector<Index> at(vector<int> const& ids) const
    {
        vector<Index> p;
        p.reserve(ids.size());
        for(int id:ids)
            p.push_back(at(id));
        return p;
    }
    void push_back(Index i)
    {
        if (to_int_data.find(i) != to_int_data.end()) return;  /// TODO: use emplace_hint()
        to_int_data.emplace(i,from_int_data.size());
        from_int_data.push_back(i);
    }
};

using MultiIndex=std::u32string ;
template<class T>
using MultiIndexG=std::vector<T> ;

inline MultiIndex& add_inplace(MultiIndex& a, MultiIndex const& b)
{
    if (a.size()!=b.size()) throw std::invalid_argument("kron of different sizes MultiIndex");
    for(auto i=0u; i<a.size(); i++)
        a[i]+=b[i];
    return a;
}

inline MultiIndex add(MultiIndex const& a, MultiIndex const& b) { MultiIndex c=a; return add_inplace(c,b); }

inline vector<int> multiIndex_as_vec(MultiIndex const& mi) { return vector<int> {mi.begin(), mi.end()}; }

inline vector<vector<int>> multiIndex_as_vec(vector<MultiIndex> const& Iset)
{
    vector<vector<int>> out;
    for(const auto& mi:Iset)
        out.push_back(multiIndex_as_vec(mi));
    return out;
}

inline vector<vector<vector<int>>> multiIndex_as_vec(vector<vector<MultiIndex>> const& Iset)
{
    vector<vector<vector<int>>> out;
    for(const auto& mi:Iset)
        out.push_back(multiIndex_as_vec(mi));
    return out;
}

inline vector<vector<vector<int>>> multiIndex_as_vec(vector<IndexSet<MultiIndex>> const& Iset)
{
    return multiIndex_as_vec(vector<vector<MultiIndex>> {Iset.begin(), Iset.end()}) ;
}

inline vector<MultiIndex> kron( vector<MultiIndex> const& I1,vector<MultiIndex> const& I2)
{
    vector<MultiIndex> R;
    for(const auto& s2:I2)
        for(const auto& s1:I1)
            R.push_back(s1+s2);
    return R;
}

inline vector<MultiIndex> add( vector<MultiIndex> const& I1,vector<MultiIndex> const& I2)
{
    vector<MultiIndex> R;
    for(const auto& s2:I2)
        for(const auto& s1:I1)
            R.push_back(add(s1,s2));
    return R;
}

inline vector<MultiIndex> set_intersection( vector<MultiIndex> I1,vector<MultiIndex> I2)
{
    std::sort(I1.begin(), I1.end());
    std::sort(I2.begin(), I2.end());
    vector<MultiIndex> R;
    std::set_intersection(I1.begin(), I1.end(), I2.begin(), I2.end(), std::back_inserter(R));
    return R;
}

inline vector<MultiIndex> set_union( vector<MultiIndex> I1,vector<MultiIndex> I2)
{
    std::sort(I1.begin(), I1.end());
    std::sort(I2.begin(), I2.end());
    vector<MultiIndex> R;
    std::set_union(I1.begin(), I1.end(), I2.begin(), I2.end(), std::back_inserter(R));
    return R;
}

template<class T>
MultiIndexG<T> to_MultiIndexG(MultiIndex const& id, vector<vector<T>> const& xs)
{
    MultiIndexG<T> out(id.size(),0);
    for(auto i=0u;i<id.size();i++) out[i]=xs[i][id[i]];
    return out;
}

template<class T>
inline vector<MultiIndexG<T>> to_MultiIndexG(vector<MultiIndex> const& Iset, vector<vector<T>>  const& xs)
{
    vector<MultiIndexG<T>> out;
    for(MultiIndex const& id:Iset) out.push_back(to_MultiIndexG(id,xs));
    return out;
}

/// converts i to tensor index. The tensor dimensions are dims
inline vector<int> to_tensorIndex(std::size_t i, vector<int> const& dims)
{
    vector<int> idv;
    idv.reserve(dims.size());
    for(size_t k=0; k<dims.size(); k++)
    {
        idv.push_back( i%dims[k] );
        i/=dims[k];
    }
    return idv;
}

/// converts tensor index to absolute index. The tensor dimensions are dims
inline std::size_t to_absIndex(vector<int> const& idv, vector<int> const& dims)
{
    std::size_t sum=0, prod=1;;
    for(size_t k=0; k<dims.size(); k++)
    {
        sum+=idv[k]*prod;
        prod*=dims[k];
    }
    return sum;
}


}// end namespace xfac

#endif // INDEX_SET_H
