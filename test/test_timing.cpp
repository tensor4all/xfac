#include<catch2/catch.hpp>

#include<iostream>
#include<bitset>
#include "xfac/grid.h"
#include "xfac/tensor/tensor_ci.h"
#include <ctime>

using namespace std;
using namespace xfac;

TEST_CASE("timing")
{
    long count=0;
    auto f=[&count](const vector<double>& xs) {
        count++;
        double x=0, y=0,c=0;
        for(auto xi:xs) {c++; x+=c*xi; y+=xi*xi/c;}
        double arg=1.0+(x+2*y+x*y)*M_PI;
        return 1+x+cos(arg) + x*x+0.5*sin(arg);
    };

    auto xi=grid::QuadratureGK15(0,1).first;
    size_t dim=10;
    auto ci=CTensorCI1<double,double>(f,vector(dim,xi));
    vector<double> ti;
    time_t  t0=time(nullptr);
    for(auto i=1; i<21/*150*/; i++) {
        ci.iterate();
        ti.push_back(difftime(time(nullptr),t0));
        if (i % 10 ==0)
            cout<<i<<" "<<ci.pivotError.back()<<" "<<ti.back()<<endl;
    }

}
