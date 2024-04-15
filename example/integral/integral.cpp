#include <iostream>
#include <vector>

#include "xfac/tensor/tensor_ci.h"
#include "xfac/grid.h"

using namespace std;
using namespace xfac;

using cmpx=complex<double>;

int main()
{
    long count=0;
    auto f=[&count](vector<double> xs) {
        count++;
        double x=0, y=0,c=0;
        for(auto xi:xs) {c++; x+=c*xi; y+=xi*xi/c;}
        double arg=1.0+(x+2*y+x*y)*M_PI;
        return cmpx(1+x+cos(arg),x*x+0.5*sin(arg));
    };
    int dim=5;
    auto [xi,wi]=grid::QuadratureGK15(0,1);
    auto ci=CTensorCI1<cmpx,double>(f, vector(dim,xi));
    cout<<"rank nEval LastSweepPivotError integral(f)\n"<<setprecision(12);
    for(int i=1;i<=120;i++)
    {
        ci.iterate();
        if (i%10==0)
            cout<<i<<" "<<count<<" "<<ci.pivotError.back()<<" "<<ci.get_TensorTrain().sum(vector(dim,wi))<<endl;
    }

    return 0;
}

