#include <iomanip>
#include <iostream>
#include <vector>
#include <bitset>

#include "xfac/tensor/tensor_ci_2.h"
#include "xfac/grid.h"

using namespace std;
using namespace xfac;

using cmpx=complex<double>;


void TestQuantics2d(function<double(double,double)> fun, double a, double b)
{
    auto f=[&](vector<double> xy) { return fun(xy[0],xy[1]); };
    auto ci=QTensorCI<double>(f, grid::Quantics {a,b,30,2}, {.bondDim=100});
    cout<<"\nrank nEval LastSweepPivotError integral(f)\n"<<setprecision(12);
    for(int i=1;i<=100;i++)
    {
        ci.iterate();
        cout<<ci.pivotError.size()-1<<" "<<ci.f.nEval()<<" "<<ci.pivotError.back()<<" "<<ci.get_qtt().integral()<<endl;
        if (ci.isDone()) break;
    }

    cout<<"\nrank pivotError\n"<<setprecision(12);
    for(auto i=0u;i<ci.pivotError.size();i++)
         cout<<i+1<<" "<<ci.pivotError[i]<<endl;
}


int main()
{
    function f1=[&](double x,double y) { return cos(2*M_PI*(x+2*y))/(1+x*x+y*y); };
    TestQuantics2d(f1,0,100);
    return 0;
}

