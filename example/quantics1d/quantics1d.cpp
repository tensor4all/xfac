#include <iomanip>
#include <iostream>
#include <vector>
#include <bitset>

#include "xfac/tensor/tensor_ci_2.h"
#include "xfac/grid.h"

using namespace std;
using namespace xfac;

void TestQuantics1d(function<double(double)> fun, double a, double b)
{
    auto ci=QTensorCI<double>(fun, grid::Quantics{a,b,30});
    cout<<"\nrank nEval pivotError integral(f)\n"<<setprecision(12);
    for(int i=1;i<=20;i++)
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
    function f1=[&](double x) { return cos(2*M_PI*x)/(1+x*x); };
    TestQuantics1d(f1,0,100);
    function f2=[&](double x) { return pow(x+0.1,5)+pow(x-0.5,4); };
    TestQuantics1d(f2,0,1);
    cout<<"exact integral = "<< pow(1.1,6)/6-pow(0.1,6)/6 + pow(0.5,5)/5+pow(0.5,5)/5 <<endl;

    return 0;
}

