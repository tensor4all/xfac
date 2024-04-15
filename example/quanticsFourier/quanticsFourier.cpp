#include <iomanip>
#include <iostream>
#include <vector>
#include <bitset>

#include "xfac/tensor/tensor_ci_2.h"

using namespace std;
using namespace xfac;

using cmpx=complex<double>;


void TestQuanticsFourier(int dim)
{
    if (dim>=64) throw invalid_argument("this implementation is for dim<64");
    long count=0;
    size_t n_point=1ul<<dim;
    auto fq=[&](const vector<int>& id)
    {
        size_t sum=0;       // compute x*y where x is formed by the bits id[k]/2, reverted, and y is formed by the bits id[j]%2
        for(auto k=0; k<dim; k++) {
            if (id[k]/2==0) continue;
            for(auto j=0; j<dim; j++) {
                if (id[j]%2==0) continue;
                sum += 1ul << (dim-1-k+j);  // reverted ---> dim-1-k
                sum %= n_point;             // actually we don't need x*y but x*y % n_point
            }
        }
        count++;
        return exp(cmpx(0,-2*M_PI*sum/n_point));
    };

    auto ci=TensorCI2<cmpx>(fq, vector(dim,4), {.bondDim=50, .fullPiv=true});
    for(auto v : {1,2,3})
        ci.addPivotsAllBonds({vector(dim,v)});
    cout<<"\nrank nEval LastSweepPivotError\n"<<setprecision(12);
    for(auto i=0u; !ci.isDone(); i++) {
        ci.iterate();
        cout<<ci.pivotError.size()-1<<" "<<count<<" "<<ci.pivotError.back()<<endl;
    }
    cout<<"\nrank pivotError\n"<<setprecision(12);
    for(auto i=0u;i<ci.pivotError.size();i++)
         cout<<i+1<<" "<<ci.pivotError[i]<<endl;
}

int main()
{
    TestQuanticsFourier(30);
    return 0;
}

