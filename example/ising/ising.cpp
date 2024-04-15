#include <iostream>
#include <vector>

#include "xfac/tensor/tensor_ci.h"

using namespace std;
using namespace xfac;


void TestIsingProblem(int L=10, int S=1,int rankMax=30)
{
    long c=0;
    auto energy=[=](vector<int> const& id) {
        double sum=0;
        for(size_t i=0;i<id.size();i++)
             for(size_t j=0;j<id.size();j++)
                sum+= (id[i]-S)*(id[j]-S)/((i-j)*(i-j)+1.0) ;
        return sum;
    };
    auto myTf=[=,&c](vector<int> const& id) { c++; return exp(-energy(id)); };

    auto ci=TensorCI1<double>(myTf, vector(L,2*S+1), {.fullPiv=true});
    auto Z=vector(L, vector(2*S+1,1.0)); //normal sum
    cout<<"rank neval pivotError sum trueError\n";
    for(int i=1;i<=rankMax;i++)
    {
        ci.iterate();
        double z= ci.get_TensorTrain().sum(Z);
        cout<<i<<" "<<c<<" "<<ci.pivotError.back()<<" "<<z<<" "<<ci.trueError()<<endl;
    }
}

int main()
{
    TestIsingProblem();
    return 0;
}

