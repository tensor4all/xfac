#include<catch2/catch.hpp>

#include<iostream>
#include<bitset>
#include "xfac/grid.h"
#include "xfac/tensor/tensor_ci.h"

using namespace std;
using namespace xfac;

using cmpx=std::complex<double>;


TEST_CASE( "Test tensor CI" )
{
    SECTION( "cos" )
    {
        int dim=5, d=10;
        long count=0;
        function myf=[&](vector<int> id) {
            count++;
            auto sum=accumulate(id.begin(), id.end(), 0.0);
            return sum+cos(sum);
        };

        TensorCI1Param p;
        p.fullPiv=true;
        p.nIter=10*d;
        auto ci=TensorCI1<double>(myf, vector(dim,d), p);
        REQUIRE( ci.P[dim/2-1].n_rows==4 );
        if (p.fullPiv)
            REQUIRE( count < int(pow(d*ci.P[dim/2-1].n_rows,2)*(dim-1)) );
    }

    SECTION( "Function in Rn" )
    {
        long count=0;
        auto f=[&count](const vector<double>& xs) {
            count++;
            double x=0, y=0,c=0;
            for(auto xi:xs) {c++; x+=c*xi; y+=xi*xi/c;}
            double arg=1.0+(x+2*y+x*y)*M_PI;
            return cmpx(1+x+cos(arg),x*x+0.5*sin(arg));
        };

        auto [xi,wi]=grid::QuadratureGK15(0,1);
        size_t dim=5;
        SECTION( "discretized" )
        {
            auto myTf=[&,xi=xi](vector<int> const& id) {
                assert(id.size()==dim);
                vector<double> xs;
                for(auto i:id) xs.push_back(xi[i]);
                return f(xs);
            };
            TensorCI1Param p;
            p.weight=vector(dim,wi);  // activate the ENV learning
            auto ci=TensorCI1<cmpx>(myTf, vector(dim,(int)xi.size()));
            cout<<"rank nEval LastSweepPivotError\n";
            while(ci.cIter <= 120 && ci.pivotError.back()>1e-12)
            {
                ci.iterate();
                if (ci.cIter%10==0)
                    cout << ci.cIter << " " << count << " " << ci.pivotError.back() << endl;
            }
            REQUIRE( ci.pivotError.back()<5e-5 );
            vector<int> ids={3,5,1,5,1};
            auto tt=ci.get_TensorTrain(dim/2);
            REQUIRE( abs(tt.eval(ids)-myTf(ids))<1e-5 );
            REQUIRE( abs(tt.eval(ids)-myTf(ids))<1e-5 );
            REQUIRE( abs(tt.sum(vector(dim,wi))-cmpx(8.4999,60.8335))<1e-3 );
            cout<<"integral="<<tt.sum(vector(dim,wi))<<endl;
            SECTION("save/load tensor train")
            {
                tt.save("tt.txt");
                auto tt2=TensorTrain<cmpx>::load("tt.txt");
                for(auto i=0u; i<tt.M.size(); i++)
                    REQUIRE(arma::norm(arma::vectorise(tt.M[i]-tt2.M[i]))<1e-14);
            }
        }
        SECTION( "directly" )
        {
            auto ci=CTensorCI1<cmpx,double>(f,vector(dim,xi), {.nIter=120});

            vector<double> x0={xi[3],xi[5],xi[1],xi[5],xi[1]};
            auto ttc=ci.get_CTensorTrain(dim/2);
            REQUIRE( abs(ttc.eval(x0)-f(x0))<1e-5 );
            auto tt=ci.get_TensorTrain(dim/2);
            REQUIRE( abs(tt.sum(vector(dim,wi))-cmpx(8.4999,60.8335))<1e-3 );

            SECTION("svd") {
                auto mps=ci.get_TensorTrain();
                mps.compressSVD();
                vector<int> ids={3,5,1,5,1};
                REQUIRE( abs(mps.eval(ids)-f(x0))<1e-5 );
                REQUIRE( abs(mps.sum(vector(dim,wi))-cmpx(8.4999,60.8335))<1e-3 );
            }
        }

        SECTION( "condition" )
        {
            TensorCI1Param p;
            p.cond=[&xi=xi](vector<int> const& id)
            {
                double sum=0;
                for(auto i:id) sum+=xi[i];
                return sum<=1;
            };
            auto ci=CTensorCI1<cmpx,double>(f,vector(dim,xi), p);
            cout<<"rank nEval LastSweepPivotError\n";
            for(int r=1; r <= 100; r++)
            {
                ci.iterate();
                if (r<10 || r%10==0)
                    cout << r << " " << count << " " << ci.pivotError.back() << endl;
            }
            for(auto b=0; b<ci.len()-1; b++)  // all bonds
                for(auto r=0u; r<ci.Iset[b+1].size(); r++) { // all pivots
                    MultiIndex ij=ci.Iset[b+1][r]+ci.Jset[b][r];
                    REQUIRE( ci.param.cond( vector<int>(ij.begin(),ij.end()) ) );
                }


        }
    }
}


TEST_CASE("quantics")
{
    SECTION("scale separation")
    {
        int dim=20;
        double tmin=-100, tmax=100;
        double dt=(tmax-tmin)/((1<<dim)-1);
        auto ft=[](double t){ return (10*exp(-t*t)+1)*cos(t); };
        size_t c=0;
        auto fq=[&](const vector<int>& id)
        {
            c++;
            bitset<32> b;
            for(auto i=0u;i<id.size();i++)
                b[i]=id[i];
            return ft(tmin+dt*b.to_ullong());
        };

        TensorCI1Param p;
        p.pivot1=vector(dim,0);
        p.reltol=1e-14;
        p.fullPiv=true;
        auto ci=TensorCI1<double>(fq, vector(dim,2), p);
//        ci.addPivotGlobal(vector(dim,1));

//        cout << "\nrank nEval LastSweepPivotError\n" << setprecision(12);
//        for (int i = 1; i <= 12; i++) {
//            ci.iterate();
//            cout << i << " " << c << " " << ci.pivotError.back()<<"\n";
//        }
//        cout << "true error=" << ci.trueError(1<<dim) << "\n";
    }

}

