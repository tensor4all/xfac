#include<catch2/catch.hpp>

#include "xfac/grid.h"
#include "xfac/tensor/tensor_ci_converter.h"

#include <iomanip>
#include <iostream>
#include <bitset>

using namespace std;
using namespace xfac;

using cmpx=std::complex<double>;



TEST_CASE("global pivot")
{
    SECTION("product of two rank 2")
    {
        // ( |000>+|001> )*( |000>+|111> )
        function f=[](vector<int> id) {
            const int n=6;
            bitset<n> b;
            for(auto i=0u; i<n; i++)
                b[i]=id[i];
            auto x=b.to_ulong();
            return (x==0 || x==7 || x==8 || x==15) ? 1.0 : 0.0;
        };
        auto ci=TensorCI1<double>(f,vector(6,2),{.nIter=2});
        SECTION("tci1 initialization") {
            vector<int> rank;
            for(auto const& Pi:ci.P) rank.push_back(Pi.n_rows);
            REQUIRE(rank==vector(6,1));
            REQUIRE(ci.trueError()==1);
        }
        auto ci2=to_tci2(ci);
        ci2.addPivotsAllBonds({{1,1,1, 1,0,0}});
        SECTION("tci2 with 1 global pivot") {
            vector<int> rank;
            for(auto const& Pi:ci2.P) rank.push_back(Pi.n_rows);
            REQUIRE(rank==vector{2,2,1,1,1,0});
            REQUIRE(ci2.trueError()==0);
        }
        SECTION("tci1 conversion") {
            ci2.makeCanonical();
            auto ci1=to_tci1(ci2);
            vector<int> rank;
            for(auto const& Pi:ci1.P) rank.push_back(Pi.n_rows);
            REQUIRE(rank==vector{2,2,1,1,1,1});
            REQUIRE(ci1.trueError()<1e-15);
        }
    }

    SECTION("quantics")
    {
        function f2=[&](double x) { return pow(x+0.1,5)+pow(x-0.5,4); };
        auto ci=QTensorCI<double>(f2, grid::Quantics{0,1,14});
        ci.addPivotsAllBonds({{0,  1, 1,   0, 0, 0, 0, 0, 0, 0, 0, 0,0,0}});
        for(auto const& Pi:ci.P) cout<<Pi.n_rows<<" ";
        cout<<endl;

        vector p2={0,  1, 1,   1, 1, 1, 1, 1, 1, 1, 1, 1,1,1};
        ci.addPivotsAllBonds({p2});
        for(auto const& Pi:ci.P) cout<<Pi.n_rows<<" ";
        cout<<endl;

        ci.makeCanonical();
        auto ci1=to_tci1(ci);
        vector<int> rank;
        for(auto const& Pi:ci1.P) rank.push_back(Pi.n_rows);
        REQUIRE(rank==vector{1,2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1});
        REQUIRE(1-ci1.get_TensorTrain().overlap(ci.tt)/ci.tt.norm2() < 1e-14);
    }

    SECTION( "copy pivots quantics" )
    {
        function f2=[&](double x) { return pow(x+0.1,5)+pow(x-0.5,4); };
        grid::Quantics grid{0,1,30};
        auto fq=[&](const vector<int>& id) { return f2(grid.id_to_coord(id)[0]); };
        auto ci=TensorCI1<double>(fq, grid.tensorDims());
        //cout<<"\nrank nEval LastSweepPivotError integral(f)\n"<<setprecision(12);
        for(int i=1;i<=20;i++)
        {
            ci.iterate();
            //cout<<i<<" "<<count<<" "<<ci.lastSweepPivotError()<<" "<<ci.sumWeighted(vector(dim,vector(2,1.0)))*dx<<endl;
            if (ci.pivotError.back()<ci.param.reltol) break;
        }

        auto exact = pow(1.1,6)/6-pow(0.1,6)/6 + pow(0.5,5)/5+pow(0.5,5)/5;
        auto res = ci.get_TensorTrain().sum1()*grid.deltaVolume;
        REQUIRE(abs(res - exact)<1e-6);

        SECTION("same function")
        {
            auto ci_1 = ci;
            auto res_1 = ci_1.get_TensorTrain().sum1()*grid.deltaVolume;
            REQUIRE(abs(res_1 - exact)<1e-6);

            auto ci_2 = to_tci2(ci);
            auto res_2 = ci_2.tt.sum1()*grid.deltaVolume;
            REQUIRE(abs(res_2 - exact)<1e-6);
        }

        SECTION("another function")
        {
            function g=[&](double x) { return (pow(x+0.1,5)-pow(x-0.5,4))*1e2; };
            auto gq=[&](const vector<int>& id) { return g(grid.id_to_coord(id)[0]); };

            auto exact_g = (pow(1.1,6)/6-pow(0.1,6)/6 - pow(0.5,5)/5-pow(0.5,5)/5)*1e2;
            auto ci2=to_tci2<double>(ci,gq);
            REQUIRE(ci2.P[grid.nBit/2].n_rows==2);
            ci2.makeCanonical();
            REQUIRE(ci2.P[grid.nBit/2].n_rows==2);
            auto ci_g = to_tci1(ci2);
            auto res_g = ci_g.get_TensorTrain().sum1()*grid.deltaVolume;
            REQUIRE(abs(res_g - exact_g)<1e-5);
        }
    }
}


TEST_CASE("quantics 1s orbital")
{
    auto ft=[](double x, double y, double z){ return exp(-sqrt(x*x+y*y+z*z)); };
    auto packed=GENERATE(false,true);
    int ms[2]={105,75};  // bond dimension depending on packed
    grid::Quantics grid {-40, 40, 30, 3, packed};
    auto weight=GENERATE_COPY(vector<vector<double>>{},
                              vector(grid.tensorLen,vector(grid.tensorLocDim,1.0)));
    auto fq=[&](const vector<int>& id) { auto r=grid.id_to_coord(id); return ft(r[0],r[1],r[2]); };

    vector<vector<int>> pivots;
    for(double x:{-1,1})
        for(double y:{-1,1})
            for(double z:{-1,1})
                pivots.push_back(grid.coord_to_id({x,y,z}));

    auto ci=TensorCI2<double>(fq,vector(grid.tensorLen,grid.tensorLocDim),
                              {.bondDim=ms[packed], .reltol=1e-7, .pivot1=pivots[0], .fullPiv=!packed,
                               .weight=weight,
                               .useCachedFunction=false});
    ci.addPivotsAllBonds(pivots);
    double error=1;
    DYNAMIC_SECTION("tci2 packed="<<packed<<" weight="<<!weight.empty()) {
        while (ci.cIter<6 && std::abs(error)>2e-5) {
            ci.iterate();
            error=ci.tt.sum1()*grid.deltaVolume/(8*M_PI)-1;
            cout<<ci.P[ci.len()/2].n_rows<<" "<<ci.pivotError.back()<<" "<<error<<endl;
        }
        for(auto i=0u; i<ci.pivotError.size(); i+=10)
            cout<<i<<" "<<ci.pivotError[i]<<endl;
        cout<<"integral/exact-1="<<error<<endl;
        REQUIRE(std::abs(error)<2e-5);
    }
    DYNAMIC_SECTION("tci1 packed="<<packed<<" weight="<<!weight.empty()) {
        ci.makeCanonical();
        auto ci1=to_tci1(ci);
        while (ci1.cIter<ci.param.bondDim && std::abs(error)>2e-5) {
            ci1.iterate();
            if (ci1.cIter%5==0) {
                error=ci1.get_TensorTrain().sum1()*grid.deltaVolume/(8*M_PI)-1;
                cout<<ci1.P[ci1.len()/2].n_rows<<" "<<ci1.pivotError.back()<<" "<<error<<endl;
            }
        }
        for(auto i=0u; i<ci1.pivotError.size(); i+=10)
            cout<<i<<" "<<ci1.pivotError[i]<<endl;
        error=ci1.get_TensorTrain().sum1()*grid.deltaVolume/(8*M_PI)-1;
        cout<<ci1.pivotError.size()<<" "<<ci1.pivotError.back()<<" "<<error<<endl;
        REQUIRE(std::abs(error)<2e-5);
    }
}
