#include<catch2/catch.hpp>

#include<iostream>
#include<bitset>
#include <set>
#include "xfac/grid.h"
#include "xfac/tensor/tensor_ci_2.h"
#include "xfac/tensor/auto_mpo.h"

using namespace std;
using namespace xfac;

using cmpx=std::complex<double>;


TEST_CASE( "Test tensor CI 2" )
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

        auto ci=TensorCI2<double>(myf, vector(dim,d), {.bondDim=10*d});
        ci.iterate(3);
        REQUIRE( ci.pivotError.size()-1==4 );
        REQUIRE( count < int(pow(d*ci.pivotError.size(),2)*(dim-2)*ci.cIter) );
    }

    SECTION( "Function in Rn" )
    {
        long count=0;
        auto f=[&count](const vector<double>& xs) {
            count++;
            double x=0, y=0,c=0;
            for(auto xi:xs) {c++; x+=c*xi; y+=xi*xi/c;}
            double arg=1.0+(x+2.11*y+x*y)*M_PI;
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
            auto ci=TensorCI2<cmpx>(myTf, vector<int>(dim,xi.size()), {.bondDim=120, .reltol=1e-10, .nRookIter=3, .useCachedFunction=true});
            while(!ci.isDone()) ci.iterate();

            cout<<"cIter="<<ci.cIter<<", rank="<<ci.pivotError.size()<<", nEval="<<ci.f.nEval()<<endl;
            for(auto i=0u; i<ci.pivotError.size()-1; i+=10)
                    cout << i << " " << ci.pivotError[i] << endl;
            for(auto i=0u; i<10; i++)
                cout << ci.pivotError.size()-10+i << " " << ci.pivotError[ci.pivotError.size()-10+i] << endl;

            vector<int> ids={3,5,1,5,1};
            REQUIRE( abs(ci.tt.eval(ids)-myTf(ids))<1e-5 );
            cout<<"integral="<<ci.tt.sum(vector(ci.len(),wi))<<endl;
            SECTION("save/load tensor train")
            {
                ci.tt.save("tt.txt");
                auto tt2=TensorTrain<cmpx>::load("tt.txt");
                for(auto i=0u; i<ci.tt.M.size(); i++)
                    REQUIRE(arma::norm(arma::vectorise(ci.tt.M[i]-tt2.M[i]))<1e-14);
            }
        }
        SECTION( "directly" )
        {
            auto ci=CTensorCI2<cmpx,double>(f,vector(dim,xi), {.bondDim=120});
            while (!ci.isDone()) ci.iterate();
            ci.iterate();
            vector<int> ids={3,5,1,5,1};
            vector<double> x0={xi[3],xi[5],xi[1],xi[5],xi[1]};
            REQUIRE( abs(ci.tt.eval(ids)-f(x0))<1e-5 );
            REQUIRE( abs(ci.get_CTensorTrain().eval(x0)-f(x0))<1e-5 );
            REQUIRE( abs(ci.tt.sum(vector(dim,wi))-cmpx(8.4999,60.8335))<1e-3 );
            ci.makeCanonical();
            REQUIRE( abs(ci.get_CTensorTrain().eval(x0)-f(x0))<1e-5 );
            REQUIRE( abs(ci.tt.sum(vector(dim,wi))-cmpx(8.4999,60.8335))<1e-3 );

            SECTION("svd") {
                auto mps=ci.tt;
                mps.compressSVD();
                vector<int> ids={3,5,1,5,1};
                REQUIRE( abs(mps.eval(ids)-f(x0))<1e-5 );
                REQUIRE( abs(mps.sum(vector(dim,wi))-cmpx(8.4999,60.8335))<1e-3 );
            }
            SECTION("from tt") {
                auto mps=ci.tt;
                auto ci2=TensorCI2<cmpx>(mps, ci.param);
                vector<int> ids={3,5,1,5,1};
                REQUIRE( abs(ci2.tt.eval(ids)-f(x0))<1e-5 );
                REQUIRE( abs(ci2.tt.sum(vector(dim,wi))-cmpx(8.4999,60.8335))<1e-3 );
            }
        }
    }

    SECTION( "global pivot" )
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
            auto ci=TensorCI2<double>(f,vector(6,2),{.bondDim=2});
            ci.iterate();
            for(auto const& x : ci.tt.M) cout<<x.n_slices<<" ";
            cout<<"trueError="<<ci.trueError()<<endl;
            ci.addPivotsAllBonds({{1,1,1, 1,0,0}});
            for(auto const& x : ci.tt.M) cout<<x.n_slices<<" ";
            cout<<"trueError="<<ci.trueError()<<endl;
        }
    }
}

TEST_CASE("quantics 2")
{
    SECTION("scale separation")
    {
        auto ft=[](double t){ return (10*exp(-t*t)+1)*cos(t); };
        grid::Quantics grid {-100, 100, 20};
        auto ci=QTensorCI<double>(ft, grid::Quantics {-100, 100, 20}, {.bondDim=12});
        ci.addPivotsAllBonds({vector(20,1)});
        ci.iterate(5);

        cout<<"cIter="<<ci.cIter<<", rank="<<ci.pivotError.size()-1<<", nEval="<<ci.f.nEval()<<endl;
        for(auto i=0u; i<ci.pivotError.size(); i++)
            cout << i << " " << ci.pivotError[i] << endl;
        cout << "true error=" << ci.trueError(1<<ci.grid.nBit) << "\n";
    }

    SECTION( "restart from tensor train" )
    {
        function f2=[&](double x) { return pow(x+0.1,5)+pow(x-0.5,4); };
        auto exact = pow(1.1,6)/6-pow(0.1,6)/6 + pow(0.5,5)/5+pow(0.5,5)/5;

        map<double,double> cache;
        auto f=[&](vector<double> y) { double x = y[0]; return cache[x]=f2(x); };

        auto ci=QTensorCI<double>(f, grid::Quantics {0.0, 1.0, 30}, {.bondDim=20, .useCachedFunction=true});
        ci.iterate(3);

        auto ci_2=QTensorCI<double>(f, grid::Quantics {0.0, 1.0, 30}, ci.get_qtt().tt, {.bondDim=20, .useCachedFunction=true});
        ci_2.iterate();
        auto res = ci_2.get_qtt().integral();
        REQUIRE(std::abs(res - exact)<1e-9);
    }

    SECTION( "copy pivots quantics" )
    {
        function f2=[&](double x) { return pow(x+0.1,5)+pow(x-0.5,4); };

        map<double,double> cache;
        auto f=[&](double x) { return cache[x]=f2(x); };

        auto ci=QTensorCI<double>(f, grid::Quantics {0.0, 1.0, 30}, {.bondDim=20, .useCachedFunction=true});
        ci.iterate(10);
        auto exact = pow(1.1,6)/6-pow(0.1,6)/6 + pow(0.5,5)/5+pow(0.5,5)/5;
        auto res = ci.get_qtt().integral();
        REQUIRE(std::abs(res - exact)<1e-9);
        cout<<"neval="<<cache.size()<<endl;

        SECTION("same function")
        {
            auto ci_2 = QTensorCI<double>(f, grid::Quantics {0.0, 1.0, 30}, ci.param);
            cout<<"neval constructor ="<<cache.size()<<endl;
            ci_2.addPivots(ci);
            cout<<"neval addPivots ="<<cache.size()<<endl;
            ci_2.makeCanonical();
            cout<<"neval makeCanical="<<cache.size()<<endl;
            auto res_2 = ci_2.get_qtt().integral();
            REQUIRE(std::abs(res_2 - exact)<1e-9);
        }

        SECTION("another function")
        {
            function g=[&](double x) { return (pow(x+0.1,5)-pow(x-0.5,4))*1e2; };
            auto exact_g = (pow(1.1,6)/6-pow(0.1,6)/6 - pow(0.5,5)/5-pow(0.5,5)/5)*1e2;
            auto ci_g = QTensorCI<double>(g, ci.grid, ci.param);
            ci_g.addPivots(ci);
            //ci_g.makeCanonical();
            auto res_g = ci_g.get_qtt().integral();
            REQUIRE(std::abs(res_g - exact_g)<1e-7);
        }
    }

    SECTION("FP")
    {
        int dim=12;
        double dx = 2.4415140415140414e-05;
        vector<double> data(size_t(1)<<dim);
        {
            ifstream in("qFP.txt");
            if (!in) throw runtime_error("file qFP.txt not found");
            for(auto &x:data) in>>x;
        }
        long count=0;
        auto fq=[&](const vector<int>& id)
        {
            count++;
            bitset<32> b;
            for(auto i=0u;i<id.size();i++)
                b[i]=id[i];
            return data.at(b.to_ullong());
        };
        auto ci=TensorCI2<double>(fq, vector(dim,2), {.pivot1=vector(dim,1)});
        ci.iterate(4);
        SECTION("normal iteration")
        {
//            cout << "\nrank PivotError\n" << setprecision(12);
//            for (auto i=0u; i<ci.pivotError.size(); i++)
//                cout << i << " " << ci.pivotError[i] << endl;
            REQUIRE(ci.trueError()<1e-13);
        }

        SECTION("copy pivots")
        {
            ci.makeCanonical();
            auto integral=ci.tt.sum1()*dx;
            REQUIRE(ci.pivotError.back()<1e-12);

            auto ci2=TensorCI2<double>(fq, vector(dim,2), ci.param);
            ci2.addPivots(ci);
            REQUIRE(ci2.pivotError.back()<1e-12);
            auto integral2=ci2.tt.sum1()*dx;
            REQUIRE(std::abs(integral-integral2)<1e-15);
            REQUIRE(ci2.trueError()<1e-13);
        }

    }

    SECTION("FP1")
    {
        int dim=12;
        double dx = 2.4415140415140414e-05;
        vector<double> data(size_t(1)<<dim);
        {
            ifstream in("qFP_1.txt");
            if (!in) throw runtime_error("file qFP_1.txt not found");
            for(auto &x:data) in>>x;
        }
        long count=0;
        auto fq=[&](const vector<int>& id)
        {
            count++;
            bitset<32> b;
            for(auto i=0u;i<id.size();i++)
                b[i]=id[i];
            return data.at(b.to_ullong());
        };

        auto ci=TensorCI2<double>(fq, vector(dim,2), {.bondDim=2, .pivot1=vector(dim,1)});
        ci.iterate(2);
        ci.makeCanonical();

        auto ci2=TensorCI2<double>(fq, vector(dim,2), ci.param);
        ci2.addPivots(ci);
        ci2.makeCanonical();
        auto i1=ci.tt.sum1()*dx;
        auto i2=ci2.tt.sum1()*dx;
        REQUIRE(std::abs(i1/i2-1)<ci.param.reltol);
        REQUIRE(ci.trueError()==ci2.trueError());
    }

    SECTION("Hiroshi example")
    {
        int nBit=20;
        double abstol=1e-4, delta=10./(1<<nBit);
        cout<<"delta="<<delta<<endl;
//        std::srand(std::time(0));
        for(auto t=0; t<100; t++) {
            std::set<double> rpoint {0};
            for(auto i=0; i<20; i++) rpoint.insert(1.0*(rand()%(1<<nBit))/(1<<nBit));
            auto ft=[&](double x){
                double y=exp(-x);
                for(auto r:rpoint)
                    if (std::abs(r-x)<delta) y += 2*abstol;
                return y;
            };
            auto ci=QTensorCI<double>(ft, grid::Quantics{0, 1, nBit}, {.bondDim=100, .fullPiv=true});

            ci.iterate(2);
//            for(const auto& Mi : ci.tt.M) cout<<Mi.n_slices<<" ";
//            cout<<"\n";
            //cout << "true error=" << ci.trueError(1<<grid.nBit) << "\n";

            ci.addPivotValues({rpoint.begin(),rpoint.end()});
            ci.iterate(4);
            auto qtt=ci.get_qtt();

//            for(const auto& Mi : ci.tt.M) cout<<Mi.n_slices<<" ";
            if (true) for(double x:rpoint) {
                double errorx=std::abs(qtt.eval({x})-ft(x));
                if (errorx>1e-8) {
                    cout<<x<<" (points)-->"<<errorx<<endl;
                    for(double x:rpoint) {
                        cout<<x;
                        for(auto id : qtt.grid.coord_to_id({x})) cout<<" "<<id;
                        cout<<endl;
                    }
//                    cout<<"Pivots:\n";
//                    for(auto const& x:ci.getPivotsAt(nBit/2)) {
//                        for(auto id : x) cout<<" "<<id;
//                        cout<<endl;
//                    }
                    break;
                }
            }
            for(auto i=0; i<200; i++) {
                auto x=1.0*(rand()%(1<<nBit))/(1<<nBit);
                double errorx=std::abs(qtt.eval({x})-ft(x));
                if (errorx>1e-8) cout<<x<<" (random)-->"<<errorx<<endl;
            }

            //cout << "\npivot added, true error=" << ci.trueError(1<<grid.nBit) << "\n";
        }
    }
}

TEST_CASE("autoMPO")
{
    using namespace autompo;
    auto SzTotal=[](int L)
    {
        auto Sz=[=](int i) { return ProdOp<> {{ i%L, LocOp<> {{-1,0},{0,1}} }}; };
        PolyOp Szt;
        for(int i=0; i<L; i++)
            Szt += Sz(i);
        return Szt;
    };
    PolyOp H=SzTotal(7);
    auto mps=H.to_tensorTrain();
    SECTION("compressCI") {
        mps.compressCI();
        REQUIRE(mps.M[mps.M.size()/2].n_rows==2);
        REQUIRE(fabs(H.overlap(mps)/mps.norm2()-1)<1e-15);
    }
    SECTION("compressSVD") {
        mps.compressSVD();
        REQUIRE(mps.M[mps.M.size()/2].n_rows==2);
        REQUIRE(fabs(H.overlap(mps)/mps.norm2()-1)<1e-15);
    }
}
