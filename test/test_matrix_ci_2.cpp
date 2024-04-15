#include<catch2/catch.hpp>
#include "xfac/tensor/tensor_ci_2.h"
#include "xfac/grid.h"

using namespace arma;
using namespace xfac;

using cmpx=std::complex<double>;

TEST_CASE( "Test matrix ci 2" )
{
    // Checks that CI is exact for matrix of finite rank.
    SECTION( "fixed rank" )
    {
        int n=100, rank=50;
        mat A = mat(n,rank,fill::randn) * mat(rank,n,fill::randn); // mat is armadillo matrix
        function myf=[&](vector<int> const& id){ return A(id[0],id[1]); };
        vector<int> dims={int(A.n_rows), int(A.n_cols)};
        auto ci=TensorCI2<double>(myf, dims, {.bondDim=rank+10});
        ci.iterate(6);

        REQUIRE( ci.pivotError.size()-1 == rank );
        REQUIRE( ci.trueError() <= 1e-13*norm(A) );
        int i=rand()%A.n_rows, j=rand()%A.n_cols;
        REQUIRE( std::abs(ci.tt.eval({i, j})-A(i, j)) < 1e-10 );
    }

    SECTION("continuous function")
    {
        function myf=[](double x,double y) {
            double arg=1.0+(x+2*y+x*y)*M_PI;
            return cmpx(1+x+cos(arg),x*x+0.5*sin(arg));
        };
        auto xi=grid::linspace<double>(0,1,100);
        int rank=15;

        SECTION( "discretized" )
        {
            cx_mat A(xi.size(), xi.size());
            for(auto i=0u;i<xi.size();i++)
                for(auto j=0u;j<xi.size();j++)
                    A(i,j)=myf(xi[i],xi[j]);
            function myTf=[&](vector<int> const& id){ return A(id[0],id[1]); };
            vector<int> dims={int(A.n_rows), int(A.n_cols)};
            auto ci=TensorCI2<cmpx>(myTf, dims, {.bondDim=rank});
            ci.iterate(5);

            REQUIRE(ci.pivotError.size()-1==12);
            REQUIRE( ci.trueError() <= 1e-12*norm(A) );
            int i=rand()%A.n_rows, j=rand()%A.n_cols;
            REQUIRE( abs(ci.tt.eval({i, j})-A(i, j)) < 1e-10 );
        }

        SECTION( "directly" )
        {
            auto myTf=[&](vector<double> xs){return myf(xs[0],xs[1]);};
            auto ci=CTensorCI2<cmpx,double>(myTf, vector(2,xi), {.bondDim=2*rank});
            ci.iterate(5);
            REQUIRE( ci.trueError() <= 1e-10 );
            double x=1.0*rand()/RAND_MAX, y=1.0*rand()/RAND_MAX;
            REQUIRE( std::abs(ci.get_CTensorTrain().eval({x, y})-myf(x, y))< 1e-10 );
            ci.makeCanonical();
            REQUIRE( std::abs(ci.get_CTensorTrain().eval({x, y})-myf(x, y))< 1e-10 );
        }
    }

}

