#include<catch2/catch.hpp>

#include<iostream>
#include "xfac/grid.h"
#include "xfac/tree/tensor_tree.h"
#include "xfac/tree/tensor_tree_ci.h"

using namespace std;
using namespace xfac;

using cmpx=std::complex<double>;


TEST_CASE( "Test tensor tree" )
{
    SECTION( "cos dmrg2" )
    {
        int nBit = 25;
        int dim=3;
        grid::Quantics grid(0., 1., nBit, dim);

        function func=[&](vector<double> const& x) {
            auto sum=accumulate(x.begin(), x.end(), 0.0);
            return cos(sum) + 0.5 * cos(4*x[0] + x[1]) + 0.2 * sin(sum * sum);};
        function tfunc = [&](vector<int> xi){ return func(grid.id_to_coord(xi));};

        auto tree = makeTuckerTree(dim, nBit);

        auto ci=TensorTreeCI<double>(tfunc, tree, grid.tensorDims());
        ci.iterate(10);

        // test function interpolation
        vector<double> x = {0.3, 0.2, 0.7};
        REQUIRE ( abs(ci.tt.eval(grid.coord_to_id(x)) - func(x)) <= 1e-5 );

        // test integration over hypercube
        auto integ = ci.tt.sum() * grid.deltaVolume;
        auto integ_ref = -0.0466396;  // ref result from TensorCI2 quantics
        REQUIRE ( abs(integ - integ_ref) <= 1e-5 );
    }

    SECTION( "cos1d" )
    {
        int nBit = 25;
        int dim=1;
        grid::Quantics grid(0., 1., nBit, dim);

        function func=[&](vector<double> const& x) {return cos(M_PI*x[0]);};
        function tfunc = [&](vector<int> xi){ return func(grid.id_to_coord(xi));};

        auto tree = makeTuckerTree(dim, nBit);
        auto ci=TensorTreeCI<double>(tfunc, tree, grid.tensorDims(), {.pivot1=vector(grid.tensorLen, 0)});
        ci.addPivotsAllBonds({vector(grid.tensorLen, 1)});

        // test function interpolation
        vector<double> x = {0.2};
        REQUIRE ( abs(ci.tt.eval(grid.coord_to_id(x)) - func(x)) <= 1e-5 );

        // test integration over hypercube
        auto integ = ci.tt.sum() * grid.deltaVolume;
        REQUIRE ( std::abs(integ - 0.0) <= 1e-5 );

        SECTION( "norm")
        {
            REQUIRE(std::abs(ci.tt.norm2()*grid.deltaVolume-0.5)<1e-4);
        }
        SECTION("overlap")
        {
            function func2=[&](vector<double> const& x) {return 0.5*cos(2*M_PI*x[0])+sin(M_PI*x[0]);};
            function tfunc2 = [&](vector<int> xi){ return func2(grid.id_to_coord(xi));};

            auto ci2=TensorTreeCI<double>(tfunc2, tree, grid.tensorDims(), {.pivot1=vector(grid.tensorLen, 0)});
            ci2.addPivotsAllBonds({vector(grid.tensorLen, 1)});
            ci2.iterate(5);

            // test function interpolation
            vector<double> x = {0.2};
            REQUIRE ( std::abs(ci2.tt.eval(grid.coord_to_id(x)) - func2(x)) <= 1e-5 );

            // test integration over hypercube
            auto integ = ci2.tt.sum() * grid.deltaVolume;
            REQUIRE ( std::abs(integ - 2.0/M_PI) <= 1e-5 );
            REQUIRE( std::abs(ci2.tt.overlap(ci.tt))*grid.deltaVolume < 1e-5);
        }
    }

    SECTION( "random piv" )
    {
        srand(0);
        int num_rand_vec = 20; // number of random global vectors

        int nBit = 25;
        int dim=3;
        grid::Quantics grid(0., 1., nBit, dim);

        function func=[&](vector<double> const& x) {
            auto sum=accumulate(x.begin(), x.end(), 0.0);
            return cos(sum) + 0.5 * cos(4*x[0] + x[1]) + 0.2 * sin(sum * sum);};
        function tfunc = [&](vector<int> xi){ return func(grid.id_to_coord(xi));};

        auto tree = makeTuckerTree(dim, nBit);

        vector<vector<int>> pivots;
        for(auto i=0; i<num_rand_vec; i++) {
            vector<int> random_vec;
            for (auto j=0; j<grid.tensorLen; j++) {
                int randomBit = rand() % 2;
                random_vec.push_back(randomBit);
            }
            pivots.push_back(random_vec);
        }

        auto ci=TensorTreeCI<double>(tfunc, tree, grid.tensorDims(), {.pivot1=vector(grid.tensorLen, 0)});
        ci.addPivotsAllBonds(pivots);

        vector<double> x = {0.3, 0.2, 0.7};
        REQUIRE ( abs(ci.tt.eval(grid.coord_to_id(x)) - func(x)) <= 1e-5 );
    }

    SECTION( "cos" )
    {
        int nBit = 25;
        int dim=3;
        grid::Quantics grid(0., 1., nBit, dim);

        function func=[&](vector<double> const& x) {return cos(4*x[0] + x[1] + 2*x[2])+sin(x[0] + 2*x[1] + 4*x[2]);};
        function tfunc = [&](vector<int> xi){ return func(grid.id_to_coord(xi));};

        auto tree = makeTuckerTree(dim, nBit);

        auto ci=TensorTreeCI<double>(tfunc, tree, grid.tensorDims(), {.pivot1=vector(grid.tensorLen, 0)});
        ci.addPivotsAllBonds({vector(grid.tensorLen, 1)});
        ci.iterate(5);

        vector<double> x = {0.3, 0.2, 0.7};
        REQUIRE ( std::abs(ci.tt.eval(grid.coord_to_id(x)) - func(x)) <= 1e-5 );
        REQUIRE ( std::abs(ci.tt.sum()*grid.deltaVolume + 0.4722) <= 1e-5);
        REQUIRE ( std::abs(ci.tt.norm2()*grid.deltaVolume-1.00492) <= 1e-5);

        SECTION( "save")
        {
            ci.tt.save("ttree.txt");
            auto tt=TensorTree<double>::load("ttree.txt");
            REQUIRE ( std::abs(tt.eval(grid.coord_to_id(x)) - func(x)) <= 1e-5 );
            REQUIRE ( std::abs(tt.sum()*grid.deltaVolume + 0.4722) <= 1e-5);
            REQUIRE ( std::abs(tt.norm2()*grid.deltaVolume-1.00492) <= 1e-5);
        }
    }

    SECTION( "exp" )
    {
        int nBit = 25;
        int dim=3;
        grid::Quantics grid(0., 1., nBit, dim);

        function func=[&](vector<double> const& x) {return exp(4*x[0] + x[1] + 2*x[2]);};
        function tfunc = [&](vector<int> xi){ return func(grid.id_to_coord(xi));};

        auto tree = makeTuckerTree(dim, nBit);
        auto ci=TensorTreeCI<double>(tfunc, tree, grid.tensorDims(), {.pivot1=vector(grid.tensorLen, 0)});

        vector<double> x = {0.3, 0.2, 0.7};
        REQUIRE ( abs(ci.tt.eval(grid.coord_to_id(x)) - func(x)) <= 1e-5 );
    }

    SECTION( "Function in Rn" )
    {
        int nBit = 25;
        int dim = 5;
        grid::Quantics grid(0., 1., nBit, dim);

        auto f=[](const vector<double>& xs) {
            double x=0, y=0,c=0;
            for(auto xi:xs) {c++; x+=c*xi; y+=xi*xi/c;}
            double arg=1.0+(x+2.11*y+x*y)*M_PI;
            return cmpx(1+x+cos(arg),x*x+0.5*sin(arg));
        };
        function tfunc = [&](vector<int> xi){ return f(grid.id_to_coord(xi));};

        auto tree = makeTuckerTree(dim, nBit);

        // evaluation point is taken identical to similar test for TensorCI2
        vector<double> x0 = {0.12923440720030277, 0.297077424311301408, 0.0254460438286207569,
                             0.297077424311301408, 0.0254460438286207569};

        SECTION( "compression" )
        {
            auto ci=TensorTreeCI<cmpx>(tfunc, tree, grid.tensorDims(), {.bondDim=400});
            ci.iterate(3);

            // test function interpolation
            REQUIRE ( abs(ci.tt.eval(grid.coord_to_id(x0)) - f(x0)) < 1e-5 );

            // test integration over hypercube
            auto integ = ci.tt.sum() * grid.deltaVolume;
            auto integ_ref = cmpx(8.4999,60.8335);  // ref result from TensorCI2
            REQUIRE ( abs(integ - integ_ref) < 1e-3 );

            //for (auto const & m : ci.tt.M) std::cout << "ranks= " <<  m.n_rows << "  " << m.n_cols << " " << m.n_slices << "\n";
            //std::cout <<std::setprecision(12)<< "uncompressed: f= " << ci.tt.eval(grid.coord_to_id(x)) << " int= " << integ << "\n";

            SECTION("SVD") {
                auto mps=ci.tt;
                mps.compressSVD();
                integ = mps.sum() * grid.deltaVolume;
                REQUIRE( abs(mps.eval(grid.coord_to_id(x0))-f(x0)) < 1e-5 );
                REQUIRE( abs(integ - integ_ref) < 1e-3 );
                //for (auto const & m : mps.M) std::cout << "SVD ranks= " <<  m.n_rows << "  " << m.n_cols << " " << m.n_slices << "\n";
                //std::cout <<std::setprecision(12)<< "SVD: f= " << mps.eval(grid.coord_to_id(x)) << " int= " << integ << "\n";
            }

            SECTION("LU") {
                auto mps=ci.tt;
                mps.compressLU();
                integ = mps.sum() * grid.deltaVolume;
                REQUIRE( abs(mps.eval(grid.coord_to_id(x0))-f(x0)) < 1e-5 );
                REQUIRE( abs(integ - integ_ref) < 1e-3 );
                //for (auto const & m : mps.M) std::cout << "LU ranks= " <<  m.n_rows << "  " << m.n_cols << " " << m.n_slices << "\n";
                //std::cout <<std::setprecision(12)<< "LU: f= " << mps.eval(grid.coord_to_id(x)) << " int= " << integ << "\n";
            }

            SECTION("CI") {
                auto mps=ci.tt;
                mps.compressCI();
                integ = mps.sum() * grid.deltaVolume;
                REQUIRE( abs(mps.eval(grid.coord_to_id(x0))-f(x0)) < 1e-5 );
                REQUIRE( abs(integ - integ_ref) < 1e-3 );
                //for (auto const & m : mps.M) std::cout << "CI ranks= " <<  m.n_rows << "  " << m.n_cols << " " << m.n_slices << "\n";
                //std::cout <<std::setprecision(12)<< "CI: f= " << mps.eval(grid.coord_to_id(x)) << " int= " << integ << "\n";
            }
        }

        SECTION( "condition" )
        {
            auto cond=[](vector<double> const& x)
            {
                double sum=0;
                for(auto xi:x) sum+=xi;
                return sum<=1;
            };
            auto tCond = [&](vector<int> xi){ return cond(grid.id_to_coord(xi));};

            for (auto fullPiv : {true, false}) {
                TensorCIParam p;
                p.bondDim=400;
                p.fullPiv=fullPiv;
                p.cond=tCond;

                auto ci=TensorTreeCI<cmpx>(tfunc, tree, grid.tensorDims(), p);
                ci.iterate(3);

                REQUIRE( cond(x0) == true );  // make sure condition is fulfilled on evaluation point
                REQUIRE( abs(ci.tt.eval(grid.coord_to_id(x0))-f(x0))<1e-5 );

                for (auto [from,to]:tree.leavesToRoot()) {
                    for(auto r=0u; r<ci.Iset[{from, to}].size(); r++) { // all pivots
                        MultiIndex ij=add(ci.Iset[{from, to}].at(r), ci.Iset[{to, from}].at(r));
                        REQUIRE( ci.param.cond( vector<int>(ij.begin(),ij.end()) ) );
                    }
                }
            }
        }
    }
}
