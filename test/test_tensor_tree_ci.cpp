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

        function func=[&](vector<double> const& x) {return cos(4*x[0] + x[1] + 2*x[2]);};
        function tfunc = [&](vector<int> xi){ return func(grid.id_to_coord(xi));};

        auto tree = makeTuckerTree(dim, nBit);

        auto ci=TensorTreeCI<double>(tfunc, tree, grid.tensorDims(), {.pivot1=vector(grid.tensorLen, 0)});
        ci.addPivotsAllBonds({vector(grid.tensorLen, 1)});

        vector<double> x = {0.3, 0.2, 0.7};
        REQUIRE ( abs(ci.tt.eval(grid.coord_to_id(x)) - func(x)) <= 1e-5 );
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

}
