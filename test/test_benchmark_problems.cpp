#include<catch2/catch.hpp>

#include<iostream>
#include "xfac/grid.h"
#include "xfac/tree/tensor_tree_ci.h"
#include "xfac/tensor/tensor_ci_2.h"

using namespace std;
using namespace xfac;

using cmpx=std::complex<double>;


TEST_CASE( "Test benchmark problems", "[.][slow]" )
{
    SECTION( "exp-x integral")  // integral from the xfac paper in Fig. 8
    {
        int nBit = 30;
        int dim = 3;
        grid::Quantics grid{-40., 40., nBit, dim};

        function func = [](double x, double y, double z){ return exp( -sqrt(x*x + y*y + z*z) ); };
        function tfunc = [&](vector<int> xi){ auto r = grid.id_to_coord(xi); return func(r[0], r[1], r[2]); };

        vector<vector<int>> pivots;
        for (auto x:{-1., 1.})
            for (auto y:{-1., 1.})
                for (auto z:{-1., 1.})
                    pivots.push_back(grid.coord_to_id({x, y, z}));

        double exact = 8 * M_PI;

        SECTION( "tree" )
        {
            auto tree = makeTuckerTree(dim, nBit);
            auto ci=TensorTreeCI<double>(tfunc, tree, grid.tensorDims(), {.bondDim=600, .reltol=1e-14, .fullPiv=true});
            ci.addPivotsAllBonds(pivots);

            std::cout << "# exp-x integral on tree: iteration, integral, number of evaluations, last pivot error, abs(integral / exact - 1), rank\n";
            for (auto i=0u; i< 10; i++){
                ci.iterate();
                auto integ = ci.tt.sum() * grid.deltaVolume;
                std::cout << std::setprecision(14) << i << " " << integ << " " << ci.f.nEval() << " " << ci.pivotError.back() << " " << abs(integ / exact - 1) << " " << ci.pivotError.size() << std::endl;
            }
            std::cout << "# exp-x integral on tree: rank, pivot error\n";
            for (auto i=0u; i<ci.pivotError.size(); i++)
                std::cout << std::setprecision(14) << i << " " << ci.pivotError[i] << endl;
        }

        SECTION( "ci2" )
        {
            auto ci=TensorCI2<double>(tfunc, grid.tensorDims(), {.bondDim=600, .reltol=1e-14, .fullPiv=true});
            ci.addPivotsAllBonds(pivots);

            std::cout << "# exp-x integral on ci2: iteration, integral, number of evaluations, last pivot error, abs(integral / exact - 1), rank\n";
            for (auto i=0u; i< 10; i++){
                ci.iterate();
                auto integ = ci.tt.sum1() * grid.deltaVolume;
                std::cout << std::setprecision(14) << i << " " << integ << " " << ci.f.nEval() << " " << ci.pivotError.back() << " " << abs(integ / exact - 1) << " " << ci.pivotError.size() << std::endl;
            }
            std::cout << "# exp-x integral on ci2: rank, pivot error\n";
            for (auto i=0u; i<ci.pivotError.size(); i++)
                std::cout << std::setprecision(14) << i << " " << ci.pivotError[i] << endl;
        }

    }

    SECTION( "oscintegral" )  // integral from the xfac paper, Eq. (5) and Fig. 2
    {
        int nBit = 30;
        int dim = 10;
        grid::Quantics grid(-1., 1., nBit, dim);

        function func=[](vector<double> const& x) {
            vector<double> x2;
            for (auto xi : x) x2.push_back(xi * xi);
            auto sum=accumulate(x.begin(), x.end(), 0.0);
            auto sum2=accumulate(x2.begin(), x2.end(), 0.0);
            return 1000 * cos(10 * sum2) * exp(-pow(sum, 4) / 1000);
        };
        function tfunc = [&](vector<int> xi){ return func(grid.id_to_coord(xi));};

        SECTION( "tree" )
        {
            auto tree = makeTuckerTree(dim, nBit);
            auto ci=TensorTreeCI<double>(tfunc, tree, grid.tensorDims(), {.bondDim=60});

            std::cout << "# oscillatory integral on tree: iteration, integral, number of evaluations, last pivot error, rank\n";
            for (auto i=0u; i<5; i++){
                ci.iterate();
                auto integ = ci.tt.sum() * grid.deltaVolume;
                std::cout << std::setprecision(14) << i << " " << integ << " " << ci.f.nEval() << " " << ci.pivotError.back() << " " << ci.pivotError.size() << std::endl;
            }
            std::cout << "# oscillatory integral on tree: rank, pivot error\n";
            for (auto i=0u; i<ci.pivotError.size(); i++)
                std::cout << std::setprecision(14) << i << " " << ci.pivotError[i] << endl;
        }
    }
}
