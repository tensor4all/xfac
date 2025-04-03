#include<catch2/catch.hpp>

#include<iostream>
#include <set>
#include "xfac/tree/tensor_tree.h"
#include "xfac/tree/tensor_tree_ci.h"

using namespace std;
using namespace xfac;

using cmpx=std::complex<double>;



TEST_CASE( "Test tensor tree" )
{

    SECTION( "cos1d dmrg2" )
    {
        int nBit = 25;
        int dim=1;
        grid::Quantics grid(0., 1., nBit, dim);

        function func=[&](vector<double> const& x) {return cos(x[0]);};
        function tfunc = [&](vector<int> xi){ return func(grid.id_to_coord(xi));};

        auto tree = makeTuckerTree(dim, nBit);

        auto ci=TensorTreeCI<double>(tfunc, tree, grid.tensorDims(), {.pivot1=vector(grid.tensorLen, 0)});
        ci.iterate(1);

        vector<double> x = {0.2};
        std::cout << "res= " << ci.tt.eval(grid.coord_to_id(x)) << " , res_ref= " << func(x) <<  "\n";
    }

    SECTION( "cos1d" )
    {
        int nBit = 25;
        int dim=1;
        grid::Quantics grid(0., 1., nBit, dim);

        function func=[&](vector<double> const& x) {return cos(x[0]);};
        function tfunc = [&](vector<int> xi){ return func(grid.id_to_coord(xi));};

        auto tree = makeTuckerTree(dim, nBit);

        auto ci=TensorTreeCI<double>(tfunc, tree, grid.tensorDims(), {.pivot1=vector(grid.tensorLen, 0)});
        ci.addPivotsAllBonds({vector(grid.tensorLen, 1)});
        ci.iterate(1,0);

        vector<double> x = {0.2};
        assert ( abs(ci.tt.eval(grid.coord_to_id(x)) - func(x)) <= 1e-5 );
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
        ci.iterate(1,0);

        vector<double> x = {0.3, 0.2, 0.7};
        assert ( abs(ci.tt.eval(grid.coord_to_id(x)) - func(x)) <= 1e-5 );
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
        ci.iterate(1,0);

        vector<double> x = {0.3, 0.2, 0.7};
        assert ( abs(ci.tt.eval(grid.coord_to_id(x)) - func(x)) <= 1e-5 );
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
        ci.iterate(1,0);

        vector<double> x = {0.3, 0.2, 0.7};
        assert ( abs(ci.tt.eval(grid.coord_to_id(x)) - func(x)) <= 1e-5 );
    }


    SECTION( "reshape_cube2" )
    {
        double abstol = 1e-12;
        arma::Cube<double> A(2, 3, 4, arma::fill::randu);

        {  // reshape 0.th element A(i,j,l) -> A(l,i,j)
            auto B = reshape_cube2(A, 0);
            for (auto i=0u; i<A.n_rows; i++){
                for (auto j=0u; j<A.n_cols; j++){
                    for (auto l=0u; l<A.n_slices; l++){
                        //std::cout << "ijl= " << i << " " << j << " " << l << " "<< A(i, j, l) << " "<<  B(l, i, j)<<"\n";
                        assert(abs(A(i, j, l) - B(l, i, j)) <= abstol);
                    }
                }
            }
        }

        {  // reshape 1.th element A(i,j,l) -> A(i,l,j)
            auto B = reshape_cube2(A, 1);
            for (auto i=0u; i<A.n_rows; i++){
                for (auto j=0u; j<A.n_cols; j++){
                    for (auto l=0u; l<A.n_slices; l++){
                        assert(abs(A(i, j, l) - B(i, l, j)) <= abstol);
                    }
                }
            }
        }
        {  // reshape 2.th element A(i,j,l) -> A(i,j,l)
            auto B = reshape_cube2(A, 2);
            for (auto i=0u; i<A.n_rows; i++){
                for (auto j=0u; j<A.n_cols; j++){
                    for (auto l=0u; l<A.n_slices; l++){
                        assert(abs(A(i, j, l) - B(i, j, l)) <= abstol);
                    }
                }
            }
        }
    }

    SECTION( "reshape_cube" )
    {
        double abstol = 1e-12;
        arma::Cube<double> A(2, 3, 4, arma::fill::randu);

        {  // reshape 0.th element A(i,j,l) -> A(l,j,i)
            auto B = reshape_cube(A, 0);
            for (auto i=0u; i<A.n_rows; i++){
                for (auto j=0u; j<A.n_cols; j++){
                    for (auto l=0u; l<A.n_slices; l++){
                        //std::cout << "ijl= " << i << " " << j << " " << l << " "<< A(i, j, l) << " "<<  B(l,j,i)<<"\n";
                        assert(abs(A(i, j, l) - B(l, j, i)) <= abstol);
                    }
                }
            }
        }

        {  // reshape 1.th element A(i,j,l) -> A(i,l,j)
            auto B = reshape_cube(A, 1);
            for (auto i=0u; i<A.n_rows; i++){
                for (auto j=0u; j<A.n_cols; j++){
                    for (auto l=0u; l<A.n_slices; l++){
                        assert(abs(A(i, j, l) - B(i, l, j)) <= abstol);
                    }
                }
            }
        }
        {  // reshape 2.th element A(i,j,l) -> A(i,j,l)
            auto B = reshape_cube(A, 2);
            for (auto i=0u; i<A.n_rows; i++){
                for (auto j=0u; j<A.n_cols; j++){
                    for (auto l=0u; l<A.n_slices; l++){
                        assert(abs(A(i, j, l) - B(i, j, l)) <= abstol);
                    }
                }
            }
        }
    }


    SECTION( "cubeToMat" )
    {
        double abstol = 1e-12;
        arma::Cube<double> A(2, 3, 4, arma::fill::randu);

        {  // reshape 0.th element, cube as a matrix B(jl,i)=A(i,j,l)
            arma::Mat<double> B = cubeToMat(A, 0);
            for (auto i=0u; i<A.n_rows; i++){
                for (auto j=0u; j<A.n_cols; j++){
                    for (auto l=0u; l<A.n_slices; l++){
                        auto jl = j + l * A.n_cols;
                        assert(abs(A(i, j, l) - B(jl, i)) <= abstol);
                    }
                }
            }
        }
        {  // reshape 1.th element, cube as a matrix B(il,j)=A(i,j,l)
            arma::Mat<double> B = cubeToMat(A, 1);
            for (auto i=0u; i<A.n_rows; i++){
                for (auto j=0u; j<A.n_cols; j++){
                    for (auto l=0u; l<A.n_slices; l++){
                        auto il = i + l * A.n_rows;
                        assert(abs(A(i, j, l) - B(il, j)) <= abstol);
                    }
                }
            }
        }
        {  // reshape 2.th element, cube as a matrix B(ij,l)=A(i,j,l)
            arma::Mat<double> B = cubeToMat(A, 2);
            for (auto i=0u; i<A.n_rows; i++){
                for (auto j=0u; j<A.n_cols; j++){
                    for (auto l=0u; l<A.n_slices; l++){
                        auto ij = i + j * A.n_rows;
                        assert(abs(A(i, j, l) - B(ij, l)) <= abstol);
                    }
                }
            }
        }
    }

    SECTION( "cube_vec" )
    {
        double abstol = 1e-12;

        {  // contract 0.th element
            arma::Cube<cmpx> A(2, 3, 4, arma::fill::randu);
            arma::Col<cmpx> B(2, arma::fill::randu);
            arma::Cube<cmpx> C = cube_vec(A, B, 0);
            for (auto i=0u; i<A.n_cols; i++){
                for (auto j=0u; j<A.n_slices; j++){
                    cmpx sum = 0.;
                    for (auto k=0u; k<B.n_elem; k++){
                        sum += A(k, i, j) * B(k);
                    }
                    assert(abs(C(0, i, j) - sum) <= abstol);
                }
            }
        }
        {  // contract 1.th element
            arma::Cube<cmpx> A(3, 2, 4, arma::fill::randu);
            arma::Col<cmpx> B(2, arma::fill::randu);
            arma::Cube<cmpx> C = cube_vec(A, B, 1);

            for (auto i=0u; i<A.n_cols; i++){
                for (auto j=0u; j<A.n_slices; j++){
                    cmpx sum = 0.;
                    for (auto k=0u; k<B.n_elem; k++){
                        sum += A(i, k, j) * B(k);
                    }
                    assert(abs(C(i, 0, j) - sum) <= abstol);
                }
            }
        }
        {  // contract 2.th element
            arma::Cube<cmpx> A(3, 4, 2, arma::fill::randu);
            arma::Col<cmpx> B(2, arma::fill::randu);
            arma::Cube<cmpx> C = cube_vec(A, B, 2);

            for (auto i=0u; i<A.n_rows; i++){
                for (auto j=0u; j<A.n_cols; j++){
                    cmpx sum = 0.;
                    for (auto k=0u; k<B.n_elem; k++){
                        sum += A(i, j, k) * B(k);
                    }
                    assert(abs(C(i, j, 0) - sum) <= abstol);
                }
            }
        }
    }

    SECTION( "cube_mat" )
    {
        double abstol = 1e-12;

        {  // contract 0.th element
            arma::Cube<cmpx> A(2, 3, 4, arma::fill::randu);
            arma::Mat<cmpx> B(2, 5, arma::fill::randu);
            arma::Cube<cmpx> C = cube_mat(A, B, 0);
            for (auto i=0u; i<B.n_cols; i++){
                for (auto j=0u; j<A.n_cols; j++){
                    for (auto l=0u; l<A.n_slices; l++){
                        cmpx sum = 0.;
                        for (auto k=0u; k<B.n_rows; k++){
                            sum += A(k, j, l) * B(k, i);
                        }
                        assert(abs(C(i, j, l) - sum) <= abstol);
                    }
                }
            }
        }
        {  // contract 1.th element
            arma::Cube<cmpx> A(3, 2, 4, arma::fill::randu);
            arma::Mat<cmpx> B(2, 5, arma::fill::randu);
            arma::Cube<cmpx> C = cube_mat(A, B, 1);
            for (auto i=0u; i<A.n_rows; i++){
                for (auto j=0u; j<B.n_cols; j++){
                    for (auto l=0u; l<A.n_slices; l++){
                        cmpx sum = 0.;
                        for (auto k=0u; k<B.n_rows; k++){
                            sum += A(i, k, l) * B(k, j);
                        }
                        assert(abs(C(i, j, l) - sum) <= abstol);
                    }
                }
            }
        }
        {  // contract 2.th element
            arma::Cube<cmpx> A(4, 3, 2, arma::fill::randu);
            arma::Mat<cmpx> B(2, 5, arma::fill::randu);
            arma::Cube<cmpx> C = cube_mat(A, B, 2);
            for (auto i=0u; i<A.n_rows; i++){
                for (auto j=0u; j<A.n_cols; j++){
                    for (auto l=0u; l<B.n_cols; l++){
                        cmpx sum = 0.;
                        for (auto k=0u; k<B.n_rows; k++){
                            sum += A(i, j, k) * B(k, l);
                        }
                        assert(abs(C(i, j, l) - sum) <= abstol);
                    }
                }
            }
        }
    }

    SECTION( "mat_cube" )
    {
        double abstol = 1e-12;

        {  // contract 0.th element
            arma::Cube<cmpx> A(2, 3, 4, arma::fill::randu);
            arma::Mat<cmpx> B(5, 2, arma::fill::randu);
            arma::Cube<cmpx> C = mat_cube(B, A, 0);
            for (auto i=0u; i<B.n_rows; i++){
                for (auto j=0u; j<A.n_cols; j++){
                    for (auto l=0u; l<A.n_slices; l++){
                        cmpx sum = 0.;
                        for (auto k=0u; k<B.n_cols; k++){
                            sum += B(i, k) * A(k, j, l);
                        }
                        assert(abs(C(i, j, l) - sum) <= abstol);
                    }
                }
            }
        }
        {  // contract 1.th element
            arma::Cube<cmpx> A(3, 2, 4, arma::fill::randu);
            arma::Mat<cmpx> B(5, 2, arma::fill::randu);
            arma::Cube<cmpx> C = mat_cube(B, A, 1);
            for (auto i=0u; i<A.n_rows; i++){
                for (auto j=0u; j<B.n_rows; j++){
                    for (auto l=0u; l<A.n_slices; l++){
                        cmpx sum = 0.;
                        for (auto k=0u; k<B.n_cols; k++){
                            sum += B(j, k) * A(i, k, l);
                        }
                        assert(abs(C(i, j, l) - sum) <= abstol);
                    }
                }
            }
        }

        {  // contract 2.th element
            arma::Cube<cmpx> A(4, 3, 2, arma::fill::randu);
            arma::Mat<cmpx> B(5, 2, arma::fill::randu);
            arma::Cube<cmpx> C = mat_cube(B, A, 2);
            for (auto i=0u; i<A.n_rows; i++){
                for (auto j=0u; j<A.n_cols; j++){
                    for (auto l=0u; l<B.n_rows; l++){
                        cmpx sum = 0.;
                        for (auto k=0u; k<B.n_cols; k++){
                            sum += B(l, k) * A(i, j, k);
                        }
                        assert(abs(C(i, j, l) - sum) <= abstol);
                    }
                }
            }
        }
    }
}
