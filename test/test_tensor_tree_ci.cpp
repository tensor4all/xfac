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

    SECTION( "exp1d" )
    {
                int nBit = 3;
                int dim=1;
                grid::Quantics grid(0., 1., nBit, dim);

                function func=[&](vector<double> const& x) {return exp(x[0]);};
                function tfunc = [&](vector<int> xi){ return func(grid.id_to_coord(xi));};

                auto tree = makeTuckerTree(dim, nBit);

                auto ci=TensorTreeCI<double>(tfunc, tree, grid.tensorDims(), {.pivot1=vector(grid.tensorLen, 0)});

                vector<double> x = {0.2};
                std::cout << "res= " << ci.tt.eval(grid.coord_to_id(x)) << " , res_ref= " << func(x) <<  "\n";

    }

    SECTION( "exp" )
    {
        int nBit = 10;
        int dim=3;
        long count=0;
        grid::Quantics grid(0., 1., nBit, dim);

        function func=[&](vector<double> const& x) {
            count++;
            return exp(x[0] + x[1] + x[2]);
        };

        function tfunc = [&](vector<int> xi){ return func(grid.id_to_coord(xi));};

        auto tree = makeTuckerTree(dim, nBit);

        auto ci=TensorTreeCI<double>(tfunc, tree, grid.tensorDims(), {.pivot1=vector(grid.tensorLen, 0)});

        vector<double> x = {0.2, 0.2, 0.2};
        std::cout << "res= " << ci.tt.eval(grid.coord_to_id(x)) << " , res_ref= " << func(x) <<  "\n";

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
