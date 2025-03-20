#include<catch2/catch.hpp>

#include<iostream>
#include<bitset>
#include <set>
#include "xfac/grid.h"
#include "xfac/tree/tensor_tree.h"
#include "xfac/tree/tensor_tree_ci.h"

using namespace std;
using namespace xfac;

using cmpx=std::complex<double>;


TEST_CASE( "Test tensor tree" )
{
    SECTION( "cube_mat" )
    {
        cout << "tensor tree ci\n";
        
        double abstol = 1e-12;
        {  // contract 0.th element
            arma::cube A(2, 3, 4, arma::fill::randu); 
            arma::mat B(2, 5, arma::fill::randu);
            arma::cube C = cube_mat(A, B, 0);
            for (auto i=0u; i<B.n_cols; i++){
                for (auto j=0u; j<A.n_cols; j++){
                    for (auto l=0u; l<A.n_slices; l++){
                        double sum = 0.;
                        for (auto k=0u; k<B.n_rows; k++){
                            sum += A(k, j, l) * B(k, i);
                        }
                        assert(abs(C(i, j, l) - sum) <= abstol);
                    }
                }
            }
        }
        {  // contract 1.th element
            arma::cube A(3, 2, 4, arma::fill::randu); 
            arma::mat B(2, 5, arma::fill::randu);
            arma::cube C = cube_mat(A, B, 1);
            for (auto i=0u; i<A.n_rows; i++){
                for (auto j=0u; j<B.n_cols; j++){
                    for (auto l=0u; l<A.n_slices; l++){
                        double sum = 0.;
                        for (auto k=0u; k<B.n_rows; k++){
                            sum += A(i, k, l) * B(k, j);
                        }
                        assert(abs(C(i, j, l) - sum) <= abstol);
                    }
                }
            }
        }
        {  // contract 2.th element
            arma::cube A(4, 3, 2, arma::fill::randu); 
            arma::mat B(2, 5, arma::fill::randu);
            arma::cube C = cube_mat(A, B, 2);
            for (auto i=0u; i<A.n_rows; i++){
                for (auto j=0u; j<A.n_cols; j++){
                    for (auto l=0u; l<B.n_cols; l++){
                        double sum = 0.;
                        for (auto k=0u; k<B.n_rows; k++){
                            sum += A(i, j, k) * B(k, l);
                        }
                        assert(abs(C(i, j, l) - sum) <= abstol);
                    }
                }
            }
        }
    }

    SECTION( "cube_mat" )
    {
        cout << "tensor tree ci\n";

        arma::Cube<double> A(5, 4, 3, arma::fill::randu);
        arma::Mat<double> C = arma::reshape( arma::Mat<double>(A.memptr(), A.n_elem, 1, false), 5*4, 3);
        //C.print("C:");
    }

}
