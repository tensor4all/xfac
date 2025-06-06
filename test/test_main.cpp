#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include <armadillo>


TEST_CASE("Test arma") {
    using namespace arma;
    mat A(120,10000,fill::randn), L, U, P;
    lu(L, U, P, A);

    REQUIRE(norm(P*A-L*U)<1e-14*norm(A));
    std::cout << norm(L.diag()-1.0) << std::endl;

    auto D=U.diag().eval();
    L = L*arma::diagmat(D);
    U = arma::diagmat(1.0/D)*U;

    REQUIRE(norm(P*A-L*U)<1e-14*norm(A));
    std::cout << norm(U.diag()-1.0) << std::endl;
}
