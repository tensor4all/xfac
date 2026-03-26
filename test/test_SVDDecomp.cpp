#include <catch2/catch.hpp>
#include "xfac/matrix/mat_decomp.h"

using namespace arma;
using namespace xfac;

TEST_CASE("SVDDecomp")
{
    double tol = 1e-12;

    SECTION("no reltol: full rank random matrix reconstructed exactly")
    {
        mat A(10, 8, fill::randn);
        SVDDecomp<double> sol(A, true, 0.0);
        REQUIRE((int)sol.s.size() == 8); // min(10,8)
        REQUIRE(norm(A - sol.left() * sol.right()) < tol);
    }

    SECTION("no reltol: left orthogonal")
    {
        mat A(10, 8, fill::randn);
        SVDDecomp<double> sol(A, true, 0.0);
        mat should_be_eye = sol.U.t() * sol.U;
        REQUIRE(norm(should_be_eye - eye(sol.U.n_cols, sol.U.n_cols)) < tol);
    }

    SECTION("no reltol: right orthogonal")
    {
        mat A(10, 8, fill::randn);
        SVDDecomp<double> sol(A, false, 0.0);
        mat should_be_eye = sol.V.t() * sol.V;
        REQUIRE(norm(should_be_eye - eye(sol.V.n_cols, sol.V.n_cols)) < tol);
    }

    SECTION("reltol: low rank matrix truncated to correct rank")
    {
        int n = 20, rank = 5;
        mat A = mat(n, rank, fill::randn) * mat(rank, n, fill::randn);
        SVDDecomp<double> sol(A, true, 1e-12);
        REQUIRE((int)sol.s.size() == rank);
        REQUIRE(norm(A - sol.left() * sol.right()) < tol * norm(A));
    }

    SECTION("reltol: full rank matrix")
    {
        int rank = 15;
        mat A(20, 15, fill::randn);
        SVDDecomp<double> sol(A, true, 1e-12);
        REQUIRE((int)sol.s.size() == rank);
        REQUIRE(norm(A - sol.left() * sol.right()) < tol);
    }

    SECTION("reltol: reconstruction error within tolerance")
    {
        int n = 20, rank = 5;
        mat A = mat(n, rank, fill::randn) * mat(rank, n, fill::randn);
        double reltol = 1e-6;
        SVDDecomp<double> sol(A, true, reltol);
        REQUIRE(norm(A - sol.left() * sol.right()) < reltol * norm(A));
    }

    SECTION("rankMax: caps the rank even if reltol would keep more")
    {
        int n = 20, rank = 10;
        mat A = mat(n, rank, fill::randn) * mat(rank, n, fill::randn);
        int rankMax = 4;
        SVDDecomp<double> sol(A, true, 0.0, rankMax);
        REQUIRE((int)sol.s.size() == rankMax);
    }

    // --- specific problem: exact zero singular values must be cut ---

    SECTION("exact zeros: low rank matrix with trailing zero singular values cut")
    {
        int n = 10, rank = 3;
        mat A = mat(n, rank, fill::randn) * mat(rank, n, fill::randn);
        // With reltol=0, zeros should still be cut
        SVDDecomp<double> sol(A, true, 1e-12);
        REQUIRE((int)sol.s.size() == rank);
        for (int i = 0; i < (int)sol.s.size(); i++)
            REQUIRE(sol.s[i] > 0.0);
    }

    SECTION("exact zeros: zero matrix returns rank 1 (minimum)")
    {
        mat A(10, 10, fill::zeros);
        SVDDecomp<double> sol(A, true, 1e-12);
        REQUIRE((int)sol.s.size() == 1);
    }

    SECTION("exact zeros: reconstruction still correct after cutting zeros")
    {
        int n = 10, rank = 3;
        mat A = mat(n, rank, fill::randn) * mat(rank, n, fill::randn);
        SVDDecomp<double> sol(A, true, 0.0);
        REQUIRE(norm(A - sol.left() * sol.right()) < tol * norm(A));
    }
}
