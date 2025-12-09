#include<catch2/catch.hpp>
#include "xfac/tensor/tensor_ci.h"
#include "xfac/grid.h"
#include "xfac/matrix/mat_decomp.h"

using namespace arma;
using namespace xfac;

using cmpx=std::complex<double>;
using uint = unsigned int;

TEST_CASE( "Test MatrixCI" )
{
    // Checks that CI is exact for matrix of finite rank.
    SECTION( "fixed rank" )
    {
        int n=100, rank=50;
        mat A = mat(n,rank,fill::randn) * mat(rank,n,fill::randn); // mat is armadillo matrix
        function myf=[&](vector<int> const& id){ return A(id[0],id[1]); };
        vector<int> dims={int(A.n_rows), int(A.n_cols)};
        auto ci=TensorCI1<double>( myf, dims, {.nIter=rank+1}  );
        REQUIRE( ci.pivotError.back() <= 1e-13*norm(A) );
        REQUIRE( ci.trueError() <= 1e-13*norm(A) );
        int i=rand()%A.n_rows, j=rand()%A.n_cols;
        REQUIRE( std::abs(ci.get_TensorTrain().eval({i, j})-A(i, j)) < 1e-10 );
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
            TensorCI1Param p;
            p.fullPiv=true;
            auto ci=TensorCI1<cmpx>(myTf, dims, p);
//            cout<<"rank pivotError norm(A-A~)\n";
            for(int r=1;r<=rank;r++) {
                ci.iterate();
//                cout<<r<<" "<<ci.lastSweepPivotError()<<" "<<ci.trueError()<<endl;
            }
            REQUIRE( ci.trueError() <= 1e-12*norm(A) );
            REQUIRE( ci.pivotError.back() <= 1e-12*norm(A) );
            int i=rand()%A.n_rows, j=rand()%A.n_cols;
            REQUIRE( abs(ci.get_TensorTrain().eval({i, j})-A(i, j)) < 1e-10 );
        }

        SECTION( "directly" )
        {
            auto myTf=[&](vector<double> xs){return myf(xs[0],xs[1]);};
            auto ci=CTensorCI1<cmpx,double>(myTf, vector(2,xi), {.nIter=rank});
            REQUIRE( ci.pivotError.back() <= 1e-10 );
            double x=1.0*rand()/RAND_MAX, y=1.0*rand()/RAND_MAX;
            REQUIRE( std::abs(ci.get_CTensorTrain().eval({x, y})-myf(x, y))< 1e-10 );
        }
    }
}

//------------------------- Here start the tests of internal classes ---------------------

TEST_CASE("Test CrossData")
{
    mat A(100,120,fill::randu);
    A/=norm(A);
    double tol=1e-12;
    uvec I,J;
    while(I.size()< 0.2*A.n_rows) // random pivots
    {
        uint i=rand()%A.n_rows, j=rand()%A.n_cols;
        double err=std::abs((A(i,j)-A.submat(uvec({i}),J) * A.submat(I,J).i() * A.submat(I,uvec({j})) ).eval()(0,0));
        if (err>tol) {
            I=join_cols(I,uvec({i}));
            J=join_cols(J,uvec({j}));
        }
    }
    SECTION("interpolation") {
        auto cross=CrossData<double>(I, J, A.cols(J), A.rows(I));
        SECTION("direct formula") {
            mat Aci=A.cols(J)*A.submat(I,J).i()*A.rows(I) ;
            REQUIRE(norm(Aci.rows(I)-A.rows(I))< tol);
            REQUIRE(norm(Aci.cols(J)-A.cols(J))< tol);
        }
        SECTION("at cross") {
            mat Across=cross.mat();
            REQUIRE( norm(Across.rows(I)-A.rows(I))< tol );
            REQUIRE( norm(Across.cols(J)-A.cols(J))< tol );
        }
        SECTION("each row/col") {
            for(auto i:I)
                REQUIRE( norm(conv_to<rowvec>::from(cross.row(i))-A.row(i)) < tol );
            for(auto j:J)
                REQUIRE( norm(conv_to<colvec>::from(cross.col(j))-A.col(j)) < tol );
        }
    }
    SECTION("insert") {
        auto cross=CrossData<double>(A.n_rows, A.n_cols);
        for(auto c=0u;c<I.size();c++)
            cross.addPivot(I[c], J[c], MatDense<double>(A));
        SECTION("well copied") {
            REQUIRE(norm(cross.R-A.rows(I))<tol);
            REQUIRE(norm(cross.C-A.cols(J))<tol);
        }
        SECTION("interpolation at cross") {
            mat Across=cross.mat();
            REQUIRE( norm(Across.rows(I)-A.rows(I))< tol );
            REQUIRE( norm(Across.cols(J)-A.cols(J))< tol );
        }
    }
}

TEST_CASE("rrlu")
{
    double tol=1e-12;
    SECTION("2x3 matrix") {
        mat A={{1,2,3},{4,5,6}};

        SECTION("horizontal matrix") {
            for(auto isLeft:{0,1}) {
                RRLUDecomp<double> sol(A,isLeft,tol);
                REQUIRE(arma::norm(A-sol.left()*sol.right())<tol);
            }
        }

        SECTION("vertical matrix") {
            for(auto isLeft:{0,1}) {
                RRLUDecomp<double> sol(A.t(),isLeft,tol);
                REQUIRE(arma::norm(A.t()-sol.left()*sol.right())<tol);
            }
        }

        // same as before, but now with a condition matrix (basically a boolean mask)
        // pivots should be selected only where the corresponding elements in the condition matrix are true
        arma::Mat<unsigned int> cond={{0,1,1},{1,1,0}};

        SECTION("horizontal matrix cond") {
            for(auto isLeft:{0, 1}) {
                RRLUDecomp<double> sol(A,cond,isLeft,tol);
                REQUIRE(arma::norm(A-sol.left()*sol.right())<tol);
                for (auto i=0; i<sol.row_pivots().size(); i++){
                    auto row = sol.row_pivots().at(i);
                    auto col = sol.col_pivots().at(i);
                    REQUIRE(cond(row, col) == 1);
                }
            }
        }

        SECTION("vertical matrix cond") {
            for(auto isLeft:{0,1}) {
                RRLUDecomp<double> sol(A.t(),cond.t(),isLeft,tol);
                REQUIRE(arma::norm(A.t()-sol.left()*sol.right())<tol);
                for (auto i=0; i<sol.row_pivots().size(); i++){
                    auto row = sol.row_pivots().at(i);
                    auto col = sol.col_pivots().at(i);
                    REQUIRE(cond(col, row) == 1);
                }
            }
        }
    }

    SECTION("from MPO") {
        mat A(4,7, fill::zeros);
        A.row(0).fill(1);
        A.row(3).fill(1);
        A(0,0)=-1;
        RRLUDecomp<double> sol(A);
        REQUIRE(arma::norm(A-sol.left()*sol.right())<tol);
    }
}


TEST_CASE("test setRows")
{
    arma::Cube<double> Acube(7, 3, 20, fill::randu);
    auto A = cube_as_matrix2(Acube);  // Acube(i,j,k) -> A(ij,k)
    //A/=norm(A);
    double tol=1e-12;
    uvec I,J;
    while(I.size()< 5) // random pivots
    {
        uint i=rand()%A.n_rows, j=rand()%A.n_cols;
        double err=std::abs((A(i,j)-A.submat(uvec({i}),J) * A.submat(I,J).i() * A.submat(I,uvec({j})) ).eval()(0,0));
        if (err>tol) {
            I=join_cols(I,uvec({i}));
            J=join_cols(J,uvec({j}));
        }
    }

    AdaptiveLU<double> sol(A.n_rows, A.n_cols);

    arma::Mat<double> C=A.cols(J);
    arma::Mat<double> R=A.rows(I);

    for(auto k=0u; k<I.size(); k++){
        sol.addPivotRow(I[k], R.row(k));
        sol.addPivotCol(J[k], C.col(k));
    }

    arma::Mat<double> Across=sol.mat();
    REQUIRE( norm(Across.rows(I)-A.rows(I))< tol );
    REQUIRE( norm(Across.cols(J)-A.cols(J))< tol );

    std::cout << "error after adding 5 pivots= " << arma::norm(sol.mat() - A) << std::endl;

    // Create a new cube Bcube with n rows, the number of cols and slices should match Acube.
    // Append Bcube after the last rows of Acube, such that the Acube increses the number of rows by n (number of rows of Bcube).
    int n = 2;
    arma::Cube<double> Bcube(n, Acube.n_cols, Acube.n_slices, arma::fill::randu);

    auto A_orig = Acube;
    Acube.insert_rows( Acube.n_rows, Bcube );

    A = cube_as_matrix2(Acube);

    // determine the position vector pos
    std::vector<int> pos(A_orig.n_rows * A_orig.n_cols);
    for (auto j = 0, iter = 0; j < A_orig.n_cols; j++){  // column major (fortran order)
        for (auto i = 0; i < A_orig.n_rows; i++, iter++){
            pos.at(iter) = i + j * (A_orig.n_rows + Bcube.n_rows);
        }
    }

    // check that the position vector pos is correct
    for (auto k = 0; k < A_orig.n_slices; k++){ // column major (fortran order)
        for (auto j = 0, iter = 0; j < A_orig.n_cols; j++){
            for (auto i = 0; i < A_orig.n_rows; i++, iter++){
                //std::cout << "A_orig(i, j, k)= " << A_orig(i, j, k) << " A(pos.at(iter), k)=" << A(pos.at(iter), k) << ", i= " << i << ", j= " << j << ", k= " << k << ", pos= " << pos.at(iter) << std::endl;
                REQUIRE( A_orig(i, j, k) ==  A(pos.at(iter), k) );
            }
        }
    }

    auto Bmat = cube_as_matrix2(Bcube);

    sol.setRows(A, pos);  // all positions of the old rows in the new rows, len P = L, Pc size = C.n_rows

    double error = arma::norm(sol.mat() - A);
    std::cout << "error after calling setRow= " << error << std::endl;

    {
        auto npiv = I.size();
        while(I.size()< npiv + 3) // add three more random pivots
        {
            uint i=rand()%A.n_rows, j=rand()%A.n_cols;
            double err=std::abs((A(i,j)-A.submat(uvec({i}),J) * A.submat(I,J).i() * A.submat(I,uvec({j})) ).eval()(0,0));
            if (err>tol) {
                I=join_cols(I,uvec({i}));
                J=join_cols(J,uvec({j}));
            }
        }

        arma::Mat<double> C=A.cols(J);
        arma::Mat<double> R=A.rows(I);

        for(auto k=npiv; k<I.size(); k++){
            sol.addPivotRow(I[k], R.row(k));
            sol.addPivotCol(J[k], C.col(k));
        }

        std::cout << "error after 3 more pivots= " << arma::norm(sol.mat() - A) << std::endl;

        //arma::Mat<double> Across=sol.mat();
        //REQUIRE( norm(Across.rows(I)-A.rows(I))< tol );
        //REQUIRE( norm(Across.cols(J)-A.cols(J))< tol );

        double error2 = arma::norm(sol.mat() - A);
        //REQUIRE( error2 <=  error );

    }
}

TEST_CASE("cur")
{
    double tol=1e-12;
    SECTION("2x3 matrix") {
        mat A={{1,2,3},{4,5,6}};

        SECTION("horizontal matrix") {
            for(auto isLeft:{0,1}) {
                CURDecomp<double> sol(A,isLeft,tol);
                REQUIRE(arma::norm(A-sol.left()*sol.right())<tol);
            }
        }

        SECTION("vertical matrix") {
            for(auto isLeft:{0,1}) {
                CURDecomp<double> sol(A.t(),isLeft,tol);
                REQUIRE(arma::norm(A.t()-sol.left()*sol.right())<tol);
            }
        }
    }

    SECTION("from MPO") {
        mat A(4,7, fill::zeros);
        A.row(0).fill(1);
        A.row(3).fill(1);
        A(0,0)=-1;
        CURDecomp<double> sol(A);
        REQUIRE(arma::norm(A-sol.left()*sol.right())<tol);
    }
}


TEST_CASE("arrlu")
{
    double tol=1e-12;
    auto to_mat_fun=[](mat A) {
        auto submat=[A](vector<int> I, vector<int> J){
            return  mat {A.submat(conv_to<uvec>::from(I), conv_to<uvec>::from(J))};
        };
        return MatFun<double> {A.n_rows, A.n_cols, submat} ;
    };
    SECTION("2x3 matrix") {
        mat A={{1,2,3},{4,5,6}};


        SECTION("horizontal matrix") {
            for(auto isLeft:{0,1}) {
                ARRLUDecomp<double> sol(to_mat_fun(A),{},{},isLeft, {.reltol=tol});
                REQUIRE(arma::norm(A-sol.left()*sol.right())<tol);
            }
        }

        SECTION("vertical matrix") {
            for(auto isLeft:{0,1}) {
                ARRLUDecomp<double> sol(to_mat_fun(A.t()), {}, {}, isLeft, {.reltol=tol});
                REQUIRE(arma::norm(A.t()-sol.left()*sol.right())<tol);
            }
        }
    }

    SECTION("from MPO") {
        mat A(4,7, fill::zeros);
        A.row(0).fill(1);
        A.row(3).fill(1);
        A(0,0)=-1;
        ARRLUDecomp<double> sol(to_mat_fun(A),{},{});
        REQUIRE(arma::norm(A-sol.left()*sol.right())<tol);
    }
}
