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
        int nBit = 20;
        int dim = 4;
        grid::Quantics grid(0., 1., nBit, dim);

        function f=[&](vector<double> const& x) {
            auto sum=accumulate(x.begin(), x.end(), 0.0);
            return cos(sum) + 0.5 * cos(4*x[0] + x[1]) + 0.2 * sin(sum * sum);};
        function tfunc = [&](vector<int> xi){ return f(grid.id_to_coord(xi));};

        auto tree = makeTuckerTree(dim, nBit);

        vector<double> x0 = {0.1, 0.3, 0.01, 0.9};

        SECTION( "compression" )
        {
            auto ci=TensorTreeCI<cmpx>(tfunc, tree, grid.tensorDims(), {.bondDim=120});
            ci.iterate(3);

            // test function interpolation
            REQUIRE ( abs(ci.tt.eval(grid.coord_to_id(x0)) - f(x0)) < 1e-5 );

            SECTION("SVD") {
                auto mps=ci.tt;
                mps.compressSVD();
                REQUIRE( abs(mps.eval(grid.coord_to_id(x0))-f(x0)) < 1e-5 );
            }

            SECTION("LU") {
                auto mps=ci.tt;
                mps.compressLU();
                REQUIRE( abs(mps.eval(grid.coord_to_id(x0))-f(x0)) < 1e-5 );
            }

            SECTION("CI") {
                auto mps=ci.tt;
                mps.compressCI();
                REQUIRE( abs(mps.eval(grid.coord_to_id(x0))-f(x0)) < 1e-5 );
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
                p.fullPiv=fullPiv;
                p.cond=tCond;

                auto ci=TensorTreeCI<cmpx>(tfunc, tree, grid.tensorDims(), p);
                ci.iterate(3);

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

TEST_CASE( "tree overlap")
{
    auto FF1=[](const double dim, vector<double> const& idx, const double fac) { //OBC
        int len = idx.size();
        arma::vec vecX(len); // x = {1,...,N}/(N+1)
        transform(idx.begin(), idx.end(), vecX.begin(), [&dim](double val) {
            return (val+1)/(dim+1);
        });

        arma::mat matrix(len,len);
        for (int ii=0; ii<len; ii++) {//position
            for (int jj=0; jj<len; jj++) {//states
                matrix(ii,jj) = sqrt(2/(dim+1)) * sin(M_PI * vecX(ii) * (jj+1)) * fac;
            }
        }
        return det(matrix);
    };

    int  len = 4; // # of branch
    int  dim = 8; // Dim = 2^bit
    int  bit = 3; //

    int  minD = 20;
    int  incD = 2;
    int  maxD = 60;

    // FUN
    grid::Quantics grid{0., (double)dim, bit, len};
    long count = 0;
    double fac = 1;
    auto fun = [=,&dim,&count,&fac](vector<int> const& id) {
        auto config = grid.id_to_coord(id);
        for (int ii = 0; ii < config.size()-1; ii++) {
            if (config[ii] >= config[ii+1]){
                return 1E-30;
            }
        }
        count++;
        return FF1(dim, config, fac) + 1E-30;
    };

    // SEED
    double maxF = 0.0;
    vector<vector<int>> seed;
    for (int delta=1; delta<=floor(dim/len); ++delta) {
        for (int jj=0; delta*(len-1)+jj < dim; ++jj) {

            vector<double> init(len);
            fill(init.begin(), init.end(), delta);
            init.at(0) = jj;

            partial_sum(init.begin(), init.end(), init.begin(), plus<double>());
            auto id = grid.coord_to_id(init);

            double val = fun(id);
            seed.push_back(id);
            if (abs(val) > maxF)
                maxF = abs(val);
        }
    }

    // TCI INIT
    TensorCIParam pp;
    pp.pivot1 = seed.back();
    seed.pop_back();
    pp.bondDim = minD;
    pp.reltol = 1e-20;
    pp.fullPiv=true;

    auto tree = makeTuckerTree(len, bit);
    auto ci = TensorTreeCI<double>(fun, tree, grid.tensorDims(), pp);
    if (seed.size()>0) {
        ci.addPivotsAllBonds(seed);
    }
    ci.iterate(20,2);
    cout << "ITER "
         << setw(10) << ci.param.bondDim << " "
         << setw(10) << scientific << setprecision(2) << ci.pivotError.front() << " "
         << setw(10) << scientific << setprecision(2) << ci.pivotError.back() << " "
         << setw(10) << scientific << setprecision(2) << ci.pivotError.back()/ci.pivotError.front() << " ";

    // THIS SEEMS NOT WORKING
    cout   << setw(20) << scientific << setprecision(8) << ci.tt.norm2();
    cout << endl;

    // I check norm2 element by element
    double ERR = 0.0;
    for (int ii=0; ii<dim; ii++) {
        for (int jj=ii; jj<dim; jj++) {
            for (int kk=jj; kk<dim; kk++) {
                for (int ll=kk; ll<dim; ll++) {

                    vector<double> chk;
                    chk.push_back((double)ii);
                    chk.push_back((double)jj);
                    chk.push_back((double)kk);
                    chk.push_back((double)ll);

                    auto id = grid.coord_to_id(chk);
                    ERR += pow(ci.tt.eval(id), 2);
                }
            }
        }
    }
    cout << "NORM2 = " << ERR << endl;
    REQUIRE(std::abs(ERR-ci.tt.norm2())<1e-5);
}

TEST_CASE( "bitset_error")
{
    // Test to show the bitset error by Julian Thoenniss
    cout << "=== Testing TensorTreeCI with unfused quantics ===" << endl;
    cout << "Demonstrating TensorTreeCI bug: generates non-binary indices for unfused quantics" << endl;
    cout << endl;

    int dim = 3;
    int nBit = 8;
    double a = -40;
    double b = 40;

    auto tree = makeTuckerTree(dim, nBit);
    auto grid = grid::Quantics(a, b, nBit, dim);

    // Global counter to track function calls
    int function_call_count = 0;

    // Simple test function that triggers the TensorTreeCI bug
    auto test_function = [&function_call_count, &grid] (const vector<int>& sigma) -> std::complex<double> {
        function_call_count++;

        // *** BUG DETECTION: Check for invalid indices ***
        bool has_invalid = std::any_of(sigma.begin(), sigma.end(), [](int val) { return val > 1; });

        // Print some sample indices to see what we're getting
        if (function_call_count <= 10 || function_call_count % 200 == 0) {
            cout << "Call #" << function_call_count << ": sigma = [";
            for (size_t i = 0; i < min(sigma.size(), size_t(10)); ++i) {
                cout << sigma[i];
                if (i < min(sigma.size() - 1, size_t(9))) cout << ", ";
            }
            if (sigma.size() > 10) cout << ", ...";
            cout << "] (length=" << sigma.size() << ")";
            if (has_invalid) cout << " *** HAS VALUES > 1 ***";
            cout << endl;
        }

        if (has_invalid) {
            cout << "*** BUG DETECTED (call #" << function_call_count << "): Non-binary index found in sigma = [";
            for (size_t i = 0; i < min(sigma.size(), size_t(10)); ++i) {
                cout << sigma[i];
                if (i < min(sigma.size() - 1, size_t(9))) cout << ", ";
            }
            if (sigma.size() > 10) cout << ", ...";
            cout << "] (length=" << sigma.size() << ")" << endl;

            // Print which positions have values > 1
            cout << "    Positions with values > 1: ";
            int count = 0;
            for (size_t i = 0; i < sigma.size() && count < 5; ++i) {
                if (sigma[i] > 1) {
                    cout << "pos=" << i << ":val=" << sigma[i] << " ";
                    count++;
                }
            }
            if (count >= 5) cout << "...";
            cout << endl;
        }

        // Use xfac Quantics grid (unfused, domain [-40, 40])

        // Convert quantics indices to coordinates using xfac grid
        vector<double> coords = grid.id_to_coord(sigma);

        // Create simple complex evaluation (abstract function)
        vector<complex<double>> z(3);
        for (int i = 0; i < 3; ++i) {
            z[i] = complex<double>(coords[i], 5.0/3.0);  // Simple imaginary shift
        }

        // Simple abstract computation that returns a complex value
        complex<double> result(0.0, 0.0);
        for (int i = 0; i < 3; ++i) {
            result += exp(-0.5 * z[i] * z[i]);  // Simple Gaussian-like function
        }

        return result;
    };

    // Reset function call counter
    function_call_count = 0;

    try {
        // Create TensorTreeCI - this triggers the bug
        auto ci = TensorTreeCI<complex<double>>(test_function, tree, grid.tensorDims(), {.pivot1=grid.coord_to_id(vector(dim, 0.))});

        cout << "TensorTreeCI created. Function calls during construction: " << function_call_count << endl;

        // Perform one iteration
        cout << "Performing TCI iteration..." << endl;
        int initial_calls = function_call_count;
        ci.iterate(1);
        cout << "Iteration completed. Additional function calls: " << (function_call_count - initial_calls) << endl;
        cout << "Total function calls: " << function_call_count << endl;

        cout << endl << "=== Test Summary ===" << endl;
        cout << "If you see '*** BUG DETECTED' messages above, the bug has been reproduced!" << endl;
        cout << "TensorTreeCI generates indices with values > 1 for unfused quantics" << endl;

    } catch (const exception& e) {
        cout << "Error: " << e.what() << endl;
    }
}
