#include<catch2/catch.hpp>

#include "xfac/tensor/cheby_tensor_ci.h"
#include "xfac/grid.h"
#include "../extern/cheby/chebyshev.hpp"

using namespace std;
using namespace xfac;

using cmpx=std::complex<double>;

inline int factorial(int n){ // factorial
    return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;}

inline int factorial2(int n){ // 2-step factorial 
    // factorial2(n) = 2 * 4 * 6 .. * n (even n), 3 * 5 * 7 .. n (odd n)
    return (n == 1 || n == 0) ? 1 : factorial2(n - 2) * n;}

TEST_CASE( "Test cheby tensor CI" )
{

    size_t dim=5;

    SECTION( "get_T onside grid" )
    {
        double abs_tol = 1e-10;

        mgrid grid;
        double a = 0;
        double b = 1;
        grid.push_back(make_tuple(a, b, 11));
        auto xi = chebyshev::cheby_multi_xi(grid);

        auto fx=[](const vector<double>& xs) {
            double x=0, y=0,c=0;
            for(auto xi:xs) {c++; x+=c*xi; y+=xi*xi/c;}
            double arg=1.0+(x+2*y+x*y)*M_PI;
            return cmpx(1+x+cos(arg),x*x+0.5*sin(arg));
        };

        auto fi=[&,xi=xi](vector<int> const& id) {
            assert(id.size()==dim);
            vector<double> xs;
            for(auto i:id) xs.push_back(xi[i]);
            return fx(xs);
        };

        // tensor parameter
        TensorCI1Param p;
        p.fullPiv=true;

        TensorCIParamCheby pc;
        pc.fullPiv=true;
        pc.env=false;

        // discrete tensor
        auto ci_d = TensorCI1<cmpx>(fi, vector(dim,(int)xi.size()), p);
        ci_d.iterate(20);

        // contineous tensor
        auto ci = CTensorCI1<cmpx,double>(fx, vector(dim, xi), p);
        ci.iterate(20);

        // cheby tensor
        auto cic_1 = CTensorCheby<cmpx,double>(fi, dim, grid, pc);
        auto cic_2 = CTensorCheby<cmpx,double>(fx, dim, grid, pc);
        cic_1.iterate(20);
        cic_2.iterate(20);

        for(size_t p=0; p<dim; p++) {
            for(size_t x_idx=0; x_idx<xi.size(); x_idx++) {
                arma::Mat<cmpx> T{ci_d.T3[p].col(x_idx)};  // equivalent to TensorCI::ci.get_T_at(p, x)
                REQUIRE( abs( ci.get_T_at(p, xi[x_idx]) - T ).max() <= abs_tol );
                REQUIRE( abs( cic_1.get_T_at(p, xi[x_idx]) - T ).max() <= abs_tol );
                REQUIRE( abs( cic_2.get_T_at(p, xi[x_idx]) - T ).max() <= abs_tol );
            }
        }
    }

    SECTION( "get_T onside grid" )
    {
        double abs_tol = 1e-10;

        mgrid grid;
        double a = 0;
        double b = 1;
        grid.push_back(make_tuple(a, b, 11));
        auto xi = chebyshev::cheby_multi_xi(grid);

        auto fx=[](const vector<double>& xs) {
            double x=0, y=0,c=0;
            for(auto xi:xs) {c++; x+=c*xi; y+=xi*xi/c;}
            double arg=1.0+(x+2*y+x*y)*M_PI;
            return cmpx(1+x+cos(arg),x*x+0.5*sin(arg));
        };

        auto fi=[&,xi=xi](vector<int> const& id) {
            assert(id.size()==dim);
            vector<double> xs;
            for(auto i:id) xs.push_back(xi[i]);
            return fx(xs);
        };

        // tensor parameter
        TensorCI1Param p;
        p.fullPiv=true;

        TensorCIParamCheby pc;
        pc.fullPiv=true;
        pc.env=false;

        // discrete tensor
        auto ci_d = TensorCI1<cmpx>(fi, vector(dim,(int)xi.size()), p);
        ci_d.iterate(20);

        // contineous tensor
        auto ci = CTensorCI1<cmpx,double>(fx, vector(dim, xi), p);
        ci.iterate(20);

        // cheby tensor
        auto cic_1 = CTensorCheby<cmpx,double>(fi, dim, grid, pc);
        auto cic_2 = CTensorCheby<cmpx,double>(fx, dim, grid, pc);
        cic_1.iterate(20);
        cic_2.iterate(20);

        for(size_t p=0; p<dim; p++) {
            for(size_t x_idx=0; x_idx<xi.size(); x_idx++) {
                arma::Mat<cmpx> T{ci_d.T3[p].col(x_idx)};  // equivalent to TensorCI::ci.get_T_at(p, x)
                REQUIRE( abs( ci.get_T_at(p, xi[x_idx]) - T ).max() <= abs_tol );
                REQUIRE( abs( cic_1.get_T_at(p, xi[x_idx]) - T ).max() <= abs_tol );
                REQUIRE( abs( cic_2.get_T_at(p, xi[x_idx]) - T ).max() <= abs_tol );
            }
        }
    }


    SECTION( "cheby tensor" )
    {
        int ninter = 15;
        double abs_tol = 1e-2;

        mgrid grid;
        double a = 0;
        double b = 1;
        grid.push_back(make_tuple(a, b, 40));
        auto xi = chebyshev::cheby_multi_xi(grid);  // put this before fi, otherwise fi binds on xi from outer scope

        vector<double> x_on = {xi[3],xi[5],xi[1],xi[5],xi[1]};  // onside gridpoint
        vector<double> x_off = {0.4, 0.2, 0.5, 0.7, 0.3};  // not on gridpoint

        auto fx=[](const vector<double>& xs) {
            double x=0, y=0,c=0;
            for(auto xi:xs) {c++; x+=c*xi; y+=xi*xi/c;}
            double arg=1.0+(x+2*y+x*y)*M_PI;
            return cmpx(1+x+cos(arg),x*x+0.5*sin(arg));
        };

        auto fi=[&,xi=xi](vector<int> const& id) {
            assert(id.size()==dim);
            vector<double> xs;
            for(auto i:id) xs.push_back(xi[i]);
            return fx(xs);
        };

        // tensor parameter
        TensorCI1Param p;
        p.fullPiv=true;

        TensorCIParamCheby pc;
        pc.fullPiv=true;
        pc.env=false;

        auto ci = CTensorCI1<cmpx,double>(fx, vector(dim, xi), p);
        ci.iterate(ninter);
        auto tt = ci.get_CTensorTrain();

        auto cic_0 = CTensorCheby<cmpx,double>(fx, dim, grid, pc);
        cic_0.iterate(ninter);

        SECTION( "eval" )
        {
            REQUIRE( abs(cic_0.eval(x_on) - tt.eval(x_on)) <= abs_tol );
            REQUIRE( abs(cic_0.eval(x_off) - tt.eval(x_off)) <= abs_tol );

            auto cic_1 = CTensorCheby<cmpx,double>(fx, vector(dim, grid), pc);
            cic_1.iterate(ninter);
            REQUIRE( abs(cic_1.eval(x_on) - tt.eval(x_on)) <= abs_tol );
            REQUIRE( abs(cic_1.eval(x_off) - tt.eval(x_off)) <= abs_tol );

            auto cic_2 = CTensorCheby<cmpx,double>(fi, dim, grid, pc);
            cic_2.iterate(ninter);
            REQUIRE( abs(cic_2.eval(x_on) - tt.eval(x_on)) <= abs_tol );
            REQUIRE( abs(cic_2.eval(x_off) - tt.eval(x_off)) <= abs_tol );

            auto cic_3 = CTensorCheby<cmpx,double>(fi, vector(dim, grid), pc);
            cic_3.iterate(ninter);
            REQUIRE( abs(cic_3.eval(x_on) - tt.eval(x_on)) <= abs_tol );
            REQUIRE( abs(cic_3.eval(x_off) - tt.eval(x_off)) <= abs_tol );
        }



        SECTION( "condition" )
        {
            TensorCIParamCheby pc;
            pc.ccond = [](vector<double> const& xv)
            {
                double sum=0;
                for (auto x:xv) sum += x;
                return sum <= 1;
            };

            auto ci = CTensorCheby<cmpx,double>(fx, dim, grid, pc);
            ci.iterate(100);
            for(auto b=0; b<ci.len()-1; b++)  // all bonds
                for(auto r=0u; r<ci.Iset[b+1].size(); r++) { // all pivots
                    MultiIndex ij=ci.Iset[b+1][r]+ci.Jset[b][r];
                    REQUIRE( ci.param.cond( vector<int>(ij.begin(),ij.end()) ) );
                }
        }

    }


    SECTION( "simplex" )
    {
        double abs_tol = 1E-9;

        //  calculate the v-simplex, which is defined as:
        //  F(x) = \int^x_0 dv_1 \int^{x - v_1}_0 dv_2 ... \int^{x - v_1 - v_2 .. - v_(n-1)}_0 dv_n f(v_1, v_2, ..., v_n)
        //  for different functions f(v_1, .., v_n)

        srand(1);  // set random seed

        // Integrate[Integrate[Integrate[ sin(v1 + v2 + v3)  , {v3, 0, t - v1 - v2}], {v2, 0, t - v1}], {v1, 0, t}]
        // = -1/2 t^2 cos(t) + t sin(t) + cos(t) - 1
        auto f0 =[](vector<double> v) -> cmpx {return sin(v[0] + v[1] + v[2]);};

        auto F0=[](double x) {return - 0.5 * x * x * cos(x) + x * sin(x) + cos(x) - 1;};

        //  Integrate[Integrate[y exp(-x x), {y, 0, t - x}], {x, 0, t}] = 1/8 (sqrt(π) (2 t^2 + 1) erf(t) + 2 (e^(-t^2) - 2) t)
        auto f1 =[](vector<double> v) -> cmpx {return v[1] * exp(- v[0] * v[0]);};
     
        auto F1=[](double x) {return (sqrt(M_PI) * (2 * x * x + 1) * erf(x) + 2 * x * (exp(- x * x) - 2) ) / 8.;};

        //  Integrate[Integrate[x y, {y, 0, t - x}], {x, 0, t}] =  t^4/24
        //  Integrate[Integrate[Integrate[x y z  , {z, 0, t - x - y}], {y, 0, t - x}], {x, 0, t}] = t^6/720
        auto f2 =[](vector<double> v) {
            double y=1.0;
            for(auto vi:v)
                y *= vi;
            return cmpx(y);
         };

        auto F22=[](double x) {int dim = 2; return pow (x, 2 * dim) / 24;}; // dim2 result
        auto F23=[](double x) {int dim = 3; return pow (x, 2 * dim) / 720;}; // dim3 result

        // simplex volume
        auto f3=[](vector<double> xs) {return 1.;};
        auto F3 = [](double x){int dim = 7; return pow (x, dim) / factorial(dim);};

        vector<tuple<function<cmpx(vector<double>)>, double, double, int, function<double(double)>>> battery;
        battery.push_back(make_tuple(f0, 0., 1.2, 3, F0));
        battery.push_back(make_tuple(f1, 0., 0.6, 2, F1));
        battery.push_back(make_tuple(f2, 0., 1.5, 2, F22));
        battery.push_back(make_tuple(f2, 0., 1.4, 3, F23));
        battery.push_back(make_tuple(f3, 0., 3., 7, F3));

        for (auto const& [f, a, b, dim, F] : battery){

            mgrid grid;
            grid.push_back(make_tuple(a, b, 12));

            auto ctf = CTensorCheby<cmpx,double>(f, dim, grid);
            ctf.iterate(15);
            // no lambda
            //ctf.prepare_simplex_v(grid);

            for (int i=0; i<5;i++){
                auto x0 = b * (double) rand() / RAND_MAX;
                auto Fx = ctf.simplex_v(x0);
                auto Fx_ref = F(x0);
                //cout<<"fx= " << Fx << " Fx_ref= " << Fx_ref << " Fx - Fx_ref= " << std::abs(Fx - Fx_ref)  << "\n";
                REQUIRE( abs(Fx - Fx_ref) <= abs_tol );
            }
        }
    }


}

