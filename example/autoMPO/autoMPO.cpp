#include <iomanip>
#include <iostream>
#include <vector>
#include <map>
#include <set>

#include "xfac/tensor/auto_mpo.h"


using namespace std;
using namespace xfac;
using namespace xfac::autompo;


/// Heisenberg Hamiltonian (periodic boundary condition)
PolyOp<> HeisenbergHam(int L)
{
    auto Sz=[=](int i) { return ProdOp<> {{ i%L, LocOp<> {{-1,0},{0,1}} }}; };
    auto Sp=[=](int i) { return ProdOp<> {{ i%L, LocOp<> {{0 ,0},{1,0}} }}; };
    auto Sm=[=](int i) { return ProdOp<> {{ i%L, LocOp<> {{0 ,1},{0,0}} }}; };

    PolyOp<> H;
    for(int i=0; i<L; i++) {
        H += Sz(i)*Sz(i+1) ;
        H += Sp(i)*Sm(i+1)*0.5 ;
        H += Sm(i)*Sp(i+1)*0.5;
    }
    return H;
}

PolyOp<> FreeFermion(arma::mat const& K)
{
    auto Fermi=[L=K.n_rows](int i, bool dagger)
    {
        LocOp<> create={{0,1},{0,0}};
        auto ci=ProdOp<> {{ i%L, dagger ? create : create.t() }};
        for(auto j=0; j<i; j++) ci[j]=LocOp<> {{1,0},{0,-1}};    // fermionic sign
        return ci;
    };
    PolyOp H;
    for(auto i=0u; i<K.n_rows; i++)
        for(auto j=0u; j<K.n_cols; j++)
            if (fabs(K(i,j))>1e-14)
                H += Fermi(i,true)*Fermi(j,false)*K(i,j);
    return H;
}

PolyOp<> InteractingFermion(arma::mat const& Vijkl)
{
    int L=sqrt(Vijkl.n_rows);
    auto Fermi=[=](int i, bool dagger)
    {
        LocOp<> create={{0,1},{0,0}};
        auto ci=ProdOp<> {{ i%L, dagger ? create : create.t() }};
        for(auto j=0; j<i; j++) ci[j]=LocOp<> {{1,0},{0,-1}};    // fermionic sign
        return ci;
    };
    PolyOp H;
    for(auto i=0; i<L; i++)
        for(auto j=i+1; j<L; j++)
            for(auto k=0; k<L; k++)
                for(auto l=k+1; l<L; l++)
            if (fabs(Vijkl(i+j*L,k+l*L))>1e-14)
                H += Fermi(i,true)*Fermi(j,true)*Fermi(k,false)*Fermi(l,false)*Vijkl(i+j*L,k+l*L);
    return H;
}

void TestAutoMPO(PolyOp<> H)
{
    auto mps1=H.to_tensorTrain();
    H.use_svd=true;
    H.compressEvery=1e6;
    auto mps2=H.to_tensorTrain();
    for(auto const& [method,mps] : { pair {"CI",mps1}, pair {"SVD", mps2} } ) {
        cout<<"using "<<method<<endl;
        for(auto const& Mi : mps.M)
            cout<<Mi.n_slices<<" ";
        cout<<"\noverlap: |1-<mps|H>/<mps|mps>|="<<abs(1-H.overlap(mps)/mps.norm2()) << endl;
    }
}

int main()
{
    int len=50;
    cout<<"\n------- Heisenberg Hamiltonian ------\n";
    TestAutoMPO(HeisenbergHam(len));

    cout<<"\n------- Free Fermions 2nd nearest neighbor chain ------\n";
    arma::mat K(len, len, arma::fill::zeros);
    for(int i=0; i<len; i++) {
        K(i,i)=-0.2;
        K(i,(i+1)%len)=K((i+1)%len,i)=1;
        K(i,(i+2)%len)=K((i+2)%len,i)=-0.5;
    }
    TestAutoMPO(FreeFermion(K));

    cout<<"\n------- Free Fermions random ------\n";

    arma::mat K2(20,20,arma::fill::randu);
    TestAutoMPO(FreeFermion(K2));

    cout<<"\n------- Chemistry Ham versus L ------\n";
    int L=20;
    arma::mat K3(L,L,arma::fill::randu);
    K3=K3*K3.t();
    arma::mat V(L*L,L*L,arma::fill::randu);
    cout<<"L nterms D theoretical_D\n";
    for(auto len=6; len<=L; len+=4) {
        auto Kin=K3.submat(0,0,len-1,len-1);
        auto Vijkl=V.submat(0,0,len*len-1,len*len-1);
        auto H=FreeFermion(Kin)+InteractingFermion(Vijkl);
        cout<<len<<" "<<H.nTerm()<<" ";
        auto mps=H.to_tensorTrain();
        cout<<mps.M[len/2].n_rows <<" "<< 2*pow(len/2,2)+3*(len/2)+2 << endl;
    }

    return 0;
}


