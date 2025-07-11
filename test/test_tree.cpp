#include<catch2/catch.hpp>

#include<iostream>
#include "xfac/tree/tree.h"

/// Overload outstream to write vectors of numbers to the console

template< typename T >
std::ostream & operator<<( std::ostream & o, const std::vector<T> & vec ) {
    o <<  "[ ";
    for (auto elem : vec)
        o << elem << ", ";
    o <<  "]";
    return o;
}

template< typename T >
std::ostream & operator<<( std::ostream & o, const std::set<T> & vec ) {
    o <<  "[ ";
    for (auto elem : vec)
        o << elem << ", ";
    o <<  "]";
    return o;
}

using namespace std;
using namespace xfac;

/// Overload outstream to write vectors of numbers to the console


TEST_CASE( "Test tree" )
{
    TopologyTree tree;

    // the off-diagonal of the matrix are the edges with a number of sites
    for( auto [i,j]: vector<pair<int,int>> {{0,1}, {1,2}, {1,3}, {3,4}, {3,5}})
        tree.addEdge(i,j);

    SECTION("constructor") {
        cout<<"node data neighbors\n";
        for (auto i=0u; i<tree.size(); i++)
            std::cout << i << " " << tree.neigh[i].from_int()  <<  "\n";
    }

    SECTION( "sweeping" )
    {
        for (auto [from,to]:tree.rootToLeaves()) {
            std::cout << from << " " << to;
            for (auto n:tree.neigh[from].from_int())
                if (n!=to) cout<<" "<< n;
            cout<<endl;
        }
    }

    SECTION( "contraction" )
    {
        cout<<"contraction\n";
        for (auto [from,to]:tree.leavesToRoot())
            std::cout << from << " " << to << " \n";
    }

    SECTION( "tucker tree" )
    {
        cout<<"tucker tree\n";

        for( auto [dim, nBit]: vector<pair<int,int>> {{1,3}, {2,3}, {3,1}, {4,2}, {6,3}}){

            std::cout << "make Tucker tree: dim= " << dim << " , nBit= " << nBit <<  "\n";
            tree = makeTuckerTree(dim, nBit);

            // the size follows directly from the structure of the tree
            if (dim == 1){
                REQUIRE(tree.size() == nBit);
            } else {
                REQUIRE(tree.size() == dim * (nBit + 1) - 2);
            }

            // print connections, TODO: maybe there is a better method to print tree explicitly
            for (auto [from,to]:tree.rootToLeaves()) {
                std::cout << from << " " << to;
                for (auto n:tree.neigh[from].from_int())
                    if (n!=to) cout<<" "<< n;
                cout<<endl;
            }
        }
    }

    SECTION( "splitting" )
    {
        tree = makeTuckerTree(4, 2);

        auto [s0, s1] = tree.split(4, 8);
        REQUIRE(s0 == std::set({0, 4}));
        REQUIRE(s1 == std::set({1, 2, 3, 5, 6, 7}));

        std::tie(s0, s1) = tree.split(1, 5);
        REQUIRE(s0 == std::set({1}));
        REQUIRE(s1 == std::set({0, 2, 3, 4, 5, 6, 7}));

        std::tie(s0, s1) = tree.split(8, 9);
        REQUIRE(s0 == std::set({0, 1, 4, 5}));
        REQUIRE(s1 == std::set({2, 3, 6, 7}));
    }

}
