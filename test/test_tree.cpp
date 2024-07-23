#include<catch2/catch.hpp>

#include<iostream>
#include "xfac/tree/tree.h"

using namespace std;
using namespace xfac;

/// Overload outstream to write vectors of numbers to the console
template< typename T >
std::ostream & operator<<( std::ostream & o, const set<T> & vec ) {
    o <<  "[ ";
    for (auto elem : vec)
        o << elem << ", ";
    o <<  "]";
    return o;
}


TEST_CASE( "Test tree" )
{
    Tree<int> tree;

    // the off-diagonal of the matrix are the edges with a number of sites

    for( auto [i,j]: vector<pair<int,int>> {{0,1}, {1,2}, {1,3}, {3,4}, {3,5}})
        tree.addEdge(i,j);

    SECTION("constructor") {
        cout<<"node data neighbors\n";
        for (auto i=0u; i<tree.nodes.size(); i++)
            std::cout << i << " " << tree.nodes[i] << " " << tree.neigh[i]  <<  "\n";
    }
    SECTION( "sweeping" )
    {
        for (auto [from,to]:tree.pathDepthFirst()) {
            std::cout << from+1 << " " << to+1 << " " <<  tree.edges[{from, to}];
            for (auto n:tree.neigh[from])
                if (n!=to) cout<<" "<< n+1;
            cout<<endl;
        }
    }
    SECTION( "splitting" )
    {
        auto [s0, s1] = tree.split(1, 3);
        std::cout << "s0= " << s0 <<  "\n";
        std::cout << "s1= " << s1 <<  "\n";
    }    
    SECTION( "contraction" )
    {
        cout<<"contraction\n";
        for (auto [from,to]:tree.pathLeavesToRoot())
            std::cout << from+1 << " " << to+1 << " \n";
    }

}
