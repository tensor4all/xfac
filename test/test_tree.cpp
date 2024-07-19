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
    SECTION( "basic" )
    {
        Tree<int> tree;
        
        // the off-diagonal of the matrix are the edges with a number of sites

        for( auto [i,j]: vector<pair<int,int>> {{0,1}, {1,2}, {1,3}, {3,4}, {3,5}})
            tree.addEdge(i,j);
        
        for (auto i=0u; i<tree.nodes.size(); i++)
            std::cout << "node= " << i << " data= " << tree.nodes[i] << " neighbors= " << tree.neigh[i]  <<  "\n";


        for (auto [from,to]:tree.pathDepthFirst()) {
            std::cout << from+1 << " " << to+1 << " " <<  tree.edges[{from, to}];
            for (auto n:tree.neigh[from])
                if (n!=to) cout<<" "<< n+1;
            cout<<endl;
        }

    }
}
