#include<catch2/catch.hpp>

#include<iostream>
#include <cassert>
#include "xfac/tree/tree.h"

/// Overload outstream to write vectors of numbers to the console
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
    OrderedTree<int> tree;


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
        for (auto [from,to]:tree.rootToLeaves()) {
            std::cout << from << " " << to << " " <<  tree.edges[{from, to}];
            for (auto n:tree.neigh[from])
                if (n!=to) cout<<" "<< n;
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
        for (auto [from,to]:tree.leavesToRoot())
            std::cout << from+1 << " " << to+1 << " \n";
    }

    SECTION( "is_tree" )
    {
        cout<<"test tree\n";
        OrderedTree<int> disconnected_graph;
        for( auto [i,j]: vector<pair<int,int>> {{0,1}, {1,2}, {3,4}, {3,5}}) // two disconnected graphs
            disconnected_graph.addEdge(i,j);

        OrderedTree<int> dangling_node;
        for( auto [i,j]: vector<pair<int,int>> {{0,1}, {1,2}, {1,3}, {3,4}, {3,5}})
            dangling_node.addEdge(i,j);
        dangling_node.nodes[6] = 0;

        assert(tree.isTree());
        assert(!disconnected_graph.isTree());
        assert(!dangling_node.isTree());
    }

}
