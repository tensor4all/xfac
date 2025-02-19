#include<catch2/catch.hpp>

#include<iostream>
#include <cassert>
#include "xfac/tree/tree.h"
#include "xfac/grid.h"

/// Overload outstream to write vectors of numbers to the console
template< typename T >
std::ostream & operator<<( std::ostream & o, const std::set<T> & vec ) {
    o <<  "[ ";
    for (auto elem : vec)
        o << elem << ", ";
    o <<  "]";
    return o;
}

template< typename T >
std::ostream & operator<<( std::ostream & o, const std::vector<T> & vec ) {
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
    OrderedTree tree;


    // the off-diagonal of the matrix are the edges with a number of sites

    for( auto [i,j]: vector<pair<int,int>> {{0,1}, {1,2}, {1,3}, {3,4}, {3,5}})
        tree.addEdge(i,j);

    SECTION("constructor") {
        cout<<"node data neighbors\n";
        for (auto i=0u; i<tree.nodes.size(); i++)
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
            std::cout << from << " " << to << " \n";
    }

    SECTION( "is_tree" )
    {
        cout<<"test tree\n";
        OrderedTree disconnected_graph;
        for( auto [i,j]: vector<pair<int,int>> {{0,1}, {1,2}, {3,4}, {3,5}}) // two disconnected graphs
            disconnected_graph.addEdge(i,j);

        assert(tree.isTree());
        assert(!disconnected_graph.isTree());
    }


    SECTION( "tucker tree" )
    {
        cout<<"tucker tree\n";

        for( auto [dim, nBits]: vector<pair<int,int>> {{3,1}, {4,2}, {6,3}}){

            std::cout << "make Tucker tree: dim= " << dim << " , nBits= " << nBits <<  "\n";
            tree = makeTuckerTree(dim, nBits);
            assert(tree.isTree());

            // print tree
            std::set<int> visitedNodes;
            for (auto [from,to]:tree.rootToLeaves()) {
                if (!visitedNodes.contains(from)){
                    std::cout << from << " " << to;
                    for (auto n:tree.neigh[from].from_int())
                        if (n!=to) cout<<" "<< n;
                    cout<<endl;
                    visitedNodes.insert(from);
                }
            }
        }
    }

    SECTION( "quantics tree grid" )
    {
        cout<<"quantics tree grid\n";

        int dim=3;
        int nBits=2;
        auto grid = xfac::grid::QuanticsTree(0., 1., nBits, dim);

        for (auto i0 : {0, 1}){
            for (auto j0 : {0, 1}){
                cout << "i=" << i0 << " j="<< j0 << " x=" << grid.id_to_coord(std::vector(dim, std::vector({i0, j0}))) << endl;
             }
        }

        vector<double> x = {0.5, 0.25, 0.75};
        auto bitvec = grid.coord_to_id(x);
        for (int i=0; i<dim; i++)
            cout << "i=" << i << " x[i]=" << x[i] << " bitvec[i]="<< bitvec[i] << endl;

    }

}
