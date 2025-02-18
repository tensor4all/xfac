#ifndef TREE_H
#define TREE_H

#include "xfac/index_set.h"
#include <vector>
#include <map>
#include <set>
#define ARMA_DONT_USE_OPENMP
#include <armadillo>
#include <cassert>



namespace xfac {



class Tree {
public:
    std::set<int> nodes;                  // only the physical nodes appear in this set
    std::map<int, IndexSet<int> > neigh;  // mapping from a node index to a set of its neighbors and their respective ordering

    Tree(){}

    std::size_t size() const { return neigh.size(); }

    void addEdge(int from, int to)
    {
        if (from==to) throw std::invalid_argument("Tree::addEdge wrong node indices");
        neigh[from].push_back(to);
        neigh[to].push_back(from);
    }

    /// return pair of nodes {from,to} with ordering: root -> leaves
    virtual std::vector<std::pair<int,int>> rootToLeaves(int) const = 0;

    /// return pair of nodes {from,to} with ordering: leaves -> root
    virtual std::vector<std::pair<int,int>> leavesToRoot(int) const = 0;

    /// split tree at connecting edge between node0 and node1 and return the nodes of the two subtrees
    std::pair<std::set<int>, std::set<int>> split(int node0, int node1) const
    {
        if (!(neigh.at(node0).to_int().find(node1) != neigh.at(node0).to_int().end()))
            throw std::invalid_argument("Tree::split node0 and node1 are not neighbors");
        std::set<int> s0, s1;
        s0.insert(node0);
        s1.insert(node1);
        for (auto [from, to] : rootToLeaves(node0)) {
            if (s0.contains(from) && to != node1) s0.insert(to);
            if (s0.contains(to) && from != node1) s0.insert(from);
            if (s1.contains(from) && to != node0) s1.insert(to);
            if (s1.contains(to) && from != node0) s1.insert(from);
        };
        return std::make_pair(s0, s1);
    }

    /// check if data structure is a tree
    bool isTree(int root=0) const
    {
        std::set<int> connectedNodes;
        for (auto [from, to] : rootToLeaves(root)) {
            connectedNodes.insert(to);
        }
        return connectedNodes.size() == size() ? true : false;
    }
};


class OrderedTree: public Tree {
public:

    std::vector<std::pair<int,int>> rootToLeaves(int root=0) const
    {
        std::vector<int> path;
        walk_depth_first(path,root);
        std::vector<std::pair<int,int>> out;
        for (int i=0u; i+1<path.size(); i++)
            out.push_back({path[i],path[i+1]});
        return out;
    }

    std::vector<std::pair<int,int>> leavesToRoot(int root=0) const
    {
        std::vector<std::pair<int,int>> out;
        leaves_to_root(out,root);
        return out;
    }


  private:

    void walk_depth_first(std::vector<int>& path, int nodeid, int parent=-1) const
    {
        path.push_back(nodeid);
        for ( auto n: this -> neigh.at(nodeid).from_int() )
            if (n!=parent) walk_depth_first(path, n, nodeid);
        if (parent!=-1) path.push_back(parent);
    }

    void leaves_to_root(std::vector<std::pair<int,int>>& path, int nodeid, int parent=-1) const
    {
        for ( auto n: this -> neigh.at(nodeid).from_int() )
            if (n!=parent) leaves_to_root(path, n, nodeid);
        if (parent!=-1) path.push_back({nodeid,parent});
    }

};

OrderedTree makeTuckerTree(int dim, int nBit){
    /*
     *  Return a Tucker tree.
     *  Index convention example for dim = 4, nBit = 2
     *  x: physical node (given by dim), o: quantics node (given by nBit)
     * 
     *   8     9     10    11
     *   x --- x --- x --- x
     *   |     |     |     |
     *   o 1   o 3   o 5   o 7
     *   |     |     |     |
     *   o 0   o 2   o 4   o 6  
     *
     */

    OrderedTree tree;

    // vertical connections between quantics nodes
    for(int i=0; i<dim; i++)
        for(int j=0; j<nBit-1; j++)
            tree.addEdge(i * nBit + j, i * nBit + j + 1);

    // vertical connections between quantics nodes and physical nodes
    for(int i=0; i<dim; i++)
        tree.addEdge(i * nBit + 1, dim * nBit + i);

    // horizontal connection between physical nodes
    for(int i=dim * nBit; i<dim * (nBit + 1) - 1; i++)
        tree.addEdge(i, i + 1);

    // set physical nodes
    for(int i=dim * nBit; i<(dim + 1) * nBit; i++)
        tree.nodes.insert(i);

    return tree;
}

} // end namespace xfac


#endif // TREE_H
