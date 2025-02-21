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
    std::set<int> nodes;                  // only the artificial nodes appear in this set
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

    /// split tree at connecting edge between node0 and node1 and return the physical nodes (which are not in nodes) of the two subtrees
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
        // Remove the artificial nodes from both sets
        std::set<int> s0p, s1p;
        std::set_difference(s0.begin(), s0.end(), nodes.begin(), nodes.end(), std::inserter(s0p, s0p.begin()));
        std::set_difference(s1.begin(), s1.end(), nodes.begin(), nodes.end(), std::inserter(s1p, s1p.begin()));
        return std::make_pair(s0p, s1p);
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
    /// return a Tucker tree.

    OrderedTree tree;

    if (dim < 1) throw std::invalid_argument("makeTuckerTree: requires dim > 0");
    if (nBit < 1) throw std::invalid_argument("makeTuckerTree: requires nBit > 0");

    if (dim == 1) {
        //  special case with only physical nodes, similar to the linear tensor train.
        //   o 0 ---  o 1 ---  o 2 --- . . . --- o (nBit - 1)

        // horizontal connections between physical nodes
        for(int i=0; i<nBit - 1; i++)
            tree.addEdge(i, i + 1);

    } else if (dim == 2) {
        //  special case with only physical nodes. tree plotted horizonally:
        //   o 0 ---  o 2 ---  o 4 --- . . . --- o ( 2 * nBit - 2)
        //                                       |
        //   o 1 ---  o 3 ---  o 5 --- . . . --- o (2 * nBit - 1)

        // horizontal connections between physical nodes
        for(int i=0; i<dim; i++)
            for(int j=0; j<nBit-1; j++)
                tree.addEdge(i + j * dim, i + (j + 1) * dim);

        // vertical connection between physical nodes
        tree.addEdge(2 * nBit - 2, 2 * nBit - 1);

    } else {
        //  General case. Index convention example for dim = 4, nBit = 2
        //  x: artificial node (their are dim - 2), o: physical node (there are dim * nBit)
        //  note that the left and right corners of the artificial nodes are missing
        //
        //         8     9
        //     --- x --- x --
        //    /    |     |    \
        //   o 4   o 5   o 6   o 7
        //   |     |     |     |
        //   o 0   o 1   o 2   o 3

        // vertical connections between physical nodes
        for(int i=0; i<dim; i++)
            for(int j=0; j<nBit-1; j++)
                tree.addEdge(i + j * dim, i + (j + 1) * dim);

        // vertical connections between physical nodes and artificial nodes (exclude leftmost and rightmost)
        for(int i=1; i<dim - 1; i++)
            tree.addEdge((nBit - 1) * dim + i, dim * nBit + i - 1);

        // leftmost and rightmost vertical connections between physical nodes and artificial nodes
        tree.addEdge((nBit - 1) * dim, dim * nBit);
        tree.addEdge(dim * nBit - 1, (nBit + 1) * dim - 3);

        // horizontal connection between artificial nodes
        for(int i=dim * nBit; i<dim * (nBit + 1) - 3; i++)
            tree.addEdge(i, i + 1);

        // set artificial nodes
        for(int i=dim * nBit; i<dim * (nBit + 1) - 2; i++)
            tree.nodes.insert(i);
    }
    return tree;
}


} // end namespace xfac


#endif // TREE_H
