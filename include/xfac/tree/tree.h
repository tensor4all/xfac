#ifndef TREE_H
#define TREE_H

#include "xfac/index_set.h"
#include <vector>
#include <map>
#include <set>
#include <stdexcept>
#include <cassert>

namespace xfac {

class TopologyTree {
public:
    int root = 0;                         // default root position
    std::set<int> nodes;                  // only the physical nodes appear in this set
    std::map<int, IndexSet<int> > neigh;  // mapping from a node index to a set of its neighbors and their respective ordering

    TopologyTree(){}
    TopologyTree(int root_) : root{root_} {}

    std::size_t size() const { return neigh.size(); }

    void addEdge(int from, int to)
    {
        if (from==to) throw std::invalid_argument("Tree::addEdge wrong node indices");
        neigh[from].push_back(to);
        neigh[to].push_back(from);
    }

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
        // Take the intersection between s0 and resp. s1
        // and the set of phyiscal nodes, to remove the artificial nodes
        std::set<int> s0p, s1p;
        std::set_intersection(s0.begin(), s0.end(), nodes.begin(), nodes.end(), std::inserter(s0p, s0p.begin()));
        std::set_intersection(s1.begin(), s1.end(), nodes.begin(), nodes.end(), std::inserter(s1p, s1p.begin()));
        return std::make_pair(s0p, s1p);
    }

    vector<int> leaves() const
    {
        vector<int> out;
        for(auto x:nodes) // only a physical nodes can be a leaf
            if (neigh.at(x).size()==1) out.push_back(x);
        return out;
    }

    /// return pair of nodes {from,to} with ordering: root -> leaves
    std::vector<std::pair<int,int>> rootToLeaves() const {return rootToLeaves(root);};
    std::vector<std::pair<int,int>> rootToLeaves(int root) const
    // this method climbes partly back to root in order to reach one leaf from the other
    {
        std::vector<int> path;
        walk_depth_first(path,root);
        std::vector<std::pair<int,int>> out;
        for (int i=0u; i+1<path.size(); i++)
            out.push_back({path[i],path[i+1]});
        return out;
    }

    /// return pair of nodes {from,to} with ordering: leaves -> root
    std::vector<std::pair<int,int>> leavesToRoot() const {return leavesToRoot(root);};
    std::vector<std::pair<int,int>> leavesToRoot(int root) const
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

inline bool operator==(TopologyTree const& t1, TopologyTree const& t2)
{
    return (
        t1.neigh==t2.neigh &&
        t1.nodes==t2.nodes &&
        t1.root==t2.root
        );
}


inline TopologyTree makeTuckerTree(int dim, int nBit){
    /// Return a Tucker tree.
    //  Convention: The nodes are first running over physical, then over artificial nodes.
    //  Physical nodes have index from 0 to dim * nBit - 1, the dim - 2 artificial nodes follow consecutively.
    //  Convention in the visualitions below: x refers to artificial nodes, o to physical nodes

    TopologyTree tree;

    if (dim < 1) throw std::invalid_argument("makeTuckerTree: requires dim > 0");
    if (nBit < 1) throw std::invalid_argument("makeTuckerTree: requires nBit > 0");

    if (dim == 1) {
        //  special case with only physical nodes, similar to the linear tensor train.
        //   o 0 ---  o 1 ---  o 2 --- . . . --- o (nBit - 1)

        // horizontal connections between physical nodes
        for(int i=0; i<nBit - 1; i++)
            tree.addEdge(i, i + 1);

        // set physical nodes
        for(int i=0; i<nBit; i++)
            tree.nodes.insert(i);

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

        // set physical nodes
        for(int i=0; i<2 * nBit; i++)
            tree.nodes.insert(i);

    } else {
        //  General case. Index convention example for dim = 4, nBit = 2
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

        // set physical nodes
        for(int i=0; i<dim * nBit; i++)
            tree.nodes.insert(i);
    }
    return tree;
}

} // end namespace xfac


#endif // TREE_H
