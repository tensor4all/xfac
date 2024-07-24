#ifndef TREE_H
#define TREE_H

#include <vector>
#include <map>
#include <set>
#define ARMA_DONT_USE_OPENMP
#include <armadillo>
#include <cassert>

#include<iostream>

namespace xfac {


template<class T>
class Tree {
public:
    std::map<int, T> nodes;  // nodes appear only in this map when they contain data
    std::map<std::pair<int,int>,T> edges;  // edges appear only in this map when they contain data
    std::map<int, std::set<int> > neigh;

    Tree(){}

    std::size_t size() const { return neigh.size(); }

    void addEdge(int from, int to, T data)
    {
        addEdge(from, to);
        edges[{from,to}]=data;
    }

    void addEdge(int from, int to)
    {
        if (from==to) throw std::invalid_argument("Tree::addEdge wrong node indices");
        neigh[from].insert(to);
        neigh[to].insert(from);
    }

    /// return pair of nodes {from,to}
    std::vector<std::pair<int,int>> pathDepthFirst(int root=0) const
    {
        std::vector<int> path;
        impl_walk_depth_first(path,root);
        std::vector<std::pair<int,int>> out;
        for (int i=0u; i+1<path.size(); i++)
            out.push_back({path[i],path[i+1]});
        return out;
    }

    /// return pair of nodes {from,to}
    std::vector<std::pair<int,int>> pathLeavesToRoot(int root=0) const
    {
        std::vector<std::pair<int,int>> out;
        impl_leaves_to_root(out,root);
        return out;
    }

    /// split tree at connecting edge between node0 and node1 and return the nodes of the two subtrees
    std::pair<std::set<int>, std::set<int>> split(int node0, int node1) const
    {
        if (!(neigh.at(node0).find(node1) != neigh.at(node0).end())) throw std::invalid_argument("Tree::split node0 and node1 are not neighbors");
        std::vector<int> path0, path1;
        impl_walk_depth_first(path0, node0, node1); 
        impl_walk_depth_first(path1, node1, node0);
        std::set<int> s0, s1;
        for (auto i : path0)
            if (i != node1) s0.insert(i);
        for (auto i : path1)
            if (i != node0) s1.insert(i);
        return std::make_pair(s0, s1);
    }

  protected:
    void impl_walk_depth_first(std::vector<int>& path, int nodeid, int parent=-1) const
    {
        path.push_back(nodeid);
        for ( auto n: neigh.at(nodeid) )
            if (n!=parent) impl_walk_depth_first(path, n, nodeid);
        if (parent!=-1) path.push_back(parent);
    }

    void impl_leaves_to_root(std::vector<std::pair<int,int>>& path, int nodeid, int parent=-1) const
    {
        for ( auto n: neigh.at(nodeid) )
            if (n!=parent) impl_leaves_to_root(path, n, nodeid);
        if (parent!=-1) path.push_back({nodeid,parent});
    }
};


} // end namespace xfac


#endif // TREE_H
