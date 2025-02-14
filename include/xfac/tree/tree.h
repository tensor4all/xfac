#ifndef TREE_H
#define TREE_H

#include <vector>
#include <map>
#include <set>
#define ARMA_DONT_USE_OPENMP
#include <armadillo>
#include <cassert>



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

    /// return pair of nodes {from,to} with ordering: root -> leaves
    virtual std::vector<std::pair<int,int>> rootToLeaves(int) const = 0;

    /// return pair of nodes {from,to} with ordering: leaves -> root
    virtual std::vector<std::pair<int,int>> leavesToRoot(int) const = 0;

    /// split tree at connecting edge between node0 and node1 and return the nodes of the two subtrees
    std::pair<std::set<int>, std::set<int>> split(int node0, int node1) const
    {
        if (!(neigh.at(node0).find(node1) != neigh.at(node0).end())) throw std::invalid_argument("Tree::split node0 and node1 are not neighbors");
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
        bool graph_connected = connectedNodes.size() == size() ? true : false;
        bool nodes_inside_tree = true;
        for (auto [node, _] : nodes) {
            if (!connectedNodes.contains(node)){
                nodes_inside_tree = false;
                break;
            }
        }
        return graph_connected && nodes_inside_tree;
    }
};


template<class T>
class OrderedTree: public Tree<T> {
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
        for ( auto n: this -> neigh.at(nodeid) )
            if (n!=parent) walk_depth_first(path, n, nodeid);
        if (parent!=-1) path.push_back(parent);
    }

    void leaves_to_root(std::vector<std::pair<int,int>>& path, int nodeid, int parent=-1) const
    {
        for ( auto n: this -> neigh.at(nodeid) )
            if (n!=parent) leaves_to_root(path, n, nodeid);
        if (parent!=-1) path.push_back({nodeid,parent});
    }

};


} // end namespace xfac


#endif // TREE_H
