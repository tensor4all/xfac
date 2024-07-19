#ifndef TREE_H
#define TREE_H

#include <vector>
#include <map>
#include <set>
#define ARMA_DONT_USE_OPENMP
#include <armadillo>
#include <cassert>

#include<iostream>
/// Overload outstream to write vectors of numbers to the console

namespace xfac {


template<class T>
class Tree {
public:
    //std::vector<T> nodes;
    std::map<int, T> nodes;
    std::map<std::pair<int,int>,T> edges;
    std::map<int, std::set<int> > neigh;

    Tree(){}

    std::size_t size() const { return nodes.size(); }

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

        // TODO: change later, add method to add data to node
        nodes[to] = to;
        nodes[from] = from;
    }

    /// return pair of nodes {from,to}
    std::vector<std::pair<int,int>> pathDepthFirst(int root=0)
    {
        std::vector<int> path;
        impl_walk_depth_first(path,root);
        std::vector<std::pair<int,int>> out;
        for (int i=0u; i+1<path.size(); i++)
            out.push_back({path[i],path[i+1]});
        return out;
    }

    
    /// split tree at connecting edge between node0 and node1 and return the nodes of the two subtrees
    std::pair<std::set<int>, std::set<int>> split(int node0, int node1)
    {
        if (!(neigh[node0].find(node1) != neigh[node0].end())) throw std::invalid_argument("Tree::split node0 and node1 are not neighbors");
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
};




class Tree1
{
 public:

    Tree1(arma::Mat<int> const& graph)
    {
        for(int i=0; i<graph.n_rows; i++){
            for(int j=0; j<graph.n_cols; j++){
                if (i == j){
                    node_to_legs[i] = graph(i, i);
                } else if (graph(i, j)) {
                    edge_to_connections[std::make_pair(i, j)] = graph(i, j);
                }
            }
        }
    }    
    
    auto nodes() const
    { 
        std::vector<int> node_vec;
        for (auto const& [key, val] : node_to_legs)
            node_vec.push_back(key);
        return node_vec; 
    }

    auto edges() const 
    { 
        std::vector<std::pair<int, int>> edge_vec;
        for (auto const& [key, val] : edge_to_connections)
            edge_vec.push_back(key);
        return edge_vec; 
    }
    
    auto neighbors(int node) const 
    {
        std::vector<int> neighbor_vec;
        for (auto const& [key, val] : edge_to_connections){
            auto [p1, p2] = key;
            if (p1 == node)
                neighbor_vec.push_back(p2);
        }
        return neighbor_vec; 
    }

    int num_legs(int node) const { return node_to_legs.at(node); }

    int directedEdge(int from_node, int to_node) const { return edge_to_connections.at(std::make_pair(from_node, to_node));}

    int sum_neighbor_legs(int node) const
    {
        int sum = 0;
        for (auto const& i : neighbors(node))
            sum += num_legs(i);
        return sum; 
    }


 private:

    std::map<int, int> node_to_legs;
    std::map<std::pair<int, int>, int> edge_to_connections;
};

} // end namespace xfac


#endif // TREE_H
