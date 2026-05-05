#ifndef TREE_H
#define TREE_H

#include "xfac/index_set.h"
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <queue>
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

    void add_phys_node(int n)
    {
        if (neigh.find(n)==neigh.end()) throw std::invalid_argument("Tree::add_phys_node the node is not in node list");
        nodes.insert(n);
    }

    std::vector<int> get_phys_node_list() const { return vector(nodes.begin(),nodes.end()); }

    std::vector<int> get_node_list() const
    {
        std::vector<int> output;
        for(auto& [node,neigh] : neigh)
            output.push_back(node);
        return output;
    }

    std::map<int,std::vector<int>> get_neigh_list() const
    {
        std::map<int,std::vector<int>> output;
        for(auto& [node,neigh] : neigh)
            output.emplace(node, neigh.from_int());
        return output;
    }

    /// split tree at the connecting edge between node0 and node1 and return the nodes of the two subtrees
    /// by default, only the physical nodes (which are not in nodes) are returned
    std::pair<std::set<int>, std::set<int>> split(int node0, int node1, bool physical_only=true) const
    {
        if (!(neigh.at(node0).to_int().find(node1) != neigh.at(node0).to_int().end()))
            throw std::invalid_argument("Tree::split node0 and node1 are not neighbors");
        if (node0 == node1) throw std::invalid_argument("Tree::split: wrong node indices");
        std::set<int> s0, s1;
        s0.insert(node0);
        s1.insert(node1);
        for (auto [from, to] : rootToLeaves(node0)) {
            if (s0.contains(from) && to != node1) s0.insert(to);
            if (s0.contains(to) && from != node1) s0.insert(from);
            if (s1.contains(from) && to != node0) s1.insert(to);
            if (s1.contains(to) && from != node0) s1.insert(from);
        };
        if (physical_only){
            // Take the intersection between s0 and resp. s1
            // and the set of phyiscal nodes, to remove the artificial nodes
            std::set<int> s0p, s1p;
            std::set_intersection(s0.begin(), s0.end(), nodes.begin(), nodes.end(), std::inserter(s0p, s0p.begin()));
            std::set_intersection(s1.begin(), s1.end(), nodes.begin(), nodes.end(), std::inserter(s1p, s1p.begin()));
            return std::make_pair(s0p, s1p);
        }
        return std::make_pair(s0, s1);
    }

    /// split tree at the connecting edge between node *from* and node *to* and return the two resulting subtrees
    std::pair<TopologyTree, TopologyTree> splitTree(int from, int to) const {
        TopologyTree t0, t1;
        auto [s0, s1] = split(from, to, false);
        for (auto node : s0) {
            if (nodes.find(node) != nodes.end()) t0.nodes.insert(node);
            t0.neigh.insert({node, neigh.at(node)});
        };
        for (auto node : s1) {
            if (nodes.find(node) != nodes.end()) t1.nodes.insert(node);
            t1.neigh.insert({node, neigh.at(node)});
        };
        // attribute the original root node to either t0 or t1 such that it has the same place as before.
        // for the other tree use either the *from* or the *to* node as root.
        if (s0.find(root) != s0.end()){
            t0.root = root;
        } else {
            t0.root = from;
        }
        if (s1.find(root) != s1.end()){
            t1.root = root;
        } else {
            t1.root = to;
        }
        // remove the *from* and *to* node from the trees.
        // if dangling elements and reordering might happen, replace the former nodes *from* and *two* by a special value *dummy_node*
        if (t0.neigh.at(from).pos(to) != t0.neigh.at(from).size() - 1){
            t0.neigh.at(from).replace(to, dummy_node);
        } else {
            t0.neigh.at(from).remove(to);
        }
        if (t1.neigh.at(to).pos(from) != t1.neigh.at(to).size() - 1){
            t1.neigh.at(to).replace(from, dummy_node);
        } else {
            t1.neigh.at(to).remove(from);
        }
        return std::make_pair(t0, t1);
    }

    /// return a vector of leaf nodes, element ordering not guaranteed
    vector<int> leaves() const
    {
        vector<int> out;
        for (auto& [node, neighbors] : neigh) {
            int count = 0;
            for (int n : neighbors.from_int())
                if (n != dummy_node) count++;
            if (count < 2) out.push_back(node);
        }
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

    // -----------------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------------

    /** True if the graph is connected (every node is reachable from root). */
    bool isConnected() const
    {
        if (neigh.empty()) return true;
        std::set<int> visited;
        std::queue<int> q;
        // Use the first node in neigh as the starting point (handles arbitrary root values)
        int start = neigh.begin()->first;
        q.push(start);
        visited.insert(start);
        while (!q.empty()) {
            int cur = q.front(); q.pop();
            for (int nb : neigh.at(cur).from_int())
                if (!visited.count(nb)) { visited.insert(nb); q.push(nb); }
        }
        return visited.size() == neigh.size();
    }

    /**
     * True if the graph contains no cycle.
     *
     * A tree on N nodes has exactly N-1 edges. Any extra edge introduces a
     * cycle. We detect this with a BFS that counts visited nodes: if we ever
     * try to visit a node that is already in the visited set (and it is not
     * the parent we came from), a cycle exists.
     */
    bool isTree() const
    {
        if (neigh.empty()) return true;
        std::map<int,int> parent;   // node -> parent (-1 for start)
        std::set<int> visited;
        std::queue<int> q;
        int start = neigh.begin()->first;
        parent[start] = -1;
        q.push(start);
        visited.insert(start);
        while (!q.empty()) {
            int cur = q.front(); q.pop();
            for (int nb : neigh.at(cur).from_int()) {
                if (!visited.count(nb)) {
                    visited.insert(nb);
                    parent[nb] = cur;
                    q.push(nb);
                } else if (nb != parent[cur]) {
                    return false;  // back-edge found: cycle
                }
            }
        }
        return true;
    }

    /**
     * True if node indices are {0, 1, 2, ..., N-1} in the order returned by
     * get_node_list() (which iterates the std::map in key order, i.e. sorted).
     */
    bool hasConsecutiveNodesFromZero() const
    {
        auto node_list = get_node_list();
        for (int i = 0; i < static_cast<int>(node_list.size()); ++i)
            if (node_list[i] != i) return false;
        return true;
    }

    /**
     * True if physical nodes occupy the first indices in the node list, i.e.
     * get_phys_node_list()[i] == get_node_list()[i] for every physical index i.
     */
    bool arePhysicalNodesFirst() const
    {
        auto node_list  = get_node_list();
        auto phys_nodes = get_phys_node_list();
        for (int i = 0; i < static_cast<int>(phys_nodes.size()); ++i)
            if (phys_nodes[i] != node_list[i]) return false;
        return true;
    }

    /**
     * Validate all structural invariants.
     * Throws std::invalid_argument with a descriptive message on the first
     * failing check. Returns true if all checks pass.
     */
    bool validate() const
    {
        if (!isConnected())
            throw std::invalid_argument(
                "TopologyTree::validate: tree must be connected. "
                "Use get_neigh_list() to inspect the adjacency structure.");

        if (!isTree())
            throw std::invalid_argument(
                "TopologyTree::validate: tree must be acyclic (no cycles allowed). "
                "A cycle was detected during BFS traversal.");

        if (!hasConsecutiveNodesFromZero())
            throw std::invalid_argument(
                "TopologyTree::validate: node indices must be consecutive starting from 0. "
                "Use get_node_list() to inspect current indices.");

        if (!arePhysicalNodesFirst())
            throw std::invalid_argument(
                "TopologyTree::validate: physical nodes must occupy the first indices "
                "(indices 0..nPhys-1). Use get_node_list() and get_phys_node_list() "
                "to inspect current layout.");

        return true;
    }

    void save(std::ostream &out) const
    {
        out<<root<<std::endl;
        out<<nodes.size()<<std::endl;
        for (const auto& n:nodes) out<<" "<<n;
        out<<std::endl;
        out<<neigh.size()<<std::endl;
        for(auto& [node,nSet]:neigh) {
            out<<" "<<node<<" "<<nSet.size();
            for(auto n2:nSet.from_int())
                out<<" "<<n2;
            out<<std::endl;
        }
    }
    void save(std::string fileName) const { std::ofstream out(fileName); save(out); }

    static TopologyTree load(std::ifstream& in)
    {
        TopologyTree tree;
        in>>tree.root;
        int L, L1, node, node1;
        in>>L;
        for(auto i=0;i<L;i++) {
            in>>node;
            tree.nodes.insert(node);
        }
        in>>L;
        for(auto i=0;i<L;i++) {
            in>>node>>L1;
            for(auto j=0;j<L1;j++) {
                in>>node1;
                tree.neigh[node].push_back(node1);
            }
        }
        return tree;
    }

    static TopologyTree load(std::string fileName)
    {
        std::ifstream in(fileName);
        if (in.fail()) throw std::runtime_error("TensorTrain::load fails to load file: " + fileName);
        return load(in);
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
    static constexpr int dummy_node = -1; // special value used internally, do not attribute a node to this value
};

inline bool operator==(TopologyTree const& t1, TopologyTree const& t2)
{
    return (
        t1.neigh==t2.neigh &&
        t1.nodes==t2.nodes &&
        t1.root==t2.root
        );
}

inline bool operator!=(TopologyTree const& t1, TopologyTree const& t2)
{
    return !(t1 == t2);
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
