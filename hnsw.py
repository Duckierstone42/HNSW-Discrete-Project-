import math
import random
import heapq
import matplotlib.pyplot as plt

class Node:
    def __init__(self, node_id, vector):
        self.id = node_id
        self.vector = vector
        self.connections = {}

    def add_connection(self, layer, node):
        if layer not in self.connections:
            self.connections[layer] = []
        self.connections[layer].append(node)

    def distance(self, other):
        return math.sqrt(sum([(a - b) ** 2 for a, b in zip(self.vector, other.vector)]))

class Graph:
    def __init__(self, layer_coef):
        self.nodes = []
        self.layer_coef = layer_coef
    
    def insert(self, node):
        layer = self._determine_layer()
        for l in range(layer + 1):
            neighbors = self._find_neighbors(node, l, 5)  # Find 5 neighbors for simplicity
            if l not in node.connections:
                node.connections[l] = []
            for neighbor in neighbors:
                node.add_connection(l, neighbor)
                neighbor.add_connection(l, node)
        self.nodes.append(node)

    def greedy_search_nsw(self, query, layer, entry_point, k=5):
        best_dist = float("inf")

        et = (entry_point.distance(Node(-1, query)), entry_point)
        candidates = [et]
        result = [et]
        visited = set([et])
        while len(candidates) > 0:
            c_dist, c = heapq.heappop(candidates)

            # k-1th element
            if len(result) >= k:
                if c_dist < best_dist:
                    best_dist = c_dist
                    continue
            else:
                best_dist = c_dist

            for e in c.connections[layer]:
                if e not in visited:

                    visited.add(e)
                    heapq.heappush(candidates, (e.distance(Node(-1, query)), e))

                    # TODO: only keep k elements in result
                    heapq.heappush(result, (e.distance(Node(-1, query)), e))

        return heapq.nsmallest(k, result)


    def greedy_search_hnsw(self, query, entry_point, k=5):
        candidates = [(entry_point.distance(Node(-1, query)), entry_point)]
        for layer in range(len(entry_point.connections), 0, -1):
            candidates = self.greedy_search_nsw(query, layer-1, candidates[0][1])
        
        return heapq.nsmallest(k, candidates)


    def _determine_layer(self):
        # return random.randint(0, 9)
        return abs(math.floor(random.gauss(0, self.layer_coef)))

    def _find_neighbors(self, node, layer, k):
        neighbors = sorted([x for x in self.nodes if len(x.connections) >= layer+1], key=lambda other: node.distance(other))
        return neighbors[:k]
    
    def get_layer(self, layer):
        return [n for n in self.nodes if len(n.connections) == layer + 1]

def plot_layer(graph, layer):
    layer_nodes = [n for n in graph.nodes if len(n.connections) >= layer + 1]
    x = [n.vector[0] for n in layer_nodes]
    y = [n.vector[1] for n in layer_nodes] 

    # Connect the points using plot
    connections = [(layer_nodes.index(i), layer_nodes.index(j)) for i in layer_nodes for j in i.connections[layer]]
    for start, end in connections:
        plt.plot([x[start], x[end]], [y[start], y[end]], 'k-', lw=0.5, zorder=1)  # 'k-' means black line

    plt.scatter(x, y)
 
    ax = plt.gca()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

if __name__ == "__main__":
    num_nodes = 100
    # TODO: Find layer probability properly
    layer_coef = 3

    random.seed(421)

    graph = Graph(layer_coef)

    # Insert some nodes
    for i in range(num_nodes):
        vector = (random.random(), random.random())
        graph.insert(Node(i, vector))
    
    # for i in range(6):
    #     plot_layer(graph, i)
    #     plt.show()

    # plot_layer(graph, 0)
    
    ## This is for plotting HNSW
    # Find an entry point
    max_layer = max(len(x.connections) for x in graph.nodes) - 1
    entry = random.choice(graph.get_layer(max_layer))
    query = [0.5, 0.5]
    result = graph.greedy_search_hnsw(query, entry)
    points = [x[1].vector for x in result]
    print(points)
    plot_layer(graph, 0)
    x = [n[0] for n in points]
    y = [n[1] for n in points]
    plt.scatter(x, y, c="red", zorder=1)
    plt.scatter([query[0]], [query[1]], c="green", zorder=1)
    plt.scatter([entry.vector[0]], [entry.vector[1]], c="purple", zorder=1)
    plt.show()


    ## This is for plotting NSW search
    # query = [0.5, 0.5]
    # entry = random.choice(graph.get_layer(0))
    # result = graph.greedy_search_nsw(query, 0, entry)
    # points = [x[1].vector for x in result]
    # print(points)
    # plot_layer(graph, 0)
    # x = [n[0] for n in points]
    # y = [n[1] for n in points]
    # plt.scatter(x, y, c="red")
    # plt.scatter([query[0]], [query[1]], c="green")
    # plt.scatter([entry.vector[0]], [entry.vector[1]], c="purple")
    # plt.show()
    
    # print(f"Closest node to [0.5, 0.5] is node {result.id} with vector {result.vector}")
