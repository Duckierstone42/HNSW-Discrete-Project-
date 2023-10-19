package src;

import java.util.*;

class Node {
    int id;
    double[] vector;
    Map<Integer, List<Node>> connections = new HashMap<>();

    public Node(int id, double[] vector) {
        this.id = id;
        this.vector = vector;
    }

    public void addConnection(int layer, Node node) {
        connections.computeIfAbsent(layer, k -> new ArrayList<>()).add(node);
    }

    public double distance(Node other) {
        double sum = 0;
        for (int i = 0; i < vector.length; i++) {
            sum += (vector[i] - other.vector[i]) * (vector[i] - other.vector[i]);
        }
        return Math.sqrt(sum);
    }
}

class Graph {
    List<Node> nodes = new ArrayList<>();
    Random random = new Random();

    public void insert(Node node) {
        int layer = determineLayer();
        for (int l = 0; l <= layer; l++) {
            List<Node> neighbors = findNeighbors(node, l, 5); // Find 5 neighbors for simplicity
            for (Node neighbor : neighbors) {
                node.addConnection(l, neighbor);
                neighbor.addConnection(l, node);
            }
        }
        nodes.add(node);
    }

    public Node search(double[] query) {
        Node startNode = nodes.get(random.nextInt(nodes.size()));
        Node closest = startNode;
        double closestDist = Double.MAX_VALUE;

        for (int l = 9; l >= 0; l--) { // Assuming max layer is 9
            boolean progress = true;
            while (progress) {
                progress = false;
                List<Node> neighbors = closest.connections.getOrDefault(l, new ArrayList<>());
                for (Node neighbor : neighbors) {
                    double dist = neighbor.distance(new Node(-1, query));
                    if (dist < closestDist) {
                        closest = neighbor;
                        closestDist = dist;
                        progress = true;
                    }
                }
            }
        }
        return closest;
    }

    private int determineLayer() {
        return random.nextInt(10);
    }

    private List<Node> findNeighbors(Node node, int layer, int k) {
        PriorityQueue<Node> neighbors = new PriorityQueue<>(Comparator.comparingDouble(node::distance));
        for (Node other : nodes) {
            if (other == node) continue;
            neighbors.offer(other);
            if (neighbors.size() > k) {
                neighbors.poll();
            }
        }
        return new ArrayList<>(neighbors);
    }
}

public class HNSW {
    public static void main(String[] args) {
        Graph graph = new Graph();

        // Insert some nodes
        for (int i = 0; i < 100; i++) {
            double[] vector = new double[]{Math.random(), Math.random()};
            graph.insert(new Node(i, vector));
        }

        // Search for a node
        double[] query = new double[]{0.5, 0.5};
        Node result = graph.search(query);
        System.out.println("Closest node to [0.5, 0.5] is node " + result.id + " with vector " + Arrays.toString(result.vector));
    }
}
