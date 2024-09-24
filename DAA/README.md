# Design and Analysis of Algorithms

## Greedy Method

```c
GREEDY(inputs):
    Initialize an empty set SOLUTION
    for each input:
        if FEASIBLE(input, SOLUTION):
            SOLUTION = UNION(SOLUTION, input)
    return SOLUTION
```

### 1. Prim's Algorithm
```c
PRIMS(graph):
    Initialize an empty graph MST
    Choose an arbitrary starting vertex and mark it as visited
    
    while there are unvisited nodes in the graph:
        for each visited node in the graph:
            Find the MINIMUM edge E from a visited node to an unvisited node
            if Adding E to MST does not form a cycle in MST:
                Add E to MST
    return MST
```
### 2. Kruskal's Algorithm
```c
KRUSKALS(graph):
    Initialize an empty graph MST
    while number of edges in MST < n - 1:
        Choose an unvisited edge E in MST with MINIMUM cost
        Mark E as visited
        if Adding E to MST does not form a cycle in MST:
            Add E to MST
    return MST
```
