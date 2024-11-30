# Design and Analysis of Algorithms

## Divide and Conquer

```prolog
DivideAndConquer(P):
    if Small(P):
        return Solution(P)
    else:
        Divide P into smaller instances P1, P2, ...Pk
        Apply DivideAndConquer() to each of these subproblems
        return Combine(DivideAndConquer(P1),...,DivideAndConquer(Pk))
```

### 1. Merge Sort

```prolog
MERGESORT(arr, low, high):
    if low < high:
        mid = (low + high) / 2
        MERGESORT(arr, low, mid)
        MERGESORT(arr, mid + 1, high)
        MERGE(arr, low, mid, high)
MERGE(arr, low, mid, high):
    Initialize a new array 'temp'
    index = 0
    left = low
    right = high
    while left <= mid and right <= high:
        if arr[left] <= arr[right]:
            % Pick from the left
            temp[index] = arr[left]
            left = left + 1
        else:
            % Pick from the right
            temp[index] = arr[right]
            right = right + 1
        index = index + 1
    % If the left half is not fully taken into temp
    while left <= mid:
        temp[index] = arr[left]
        left = left + 1
        index = index + 1
    % If the right half is not fully taken
    while right <= high:
        temp[index] = arr[right]
        right = right + 1
        index = index + 1
    % Replace Elements in the original array with 'temp'
    for i from low to high:
        arr[i] = temp[i - low]
```

### 2. Quick Sort

```prolog
QUICKSORT(arr, low, high):
    if low < high:
        p = PARTITION(arr, low, high)
        QUICKSORT(arr, low, p - 1)
        QUICKSORT(arr, p + 1, high)

PARTITION(arr, low, high):
    % Set the first value as the pivot
    pivot = arr[low] 
    i = low + 1
    j = high
    while i < j:
        while arr[i] <= pivot and i < high:
            % Move the left pointer
            i = i + 1
        while arr[j] > pivot and j > low:
            % Move the right pointer
            j = j - 1 
        if i < j:
            % Found two values that are in the wrong partitions
            SWAP(arr[i], arr[j])
    % Swap the pivot with element at the found pivot position
    SWAP(arr[low], arr[j])
    % Return the pivot position
    return j            
```

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
        for each visited node v in the graph:
            Find the MINIMUM edge E from a visited node to an unvisited node u
            Mark u as visited
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

## Approximation Algorithms

### 1. 2-approximation algorithm for Vertex Cover Problem

```c
Vertex-Cover(G(V, E)):
    C = φ // Vertex cover
    E` = E
    while E` != φ:
        Pick an edge {u, v} from `E`
        C = C U {u, v} // Add the edge to the vertex cover
        Remove from E` every edge incident on either u or v
    return C
```

## Randomized Algorithms

### 1. Randomized Quick Sort

```c
Quicksort(A, s, t):
   if s >= t: return
   Pick pivot p randomly from {s, s+1, ..., t}
   q = Partition(A, s, t, p)
   Quicksort(A, s, q - 1)
   Quicksort(A, q + 1, t)
```
