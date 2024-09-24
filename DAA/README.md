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

