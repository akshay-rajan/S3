import java.util.Arrays;
import java.util.Scanner;

class Solution {
    static void quickSort(int[] arr, int low, int high) {
        // If only one element remains
        if (low >= high) return; 
    
        int pivot = partition(arr, low, high); // Find the pivot
        quickSort(arr, low, pivot - 1); // Conquer the left of pivot
        quickSort(arr, pivot + 1, high); // Conquer the right of pivot
    }
    static int partition(int[] arr, int low, int high) {
        // Choose the first element in the portion as the pivot
        int pivot = arr[low];
        // Find the right position for the pivot in the sorted array
        int i = low, j = high;
        while (i < j) {
            // Move the left pointer until an element greater than pivot is encountered
            while (arr[i] <= pivot && i < high) i++; 
            // Move the right pointer until an element lesser than pivot is encountered
            while (arr[j] > pivot && j > low) j--;
            // If the above pointers are stopped before crossing each other
            if (i < j) {
                // arr[i] should be in the right half and arr[j] in the left half, so we swap
                int tmp = arr[i];
                arr[i] = arr[j];
                arr[j] = tmp;
            }
        }
        // Since i and j have crossed, j is at the pivot position, hence we swap arr[j] and pivot
        // All elements to the left of j is <= pivot and all after j is > pivot
        arr[low] = arr[j];
        arr[j] = pivot;
        // Return the pivot position
        return j;
    }
}

public class quickSort {
    public static void main(String[] args) {
        try (Scanner sc = new Scanner(System.in)) {
            System.out.print("Array Size: ");
            int n = sc.nextInt();
            int[] arr = new int[n];
            System.out.println("Enter the elements: ");
            for (int i = 0; i < n; i++) {
                arr[i] = sc.nextInt();
            }
            System.out.println("Array:\t" + Arrays.toString(arr));
            Solution.quickSort(arr, 0, n - 1);
            System.out.println("Sorted:\t" + Arrays.toString(arr));
        }
    }
}
