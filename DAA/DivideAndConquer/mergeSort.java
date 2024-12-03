import java.util.Arrays;
import java.util.Scanner;

class Solution {
    static void mergeSort(int[] arr, int low, int high) {
        // If there is more than one element
        if (low < high) {
            int mid = (low + high) / 2; // Find the middle element
            mergeSort(arr, low, mid); // Conquer the left half
            mergeSort(arr, mid + 1, high); // Conquer the right half
            merge(arr, low, mid, high); // Merge the two halves
        }
    }
    static void merge(int[] arr, int low, int mid, int high) {
        // Temporary array to store the merged array
        int[] temp = new int[high - low + 1];
        int index = 0;

        int left = low, right = mid + 1;
        while (left <= mid && right <= high) {
            // Compare the elements from the two halves and store the smaller one
            if (arr[left] <= arr[right]) {
                temp[index] = arr[left];
                left++;
            } else {
                temp[index] = arr[right];
                right++;
            }
            index++;
        }
        // Copy the remaining elements from the left half
        while (left <= mid) {
            temp[index++] = arr[left++];
        }
        // Copy the remaining elements from the right half
        while (right <= high) {
            temp[index++] = arr[right++];
        }
        // Copy the merged array back to the original array
        for (int i = low; i <= high; i++) {
            arr[i] = temp[i - low];
        }
    }
}

public class mergeSort {
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
            Solution.mergeSort(arr, 0, n - 1);
            System.out.println("Sorted:\t" + Arrays.toString(arr));
        }
    }
}
