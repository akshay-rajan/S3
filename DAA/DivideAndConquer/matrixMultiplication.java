import java.util.Arrays;
import java.util.Scanner;

class Solution {
    public static int[][] strassens(int[][] matrix1, int[][] matrix2) {
        int m1 = matrix1.length, n1 = matrix1[0].length;
        int m2 = matrix2.length, n2 = matrix2[0].length;

        if (n1 != m2) {
            System.out.println("Matrix multiplication not possible");
            return new int[0][0];
        }

        int[][] result = new int[m1][n2];
        if (m1 == 1 && n1 == 1 && m2 == 1 && n2 == 1) {
            result[0][0] = matrix1[0][0] * matrix2[0][0];
            return result;
        }

        int[][] A = new int[m1/2][n1/2];
        int[][] B = new int[m1/2][n1/2];
        int[][] C = new int[m1/2][n1/2];
        int[][] D = new int[m1/2][n1/2];

        int[][] E = new int[m2/2][n2/2];
        int[][] F = new int[m2/2][n2/2];
        int[][] G = new int[m2/2][n2/2];
        int[][] H = new int[m2/2][n2/2];


        for (int i = 0; i < m1/2; i++) {
            for (int j = 0; j < n1/2; j++) {
                A[i][j] = matrix1[i][j];
                B[i][j] = matrix1[i][j + n1/2];
                C[i][j] = matrix1[i + m1/2][j];
                D[i][j] = matrix1[i + m1/2][j + n1/2];

                E[i][j] = matrix2[i][j];
                F[i][j] = matrix2[i][j + n2/2];
                G[i][j] = matrix2[i + m2/2][j];
                H[i][j] = matrix2[i + m2/2][j + n2/2];
            }
        }

        int[][] P1 = strassens(A, subtract(F, H));
        int[][] P2 = strassens(add(A, B), H);
        int[][] P3 = strassens(add(C, D), E);
        int[][] P4 = strassens(D, subtract(G, E));
        int[][] P5 = strassens(add(A, D), add(E, H));
        int[][] P6 = strassens(subtract(B, D), add(G, H));
        int[][] P7 = strassens(subtract(A, C), add(E, F));

        int[][] C11 = add(subtract(add(P5, P4), P2), P6);
        int[][] C12 = add(P1, P2);
        int[][] C21 = add(P3, P4);
        int[][] C22 = subtract(subtract(add(P1, P5), P3), P7);

        for (int i = 0; i < m1/2; i++) {
            for (int j = 0; j < n2/2; j++) {
                result[i][j] = C11[i][j];
                result[i][j + n2/2] = C12[i][j];
                result[i + m1/2][j] = C21[i][j];
                result[i + m1/2][j + n2/2] = C22[i][j];
            }
        }
        return result;
    }
    // Add two matrices
    public static int[][] add(int[][] matrix1, int[][] matrix2) {
        int m = matrix1.length, n = matrix1[0].length;
        int[][] result = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = matrix1[i][j] + matrix2[i][j];
            }
        }
        return result;
    }
    // Subtract two matrices
    public static int[][] subtract(int[][] matrix1, int[][] matrix2) {
        int m = matrix1.length, n = matrix1[0].length;
        int[][] result = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = matrix1[i][j] - matrix2[i][j];
            }
        }
        return result;
    }
}

public class matrixMultiplication {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int[][] matrix1 = readMatrix(sc);
        int[][] matrix2 = readMatrix(sc);
        int[][] result = Solution.strassens(matrix1, matrix2);
        System.out.println("Product: ");
        printMatrix(result);
        sc.close();
    }
    public static int[][] readMatrix(Scanner sc) {
        System.out.print("Number rows and columns: ");
        int rows = sc.nextInt();
        int cols = sc.nextInt();
        int[][] matrix = new int[rows][cols];
        System.out.println("Elements of the matrix: ");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = sc.nextInt();
            }
        }
        return matrix;
    }
    public static void printMatrix(int[][] matrix) {
        for (int[] row: matrix) {
            System.out.println(Arrays.toString(row));
        }
    }
}
