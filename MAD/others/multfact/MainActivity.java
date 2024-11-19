package com.example.practice;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class MainActivity extends AppCompatActivity {

    Button mulBtn, rangeBtn, factBtn;
    EditText mulInput, rangeInput, factInput;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        Intent intent = new Intent(this, MainActivity2.class);

        // Multiplication table
        mulBtn = findViewById(R.id.mulBtn);
        mulInput = findViewById(R.id.mulInput);
        rangeInput = findViewById(R.id.rangeInput);
        mulBtn.setOnClickListener(v -> {
            int n = Integer.parseInt(mulInput.getText().toString());
            int range = Integer.parseInt(rangeInput.getText().toString());
            String nString = Integer.toString(n);
            StringBuilder op = new StringBuilder(nString + " x 1 = " + nString);
            for (int i = 2; i <= range; i++) {
                op.append("\n");
                op.append(nString).append(" x ").append(Integer.toString(i)).append(" = ").append(Integer.toString(n * i));
            }
            intent.putExtra("output", op.toString());
            startActivity(intent);
        });

        // Factorial
        factBtn = findViewById(R.id.factBtn);
        factInput = findViewById(R.id.factInput);
        factBtn.setOnClickListener(v -> {
            int n = Integer.parseInt(factInput.getText().toString());
            StringBuilder op = new StringBuilder(Integer.toString(factorial(n)));
            intent.putExtra("output", op.toString());
            startActivity(intent);
        });

    }

    private int factorial(int n) {
        int num = 1;
        for (int i = 1; i <= n; i++) {
            num *= i;
        }
        return num;
    }
}