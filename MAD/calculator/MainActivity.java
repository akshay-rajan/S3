package com.example.calculator;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class MainActivity extends AppCompatActivity {

    TextView display;
    String first = null;
    String second = null;
    String op = null;
    Button add, subtract, multiply, divide, equals, clear, back, dot;
    Button[] digits = new Button[10];

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

        display = (TextView) findViewById(R.id.display);

        add = (Button) findViewById(R.id.add);
        subtract = (Button) findViewById(R.id.subtract);
        multiply = (Button) findViewById(R.id.multiply);
        divide = (Button) findViewById(R.id.divide);
        Button[] operators = {add, subtract, multiply, divide};

        equals = (Button) findViewById(R.id.equals);
        clear = (Button) findViewById(R.id.clear);
        back = (Button) findViewById(R.id.back);

        for (int i = 0; i < 10; i++) {
            digits[i] = (Button) findViewById(getResources().getIdentifier("b" + i, "id", getPackageName()));
        }

        // Handle operator clicks
        for (Button sign: operators) {
            sign.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    // If no entry in the display, do nothing. Otherwise
                    if (first != null || second != null) {
                        String operator = sign.getText().toString();
                        if (first != null && second != null) {
                            // Both numbers are set, hence perform the previously stored operation
                            first = calculate(first, second, op);
                            second = null;
                            display.setText(first);
                        } else if (first == null) {
                            // If the second number is set, but not the first number
                            first = second;
                            second = null;
                        }
                        // If the first number is the only number set, store the operator
                        op = operator;
                    }
                }
            });
        }

        // Handle equals button
        equals.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (first == null || second == null) return;
                first = calculate(first, second, op);
                if (first == null) {
                    display.setText("ERROR");
                } else {
                    display.setText(first);
                }
                second = null;
                op = null;
            }
        });

        // Handle backspace
        back.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (first != null && second == null) {
                    // If the first number is currently on display
                    first = first.substring(0, first.length() - 1);
                    display.setText(first);
                } else if (first != null) {
                    // If the second number is currently on display
                    second = second.substring(0, second.length() - 1);
                    display.setText(second);
                }
            }
        });

        // Clear the display and reset the calculator
        clear.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                first = null;
                second = null;
                op = null;
                display.setText("0");
            }
        });

        // For each digit button
        for (Button b: digits) {
            b.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    // Get the digit from the button
                    String digit = ((Button) view).getText().toString();
                    String curr = "0";
                    // Append the digit to the current number in the display
                    if (first == null) {
                        first = digit;
                        curr = first;
                    } else if (second == null && op == null) {
                        first += digit;
                        curr = first;
                    } else if (second == null) {
                        second = digit;
                        curr = second;
                    } else {
                        second += digit;
                        curr = second;
                    }
                    display.setText(curr);
                }
            });
        }
    }
    // Perform the calculation
    public String calculate(String first, String second, String op) {
        long f = Long.parseLong(first);
        long s = Long.parseLong(second);

        long ans = 0;
        switch (op) {
            case "+":
                ans = f + s;
                break;
            case "-":
                ans = f - s;
                break;
            case "×": // (U+00D7)
                ans = f * s;
                break;
            case "÷": // (U+00F7)
                if (s == 0) return null;
                ans = f / s;
                break;
        }
        return Long.toString(ans);
    }
}