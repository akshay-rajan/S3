package com.example.sharedpreferences;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class MainActivity extends AppCompatActivity {

    SharedPreferences preferences;
    TextView message;
    Button logout;

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

        message = findViewById(R.id.message);
        logout = findViewById(R.id.logout);

        // Get the Shared Preferences
        preferences = getSharedPreferences("credentials", MODE_PRIVATE);

        // Check if username is in SharedPreferences
        String username = preferences.getString("username", "");
        if (username.isEmpty()) {
            // Redirect to login activity
            Intent i = new Intent(this, MainActivity2.class);
            startActivity(i);
            finish();
        }
        // Display the username, if credentials are found
        String msg = "Hello, " + username + "!";
        message.setText(msg);

        // Logout button
        logout.setOnClickListener(e -> {
            // Clear the Shared Preferences of the username
            SharedPreferences.Editor editor = preferences.edit();
            editor.putString("username", "");
            editor.apply();
            // Redirect to login activity
            Intent i = new Intent(this, MainActivity2.class);
            startActivity(i);
            finish();
        });

    }
}