package com.example.activitylifecycle;

import android.os.Bundle;
import android.os.PersistableBundle;
import android.util.Log;

import androidx.activity.EdgeToEdge;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class MainActivity extends AppCompatActivity {

    static final String TAG = "MainActivity";

    // Activity is being created
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
        Log.d(TAG, "Created!");
    }

    // User navigates to the activity, and the activity is started
    @Override
    protected void onStart() {
        super.onStart();
        Log.d(TAG, "Started!");
    }

    // User returns to a paused activity
    @Override
    protected void onResume() {
        super.onResume();
        Log.d(TAG, "Resumed!");
    }

    // User navigates away from the activity but it's still partially visible
    @Override
    protected void onPause() {
        super.onPause();
        Log.d(TAG, "Paused!");
    }

    // User navigates away from the activity and it's no longer visible
    @Override
    protected void onStop() {
        super.onStop();
        Log.d(TAG, "Stopped!");
    }

    // Activity is being restarted from stopped state, when user navigates back to it
    @Override
    protected void onRestart() {
        super.onRestart();
        Log.d(TAG, "Restarted!");
    }

    // Activity is finished or destroyed by the system
    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.d(TAG, "Destroyed!");
    }
}