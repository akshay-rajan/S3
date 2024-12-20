package com.example.radiocheckbox;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class MainActivity extends AppCompatActivity {

    RadioGroup department;
    LinearLayout events;
    Button register;

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

        department = (RadioGroup) findViewById(R.id.department);
        events = (LinearLayout) findViewById(R.id.events);
        register = (Button) findViewById(R.id.register);

        register.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Get the selected department from the radio group
                int dept_id = department.getCheckedRadioButtonId();
                RadioButton dept = (RadioButton) findViewById(dept_id);

                // Get the selected events from the check boxes
                StringBuilder events_selected = new StringBuilder();
                for (int i = 0; i < events.getChildCount(); i++) {
                    CheckBox c = (CheckBox) events.getChildAt(i);
                    if (c.isChecked())
                        events_selected.append(c.getText().toString()).append(", ");
                }

                // Display a toast message with the selected department and events
                Toast t = Toast.makeText(
                        getApplicationContext(),
                        "Department: " + dept.getText() + "\nEvents: " + events_selected.toString(),
                        Toast.LENGTH_SHORT
                );
                t.show();
            }
        });
    }
}