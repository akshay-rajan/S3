# Mobile Application Development Lab

Android Studio is the official IDE for Android OS, based on *JetBrains IntelliJ IDEA* software.
It provides a comprehensive suite of tools to help developers create, test and debug Android applications.

Android Studio uses *Gradle* to manage the build process, packaging APKs etc.

## Getting Started

**Android Studio** can be downloaded from the [official page](https://developer.android.com/studio/#downloads).

To create a new project, run *File -> New -> New Project*.
Now select a project template (e.g., *Empty Views Activity*), configure the project by providing a name, save location etc. and click Finish.

> Create a new virtual device to run the app on by *Device Manager -> + -> Create Virtual Device*  

Click **Run 'app' (Shift + F10)** to run on the emulator or smartphone connected via WiFi or USB.

> If Gradle Build fails, make sure `compileSdk` and `targetSdk` is set to `34` on *Gradle Scripts -> build.gradle.kts*

#### Project Structure

```
app
    ├── manifests
    │   └── AndroidManifest.xml
    ├── java
    │   └── com.example.myapplication
    │       └── MainActivity.java
    └── res
        ├── drawable
        ├── layout
        │   └── activity_main.xml
        ├── mipmap
        ├── values
        └── xml
Gradle Scripts
```

`app` is the main module of the application.
Metadata, permissions, etc. is contained in the `AdroidManifest.xml` in the `manifests` folder.
`java` folder contains the Java / Kotlin source code. 
All the non-code resources are located in the `res` folder, such as layouts (`layout`), images (`drawable`), icons (`mipmap`), strings (`values`), **XML** etc.
`Gradle Scripts` define how the project is built and managed.

The `xml` could be considered as the frontend and the `java` file as the backend of our application.

#### Widgets and Layouts

Widgets are the basic building blocks of an Android application's UI. 
They are interactive components that users can interact with. 
Common Widgets include:

- **Button**
- **TextView**: Displays text.
- **EditText**: Text input field.
- **ImageView**: Displays an image.
- **CheckBox**
- **RadioButton**
- **ProgressBar**

Layouts are containers that define the structure for a UI in an app. They hold widgets and other layouts, arranging them on the screen according to specific rules.
Common Layouts include:

- **LinearLayout**: Arranges its children in a single row or column.
- **RelativeLayout**: Arranges its children in relation to each other or to the parent container.
- **ConstraintLayout**: A flexible layout that allows you to position and size widgets.
- **FrameLayout**: A simple layout that can hold one child view, making it useful for displaying a single item.
- **GridLayout**: Arranges its children in a grid.

![hierarchy](./others/android_class_hierarchy_view.svg)

Layouts are a type of ViewGroup.
**ViewGroup** is a subclass of View that can contain other Views. 
It acts as a container for other views.
**View** is the simplest UI component in Android. Widgets are a specific type of View.


Class Hierarchy in Java:
```mermaid
graph LR;
    Object-->View;
    View-->ViewGroup;
    View-->Widgets;    
```

> To build the app as an **APK**, go to *Build -> Build App Bundles / APKs -> Build APKs*


## Features

### GridLayout

GridLayout is a layout manager that allows you to place child elements in a grid of rows and columns. 

```xml
<GridLayout
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:columnCount="2"
    android:rowCount="3">
    <!-- Child elements -->
</GridLayout>
```

### Intent

An Intent provides binding between two components, such as two activities.

> To create a new Activity inside our app, do *File -> New -> Activity*

Intent is used to start another activity, start a service, deliver a broadcast, etc.

#### 1. Explicit Intent

Explicit intents specify the component **in the same app** to start, by name (of the class).

```java
Intent intent = new Intent(CurrentActivity.this, NextActivity.class);
intent.putExtra("key", "value"); // Passing data to the next component
startActivity(intent);
```
```java
// Accessing the passed data
Intent intent = getIntent();
String value = intent.getStringExtra("key");
```

#### 2. Implicit Intent

Implicit intents do not name a specific component but declare a general action to perform, which allows a component **from another app** to handle it.

```java
// Launch a browser to display a specified URL
Intent intent = new Intent(Intent.ACTION_VIEW);
intent.setData(Uri.parse("http://www.example.com"));
startActivity(intent);

// Dial a phone number
Intent intent = new Intent(Intent.ACTION_DIAL);
intent.setData(Uri.parse("tel:+123456789"));
startActivity(intent);

// Send an email
Intent intent = new Intent(Intent.ACTION_SENDTO);
intent.setData(Uri.parse("mailto:example@example.com"));
intent.putExtra(Intent.EXTRA_SUBJECT, "Subject");
intent.putExtra(Intent.EXTRA_TEXT, "Email body");
startActivity(intent);

// Share text content with other apps
Intent intent = new Intent(Intent.ACTION_SEND);
intent.setType("text/plain");
intent.putExtra(Intent.EXTRA_TEXT, "This is the text to share.");
startActivity(Intent.createChooser(intent, "Share via"));
```

### Toast

A Toast provides simple feedback about an operation in a small popup. 
It only fills the amount of space required for the message and the current activity remains visible and interactive.

```java
Toast.makeText(getApplicationContext(), "Hello, World!", Toast.LENGTH_SHORT).show();
```

### Spinner

A Spinner is a widget that allows the user to select an item from a **dropdown list**.

```java
Spinner spinner = findViewById(R.id.spinner);
ArrayAdapter<CharSequence> adapter = ArrayAdapter.createFromResource(this, R.array.planets_array, android.R.layout.simple_spinner_item);
adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
spinner.setAdapter(adapter);
```

### Activity Lifecycle

>An activity is a fundamental component that manages the user interface and interactions of an application.

An Activity in Android goes through different states during its lifetime. 
These states are managed by the system and the developer can override the methods to handle these states.

![lifecycle](./others/activity_lifecycle.png)

### Logs

Logs are used to  print out messages that help the developer understand the flow of the code, to the log (*LogCat*).

```java
public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity"; // Common practice
    // ...
    Log.d(TAG, "Debug message");
    Log.v(TAG, "Verbose message");
    Log.i(TAG, "Info message");
    Log.w(TAG, "Warning message");
    Log.e(TAG, "Error message");
    // ...
}
```

### Shared Preferences

**Shared Preferences** is a local storage area used to store and retrieve primitive data.
It is a light weight mechanism to store a known set of values like storing UI states (favourites, stars), user preferences (game level), application settings (themes) etc.
We can create or modify shared preferences via `getSharedPreferences(key, mode)`.
Modes include `public`, `private` and `append`.
- `SharedPreferences.Editor` is used to write or edit data in the SP file.
- `SharedPreferences.OnSharePreferenceChangeListener()` listens to changes in the shared preferences file.


<!-- ## 2. Layouts -->

<!-- ## 3. Activity -->

---

References:
- [CodeWithHarry's Course](https://www.youtube.com/playlist?list=PLu0W_9lII9aiL0kysYlfSOUgY5rNlOhUd)
- [Geeks For Geeks](https://www.geeksforgeeks.org/android-studio-tutorial/)