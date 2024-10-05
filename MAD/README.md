# Mobile Application Development Lab

Android Studio is the official IDE for Android OS, based on *JetBrains IntelliJ IDEA* software.
It provides a comprehensive suite of tools to help developers create, test and debug Android applications.
Android Studio uses **Gradle** to manage the build process, packaging APKs etc.

## 1. Getting Started

**Android Studio** can be downloaded from [here](https://developer.android.com/studio/#downloads).

To create a new project, run *File -> New -> Empty Views Activity*.

> Create a new virtual device to run the app on by *Device Manager -> + -> Create Virtual Device*  

Click **Run 'app' (Shift + F10)** to run on the Emulator / Smartphone connected via WiFi or USB.

> If Gradle Build fails, make sure `compileSdk` and `targetSdk` is set to `34` on *Gradle Scripts -> build.gradle.kts*

#### Project Structure:

```
app
    manifests
    java
        com.example.myapplication
            MainActivity.java
    res (resources)
        drawable
        layout
            activity_main.xml
        mipmap
        values
        xml
Gradle Scripts
```

The `xml` could be considered as the frontend and the `java` file as the backend of our application.




<!-- ## 2. Layouts -->

<!-- ## 3. Activity -->

---

References:
- [CodeWithHarry's Course](https://www.youtube.com/playlist?list=PLu0W_9lII9aiL0kysYlfSOUgY5rNlOhUd)
- [Geeks For Geeks](https://www.geeksforgeeks.org/android-studio-tutorial/)