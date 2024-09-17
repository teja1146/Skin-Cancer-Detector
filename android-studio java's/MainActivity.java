package com.example.skincancerdetector;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_IMAGE_CAPTURE = 1;
    private static final int REQUEST_IMAGE_PICK = 2;
    private static final String MODEL_NAME = "skin_cancer_detection_model.tflite";

    private ImageView imageView;
    private ProgressBar progressBar;

    private Interpreter tflite;

    private final ActivityResultLauncher<Intent> takePictureLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    Bitmap bitmap = (Bitmap) result.getData().getExtras().get("data");
                    if (bitmap != null) {
                        imageView.setImageBitmap(bitmap);
                    } else {
                        Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show();
                    }
                }
            }
    );

    private final ActivityResultLauncher<Intent> pickPhotoLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    Uri imageUri = result.getData().getData();
                    try {
                        InputStream imageStream = getContentResolver().openInputStream(imageUri);
                        Bitmap bitmap = BitmapFactory.decodeStream(imageStream);
                        if (bitmap != null) {
                            imageView.setImageBitmap(bitmap);
                        } else {
                            Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show();
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e("MainActivity", "Failed to select image: " + e.getMessage());
                        Toast.makeText(this, "Failed to select image", Toast.LENGTH_SHORT).show();
                    }
                }
            }
    );

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        progressBar = findViewById(R.id.progressBar);
        Button takePhotoButton = findViewById(R.id.takePhotoButton);
        Button choosePhotoButton = findViewById(R.id.choosePhotoButton);
        Button detectButton = findViewById(R.id.detectButton);

        progressBar.setVisibility(View.GONE);

        loadModel();

        takePhotoButton.setOnClickListener(v -> dispatchTakePictureIntent());

        choosePhotoButton.setOnClickListener(v -> openGallery());

        detectButton.setOnClickListener(v -> detectSkinCancer());

        // Request camera permission if not granted
        requestCameraPermission();
    }

    private void loadModel() {
        try {
            tflite = new Interpreter(loadModelFile());
            Log.d("MainActivity", "Model loaded successfully");
        } catch (IOException e) {
            e.printStackTrace();
            Log.e("MainActivity", "Error loading model: " + e.getMessage());
            Toast.makeText(this, "Error loading model", Toast.LENGTH_SHORT).show();
        }
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd(MODEL_NAME);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void requestCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA},
                    REQUEST_IMAGE_CAPTURE);
        }
    }

    private void dispatchTakePictureIntent() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        takePictureLauncher.launch(takePictureIntent);
    }

    private void openGallery() {
        Intent pickPhoto = new Intent(Intent.ACTION_PICK,
                android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        pickPhotoLauncher.launch(pickPhoto);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        // Handle activity results if needed
    }

    private void detectSkinCancer() {
        if (imageView.getDrawable() instanceof BitmapDrawable) {
            Bitmap image = ((BitmapDrawable) imageView.getDrawable()).getBitmap();
            if (tflite != null) {
                progressBar.setVisibility(View.VISIBLE);
                float[][] result = predict(image);
                progressBar.setVisibility(View.GONE);
                displayResult(result);
            } else {
                Toast.makeText(this, "Model not loaded", Toast.LENGTH_SHORT).show();
                Log.e("MainActivity", "TFLite interpreter is null");
            }
        } else {
            Toast.makeText(this, "Please select an image first", Toast.LENGTH_SHORT).show();
        }
    }

    private float[][] predict(Bitmap image) {
        int imageSizeX = 224;
        int imageSizeY = 224;
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(4 * imageSizeX * imageSizeY * 3);
        inputBuffer.order(ByteOrder.nativeOrder());
        inputBuffer.rewind();

        Bitmap resizedImage = Bitmap.createScaledBitmap(image, imageSizeX, imageSizeY, true);

        int[] intValues = new int[imageSizeX * imageSizeY];
        resizedImage.getPixels(intValues, 0, resizedImage.getWidth(), 0, 0, resizedImage.getWidth(), resizedImage.getHeight());

        for (int pixelValue : intValues) {
            inputBuffer.putFloat(((pixelValue >> 16) & 0xFF) / 255.0f);
            inputBuffer.putFloat(((pixelValue >> 8) & 0xFF) / 255.0f);
            inputBuffer.putFloat((pixelValue & 0xFF) / 255.0f);
        }

        float[][] output = new float[1][3]; // Adjusted to match the output shape of the model
        tflite.run(inputBuffer, output);
        return output;
    }

    private void displayResult(float[][] result) {
        if (result != null && result.length > 0 && result[0].length > 0) {
            float[] predictions = result[0];
            String message;

            // Correctly map indices to class labels
            String[] classLabels = {"Benign", "Healthy Skin", "Malignant"};
            int maxIndex = 0;
            float maxConfidence = predictions[0];
            for (int i = 1; i < predictions.length; i++) {
                if (predictions[i] > maxConfidence) {
                    maxConfidence = predictions[i];
                    maxIndex = i;
                }
            }

            message = classLabels[maxIndex] + " detected with " + (maxConfidence * 100) + "% confidence";
            Toast.makeText(this, message, Toast.LENGTH_LONG).show();
        } else {
            Toast.makeText(this, "No result from model", Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_IMAGE_CAPTURE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Camera permission granted", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (tflite != null) {
            tflite.close();
        }
    }
}
