package com.example.plantidentification;

import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.annotation.Nullable;
import android.support.v4.content.FileProvider;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.text.SpannableString;
import android.text.method.LinkMovementMethod;
import android.text.util.Linkify;
import android.util.Base64;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.AuthFailureError;
import com.android.volley.DefaultRetryPolicy;
import com.android.volley.Request;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

public class MainActivity extends AppCompatActivity implements View.OnClickListener{

    private Button UploadBn, ChooseBn, CameraBn;
    private ImageView imgView;
    private final int IMG_REQUEST = 1, CAMERA_PIC_REQUEST = 2;
    private Bitmap bitmap;
    private ProgressBar pBar;
    private String UploadUrl = "http://192.168.0.200:5000/";
    private String mCurrentPhotoPath;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        UploadBn = (Button)findViewById(R.id.uploadBn);
        ChooseBn = (Button)findViewById(R.id.chooseBn);
        CameraBn = (Button)findViewById(R.id.CameraBtn);
        imgView = (ImageView) findViewById(R.id.imageView);
        pBar = (ProgressBar) findViewById(R.id.progressBar);
        ChooseBn.setOnClickListener(this);
        UploadBn.setOnClickListener(this);
        CameraBn.setOnClickListener(this);
    }

    @Override
    public void onClick(View v) {
        switch(v.getId()) {
            case R.id.chooseBn:
                selectImage();
                break;

            case R.id.uploadBn:
                pBar.setVisibility(View.VISIBLE);
                uploadImage();
                break;

            case R.id.CameraBtn:
                cameraImage();
                break;
        }

    }

    private void cameraImage(){

        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        File photoFile = null;
        try {
            photoFile = createImageFile();
        } catch (IOException ex) {
            // Error occurred while creating the File
        }
        // Continue only if the File was successfully created
        if (photoFile != null) {
            Uri photoURI = FileProvider.getUriForFile(this,
                    "com.example.android.plantidentification",
                    photoFile);
            takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
            startActivityForResult(takePictureIntent, CAMERA_PIC_REQUEST);
        }

    }


    private void selectImage(){
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(intent,IMG_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode == IMG_REQUEST && resultCode == RESULT_OK && data!=null){
           Uri path = data.getData();
            try {
                bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(),path);
                imgView.setImageBitmap(bitmap);
                imgView.setVisibility(View.VISIBLE);

            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        if(requestCode == CAMERA_PIC_REQUEST){

            File file = new File(mCurrentPhotoPath);
            try {
                bitmap = MediaStore.Images.Media
                        .getBitmap(getContentResolver(), Uri.fromFile(file));
            } catch (IOException e) {
                e.printStackTrace();
            }
            if (bitmap != null) {
                imgView.setImageBitmap(bitmap);
                imgView.setVisibility(View.VISIBLE);
            }

        }
    }

    private void uploadImage(){
        StringRequest stringRequest = new StringRequest(Request.Method.POST, UploadUrl,
                new Response.Listener<String>() {
                    @Override
                    public void onResponse(String s) {
                        System.out.println(s);
                        pBar.setVisibility(View.GONE);
                        try {
                            JSONObject jsonObject = new JSONObject(s);
                            String Response = jsonObject.getString("response");
                            String sci = jsonObject.getString("sci");
                            String edible = jsonObject.getString("edible");
                            String url = jsonObject.getString("url");
                            String probability = jsonObject.getString("probability");
                            SpannableString MyPlant = new SpannableString("\nYour Plant is a "+ Response +
                                    "\n\n\n\t\t" + sci + "\n\n\t\tEdible: "+ edible + "\n\n\t"+ url +
                                    "\n\n\t\tConfidence: " + probability + "%") ;
                            Linkify.addLinks(MyPlant, Linkify.ALL);
                            //Display popup
                            AlertDialog alertdialog = new AlertDialog.Builder(MainActivity.this, R.style.AlertDialogStyle)
                            .setIcon(R.drawable.ic_launcher_leaf_round)
                            .setTitle("Classification")
                            .setMessage(MyPlant)
                            .setPositiveButton("Okay", new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialog, int which) {
                                    //Hide Image
                                    imgView.setImageResource(0);
                                    imgView.setVisibility(View.GONE);
                                }
                            }).show();

                            //Make Link clickable
                            ((TextView)alertdialog.findViewById(android.R.id.message)).setMovementMethod(LinkMovementMethod.getInstance());


                            //display toast with result
                            Toast.makeText(MainActivity.this,Response, Toast.LENGTH_LONG ).show();

                        } catch (JSONException e) {
                            e.printStackTrace();
                        }

                    }
                }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError volleyError) {
                pBar.setVisibility(View.GONE);
                Toast.makeText(MainActivity.this,volleyError.toString(), Toast.LENGTH_LONG ).show();
            }
        }){
            @Override
            protected Map<String, String> getParams() throws AuthFailureError {
                Map<String, String> params = new HashMap<>();
                params.put("image", imageToString(bitmap));
                return params;
            }
        };
        stringRequest.setRetryPolicy(new DefaultRetryPolicy(
                0,
                DefaultRetryPolicy.DEFAULT_MAX_RETRIES,
                DefaultRetryPolicy.DEFAULT_BACKOFF_MULT));
       MySingleton.getInstance(MainActivity.this).addToRequestQue(stringRequest);

    } // End Upload Image

    private String imageToString(Bitmap bitmap){
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream );
        byte[] imgBytes = byteArrayOutputStream.toByteArray();
        return Base64.encodeToString(imgBytes, Base64.DEFAULT);
    }

    private File createImageFile() throws IOException {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(
                imageFileName,  /* prefix */
                ".jpg",         /* suffix */
                storageDir      /* directory */
        );

        // Save a file: path for use with ACTION_VIEW intents
        mCurrentPhotoPath = image.getAbsolutePath();
        return image;
    }


}
