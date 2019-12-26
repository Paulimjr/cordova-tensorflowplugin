package com.tensorflow.fidelidade.plugin;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;

import org.apache.cordova.CallbackContext;
import org.apache.cordova.CordovaPlugin;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.io.InputStream;
import java.util.AbstractMap;
import java.util.Map;
import java.util.PriorityQueue;

public class TensorFlowFidelidadePlugin extends CordovaPlugin {

    private static final String TAG = TensorFlowFidelidadePlugin.class.getSimpleName();
    private TIOModel model;

    //Models (Enquadramento, Qualidade da Imagem)
    private static final String ENQ_MODEL = "enq_model";
    private static final String QUALITY_MODEL = "quality_model";
    private static final String UNET_VEHICLE_MODEL = "unet_vehicle_model";
    private static final String ACTION_LOAD_MODEL = "loadModel";

    static final String ENQ_KEY = "enquadramento";
    static final String IS_QUALITY_KEY = "isGoodQuality";

    // Handler to execute in Second Thread
    // Create a background thread
    private HandlerThread mHandlerThread = new HandlerThread("HandlerThread");
    private Handler mHandler;
    private CallbackContext callbackContext;

    @Override
    public boolean execute(String action, JSONArray args, final CallbackContext callbackContext) throws JSONException {
        this.callbackContext = callbackContext;

        if (action != null && action.equalsIgnoreCase(ACTION_LOAD_MODEL)){
            //Call the load model method.

            if (args != null && args.length() > 0) {

                JSONObject object = args.getJSONObject(0);

                String modelName = object.getString("modelName");
                String imagePath = object.getString("imagePath");

                if (modelName != null || imagePath != null) {
                    this.loadModel(modelName, null); // TODO: After we need to change this line to get the File path.
                } else {
                    callbackContext.error("Invalid or not found action!");
                }

            } else  {
                callbackContext.error("The arguments can not be null!");
            }

        } else  {
            callbackContext.error("Invalid or not found action!");
        }

        return true;

    }

    /**
     * Load model to Tensor Flow Lite to execute a function
     */
    private void loadModel(String modelName, Bitmap image) {
        try {

            TIOModelBundleManager manager = new TIOModelBundleManager(getApplicationContext(), "");
            // load the model
            TIOModelBundle bundle = manager.bundleWithId(modelName);
            model = bundle.newModel();
            model.load();

            // Model loaded success -- Resize Image
            Bitmap imageResized;

            // Switch to know what is the model will be executed.
            switch (modelName) {
                case ENQ_MODEL: {
                    imageResized = this.resizeImage(image, 64);
                    this.executeFrameworkModel(imageResized);
                    break;
                }

                case QUALITY_MODEL: {
                    imageResized = this.resizeImage(image, 224);
                    this.executeQualityModel(imageResized);
                    break;
                }

                case UNET_VEHICLE_MODEL: {
                    //TODO do nothing...
                    break;
                }
            }

        } catch (Exception e) {
            Log.v(TAG, e.getMessage());
        }

    }

    /**
     * Resize the Image to use on TensorFlow Model
     */
    private synchronized Bitmap resizeImage(Bitmap img, int resizeImage) {

        try {

            if (img == null) {
                // Resizing the image
                InputStream bitmap = getAssets().open("img_black.jpg");
                Bitmap bMap = BitmapFactory.decodeStream(bitmap);
                return Bitmap.createScaledBitmap(bMap, resizeImage, resizeImage, false);

            } else {
                Log.e(TAG, "Error to resize the image!");
            }
        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, e.getMessage());
        }

        return null;
    }

    /**
     * Execute the Quality Model
     *
     * @param imageResized the image resized
     */
    private void executeQualityModel(Bitmap imageResized) {
        this.mHandlerThread.start();
        this.mHandler = new Handler(mHandlerThread.getLooper());
        JSONObject object = new JSONObject();

        this.mHandler.post(() -> {
            // Run the model on the input
            float[] result;

            try {
                result = (float[]) model.runOn(imageResized);

                if (result.length > 0) {
                    if (result[0] > result[1]) {
                        object.put(IS_QUALITY_KEY, false);
                    } else {
                        object.put(IS_QUALITY_KEY, true);
                    }

                    Log.v(TAG, "Success: "+object.toString());
                }

            } catch (TIOModelException | JSONException e) {
                e.printStackTrace();
            }
        });
    }

    /**
     * Execute the Framework Model
     *
     * @param imageResized the image resized
     */
    private void executeFrameworkModel(Bitmap imageResized) {
        this.mHandlerThread.start();
        this.mHandler = new Handler(mHandlerThread.getLooper());
        JSONObject object = new JSONObject();

        this.mHandler.post(() -> {
            // Run the model on the input
            float[] result = new float[0];

            try {
                result = (float[]) model.runOn(imageResized);
            } catch (TIOModelException e) {
                e.printStackTrace();
            }

            // Build a PriorityQueue of the predictions
            PriorityQueue<Map.Entry<Integer, Float>> pq = new PriorityQueue<>(10, (o1, o2) -> (o2.getValue()).compareTo(o1.getValue()));
            for (int i = 0; i < 13; i++) {
                pq.add(new AbstractMap.SimpleEntry<>(i, result[i]));
            }

            try {
                // Show the 10 most likely predictions
                String[] labels = ((TIOVectorLayerDescription) model.descriptionOfOutputAtIndex(0)).getLabels();

                for (int i = 0; i < 1; i++) {

                    Map.Entry<Integer, Float> e = pq.poll();

                    if (e != null)
                        object.put(ENQ_KEY, labels[e.getKey()]);

                    Log.v(TAG, "Success: "+object.toString());
                }
            } catch (JSONException e) {
                e.printStackTrace();
            }

        });
    }
}