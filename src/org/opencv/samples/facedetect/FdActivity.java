package org.opencv.samples.facedetect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Iterator;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;

public class FdActivity extends Activity implements CvCameraViewListener2 {

    private static final String    TAG                 = "OCVSample::Activity";
    private static final Scalar    FACE_RECT_COLOR     = new Scalar(0, 255, 0, 255);
    public static final int        JAVA_DETECTOR       = 0;
    public static final int        NATIVE_DETECTOR     = 1;
    public static final int        FIRST       = 0;
    public static final int        SECOND     = 1;

    private MenuItem               mItemFace50;
    private MenuItem               mItemFace40;
    private MenuItem               mItemFace30;
    private MenuItem               mItemFace20;
    private MenuItem               mItemType;
    private MenuItem               mCameraType;

    private ColorBlobDetector      mDetector;
    private Scalar                 mBlobColorRgba;
    private Scalar                 mBlobColorHsv;
    private Mat                    mSpectrum;
    private Scalar                 CONTOUR_COLOR;
    private Size                   SPECTRUM_SIZE;
    private double                 sum;
    private double                 itensity = 0;
    private int                    faceX = 0;
    private int                    faceY = 0;
    
    private Mat                    mRgba;
    private Mat                    mGray;
    private File                   mCascadeFile;
    private CascadeClassifier      mJavaDetector;
    private DetectionBasedTracker  mNativeDetector;

    private int                    mDetectorType       = JAVA_DETECTOR;
    private String[]               mDetectorName;
    private String[]               mCameraName;
    private int                    mCamera         =FIRST;

    private float                  mRelativeFaceSize   = 0.2f;
    private int                    mAbsoluteFaceSize   = 0;
    
    //private int                    mCameraIndex=-1;

    private CameraBridgeViewBase   mOpenCvCameraView;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("detection_based_tracker");

                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);

                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public FdActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        mDetectorName[NATIVE_DETECTOR] = "Native (tracking)";
        
        mCameraName = new String[2];
        mCameraName[FIRST] = "Main camera";
        mCameraName[SECOND] = "Front camera";

        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.face_detect_surface_view);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setCameraIndex(1);
        //1 -> front camera
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        //mRgba = new Mat();
        mDetector = new ColorBlobDetector();
        // my skin 243.609375 34.953125 238.53125 0.0
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mDetector = new ColorBlobDetector();
        mSpectrum = new Mat();
        mBlobColorRgba = new Scalar(255);
        mBlobColorHsv = new Scalar(255);
        SPECTRUM_SIZE = new Size(200, 64);
        CONTOUR_COLOR = new Scalar(255,0,0,255);
        //mBlobColorHsv.val[0]=243.609375;
        //mBlobColorHsv.val[1]=34.953125;
        //mBlobColorHsv.val[2]=238.53125;
        //mBlobColorHsv.val[3]=0.0;
        //mDetector.setHsvColor(mBlobColorHsv);
        //mBlobColorRgba = converScalarHsv2Rgba(mBlobColorHsv);
        //mSpectrum = new Mat();
        //Imgproc.resize(mDetector.getSpectrum(), mSpectrum, SPECTRUM_SIZE);
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }
    
    private Scalar converScalarHsv2Rgba(Scalar hsvColor) {
        Mat pointMatRgba = new Mat();
        Mat pointMatHsv = new Mat(1, 1, CvType.CV_8UC3, hsvColor);
        Imgproc.cvtColor(pointMatHsv, pointMatRgba, Imgproc.COLOR_HSV2RGB_FULL, 4);

        return new Scalar(pointMatRgba.get(0, 0));
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
            mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
        }

        MatOfRect faces = new MatOfRect();

        if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetector != null)
                mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        }
        else if (mDetectorType == NATIVE_DETECTOR) {
            if (mNativeDetector != null)
                mNativeDetector.detect(mGray, faces);
        }
        else {
            Log.e(TAG, "Detection method is not selected!");
        }

        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++){

//----------------------------------------------------------------------------------------------

            Mat onlyFace = mRgba.submat(facesArray[i]);
            Core.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
        
            /////////////////////////////////////////////////////////////////////////////////
    	    if (onlyFace.height() > 0 && onlyFace.width() > 0 && onlyFace.height() >17 && onlyFace.width() > 17){
                faceY = (int) (onlyFace.height()/2);
                faceX =(int) (onlyFace.width()/2);
                Log.i(TAG,faceY+"  "+faceX);
                //Rect roi = new Rect( (int) faceX*20, (int) faceY*50, (int) faceX*80, (int) faceY*65);
                Rect roi = new Rect(faceX, faceY, faceX+8,faceY+8);
                Mat subFaceRect = new Mat (onlyFace, roi);
                Mat colorLabel1 = onlyFace.submat(roi);
                colorLabel1.setTo(mBlobColorRgba);
    	
    	
                // Calculate average color of touched region
                mBlobColorHsv = Core.sumElems(subFaceRect);
                int pointCount = subFaceRect.width()*subFaceRect.height();
                for (int x = 0; x < mBlobColorHsv.val.length; x++)
                    mBlobColorHsv.val[x] /= pointCount;

        
                Core.putText(mRgba,"  "+pointCount+"  "+subFaceRect.width()+"  "+subFaceRect.height(), new Point(10, 200), 2/* CV_FONT_HERSHEY_COMPLEX */, 2, new Scalar(255, 0, 0, 255), 3);
                mBlobColorRgba = converScalarHsv2Rgba(mBlobColorHsv);

                Log.i(TAG, "Touched rgba color: (" + mBlobColorHsv.val[0] + ", " + mBlobColorHsv.val[1] + ", " + mBlobColorHsv.val[2] + ", " + mBlobColorHsv.val[3] + ")");
                mDetector.setHsvColor(mBlobColorHsv);
                Imgproc.resize(mDetector.getSpectrum(), mSpectrum, SPECTRUM_SIZE);
                ///////////////////////////////////////////////////////////////////////
        
                mDetector.process(onlyFace);
                List<MatOfPoint> contours = mDetector.getContours();
                Log.e(TAG, "Contours count: " + contours.size());
                Imgproc.drawContours(onlyFace, contours, -1, CONTOUR_COLOR);
               	//skin collor in upper right corner
               	Mat colorLabel = mRgba.submat(4, 68, 4, 68);
               	colorLabel.setTo(mBlobColorRgba);

               	//Mat spectrumLabel = mRgba.submat(4, 4 + mSpectrum.rows(), 70, 70 + mSpectrum.cols());
               	//mSpectrum.copyTo(spectrumLabel);
               	Iterator<MatOfPoint> each = contours.iterator();
               	itensity=0;
               	sum=0;
               	while (each.hasNext()){
               		MatOfPoint wrapper = each.next();
               		double area = Imgproc.contourArea(wrapper);
               		itensity = area + itensity;
               		sum=sum+1;
               	}
               	double average = itensity / sum;
               	Log.i(TAG, "Touched rgba color: "+String.valueOf(average)+" .");
               	Core.putText(mRgba, String.valueOf(average), new Point(10, 100), 3/* CV_FONT_HERSHEY_COMPLEX */, 2, new Scalar(255, 0, 0, 255), 3);
    	    	}
        }
//----------------------------------------------------------------------------------------------        
        
        
        return mRgba;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemFace50 = menu.add("Face size 50%");
        mItemFace40 = menu.add("Face size 40%");
        mItemFace30 = menu.add("Face size 30%");
        mItemFace20 = menu.add("Face size 20%");
        mCameraType = menu.add(mCameraName[mCamera]);
        mItemType   = menu.add(mDetectorName[mDetectorType]);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemFace50)
            setMinFaceSize(0.5f);
        else if (item == mItemFace40)
            setMinFaceSize(0.4f);
        else if (item == mItemFace30)
            setMinFaceSize(0.3f);
        else if (item == mItemFace20)
            setMinFaceSize(0.2f);
        else if (item == mItemType) {
            int tmpDetectorType = (mDetectorType + 1) % mDetectorName.length;
            item.setTitle(mDetectorName[tmpDetectorType]);
            setDetectorType(tmpDetectorType);
        }
        else if (item == mCameraType) {
            int tmpCamera = (mCamera + 1) % mCameraName.length;
            item.setTitle(mCameraName[tmpCamera]);
            setCameraDetectorType(tmpCamera);
        }
        return true;
    }

    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }

    private void setDetectorType(int type) {
        if (mDetectorType != type) {
            mDetectorType = type;

            if (type == NATIVE_DETECTOR) {
                Log.i(TAG, "Detection Based Tracker enabled");
                mNativeDetector.start();
            } else {
                Log.i(TAG, "Cascade detector enabled");
                mNativeDetector.stop();
            }
        }
        
    }
    private void setCameraDetectorType(int type) {
        if (mCamera != type) {
            mCamera = type;

            if (type == FIRST) {
                Log.i(TAG, "Main camera");
                //mOpenCvCameraView.setCameraIndex(1);
                mOpenCvCameraView.disableView();
                mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
                mOpenCvCameraView.setCameraIndex(1);
                //1 -> main camera
                mOpenCvCameraView.setCvCameraViewListener(this);
                OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
            } else {
                Log.i(TAG, "Front camera");
                //mOpenCvCameraView.setCameraIndex(-1);
                mOpenCvCameraView.disableView();
                mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
                mOpenCvCameraView.setCameraIndex(-1);
                //-1 -> front camera
                mOpenCvCameraView.setCvCameraViewListener(this);
                OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
            }
        }
        
    }
}
