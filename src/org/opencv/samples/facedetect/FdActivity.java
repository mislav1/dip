package org.opencv.samples.facedetect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.text.DecimalFormat;
import java.util.ArrayList;
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
    public static final int        FIRST               = 0;
    public static final int        SECOND              = 1;

    private MenuItem               mItemFace50;
    private MenuItem               mItemFace40;
    private MenuItem               mItemFace30;
    private MenuItem               mItemFace20;
    private MenuItem               mItemType;
    private MenuItem               mCameraType;
    private MenuItem               mItemExit;

    private ColorBlobDetector      mDetector;
    private Scalar                 mBlobColorRgba;
    private Scalar                 mBlobColorHsv;
    private Mat                    mSpectrum;
    private Scalar                 CONTOUR_COLOR;
    private Size                   SPECTRUM_SIZE;
    
    private Mat                    mRgba;
    private Mat                    mGray;
    private File                   mCascadeFile;
    private CascadeClassifier      mJavaDetector;
    private DetectionBasedTracker  mNativeDetector;

    private int                    mDetectorType       = JAVA_DETECTOR;
    private String[]               mDetectorName;
    private String[]               mCameraName;
    private int                    mCamera             =FIRST;
    private int                    skinColorDelay      = 50;
    private int                    skinColorCounter    =0;
    private int                    faceDetectionDelay  =80;
    private int                    faceDetectionCounter=0;
    
    private MatOfRect              oldFaces;
    private MatOfRect              faces; 

    private double                 oldRgbValue         =0;
    private double                 red                 =0;
    private double                 pulse               =0;
    private float                  mRelativeFaceSize   = 0.2f;
    private int                    mAbsoluteFaceSize   = 0;
    
    private boolean                skipFirstPixel      =true;
    
    List                           pulseList           = new ArrayList();
    List                           pulseListMinus      = new ArrayList();
    
    private String                 stringListPulse     = "";
    private String                 stringListRgb       = "";
    private String                 stringListMinus     = "";
    private String                 stringListKalman     = "";

    private CameraBridgeViewBase   mOpenCvCameraView;
    
    private CustomLog PulseLogPulse = new  CustomLog("pulse.txt");
    private CustomLog PulseLogRgb = new  CustomLog("rgb.txt");
    private CustomLog PulseLogMinus = new  CustomLog("minus.txt");
    private CustomLog PulseLogKalman = new  CustomLog("kalman.txt");
    
    private Kalman                 kalman              = new Kalman();

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
    	PulseLogPulse.appendLog("[ "+stringListPulse+" ];");
    	PulseLogMinus.appendLog("[ "+stringListMinus+" ];");
    	PulseLogRgb.appendLog("[ "+stringListRgb+" ];");
    	PulseLogKalman.appendLog("[ "+stringListKalman+" ];");
        super.onDestroy();
        mOpenCvCameraView.disableView();
        System.exit(0);
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mDetector = new ColorBlobDetector();
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mDetector = new ColorBlobDetector();
        mSpectrum = new Mat();
        SPECTRUM_SIZE = new Size(200, 64);
        mBlobColorRgba = new Scalar(255);
        mBlobColorHsv = new Scalar(255);
        CONTOUR_COLOR = new Scalar(255,0,0,255);
        //mBlobColorHsv.val[0]=243.609375;
        //mBlobColorHsv.val[1]=34.953125;
        //mBlobColorHsv.val[2]=238.53125;
        //mBlobColorHsv.val[3]=0.0;
        //mDetector.setHsvColor(mBlobColorHsv);
        //mBlobColorRgba = converScalarHsv2Rgba(mBlobColorHsv);
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
    
    public void PixelItensity (List<MatOfPoint> skinPixelOnFace, Mat face){

    	double itensity=0;
       	double sum=0;
    	Iterator<MatOfPoint> itr = skinPixelOnFace.iterator(); 
        while(itr.hasNext()){
            MatOfPoint tmp = itr.next();
            List<Point> listOfPoints = tmp.toList();            
            Iterator<Point> iterator = listOfPoints.iterator();
            
            while(iterator.hasNext()){
                Point p = iterator.next();
                
                int x = (int) p.x;
                int y = (int) p.y;
                
                byte[] px = new byte[4];
                face.get(y, x, px);
                //Log.i(TAG,"rgb ->"+px[0]);
                
                byte r = (byte) px[0];
                int pixel_r = r & 0xFF;
                
                byte g = (byte) px[1];
                int pixel_g = g & 0xFF;
                
                byte b = (byte) px[2];
                int pixel_b = b & 0xFF;
                
                int pixel=(pixel_r + pixel_g + pixel_b)/3;
                itensity = pixel_r + itensity;
                ++sum;

             }
        }
       	double average = itensity / sum;
       	
       	if(!skipFirstPixel){
       	//Log.i(TAG,"average rgb value->"+average);
       	//Log.i(TAG,"itensity value->"+itensity);
       	//Log.i(TAG,"sum value->"+sum);
       	
       	//Log.i(TAG, "Touched rgba color: "+String.valueOf(average)+" .");
       	//Core.putText(mRgba, String.valueOf(average), new Point(10, 130), 3/* CV_FONT_HERSHEY_COMPLEX */, 2, new Scalar(255, 0, 0, 0), 3);   	
        
       	//Log.i(TAG,"average rgb value->"+average);       	
       	//pulseList.add(average);      	
       	//Log.i(TAG,"pulse list->"+pulseList);   	
        red = average - oldRgbValue;     
        //pulseListMinus.add(red);
        //Log.i(TAG,"pulse list minus ->"+pulseListMinus);    
        //Log.i(TAG,"red = average - oldRgbValue ->"+red);  
        
        String strRgb = new DecimalFormat("###.#####").format(average);
        stringListRgb +=" "+strRgb+",";
        
        String strMinus = new DecimalFormat("###.#####").format(red);
        stringListMinus +=" "+strMinus+",";   
        
        double[] pulse = kalman.getEstimation(red);  
        
        String strPulse = new DecimalFormat("###.#####").format(pulse[0]);
        stringListPulse +=" "+strPulse+",";
        
        String strKalman = new DecimalFormat("###.#####").format(pulse[1]);
        stringListKalman +=" "+strKalman+",";
        
        Core.putText(mRgba, String.valueOf((int)(pulse[0])), new Point(10, 300), 20/* CV_FONT_HERSHEY_COMPLEX */, 2, new Scalar(255, 255, 255, 0), 6);
       	
        oldRgbValue = average;
       	}
       	else{
       		oldRgbValue = average;
            skipFirstPixel=false;
       	}
       	
    }
    
    public void CalculateSkinColorRange_2 ( Rect face) {

        int cols = face.height;
        int rows = face.width;

        Point top = face.tl();
        
        int x = (int) (top.x + (int) cols/2);
        int y = (int) (top.y + (int) rows/2);

        if ((x < 0) || (y < 0) || (x > cols) || (y > rows)) return;
        
        Rect touchedRect = new Rect();

        //touchedRect.x = (x>20) ? x-20 : 0;
        //touchedRect.y = (y>20) ? y-20 : 0;

        //touchedRect.width = (x+40 < cols) ? x + 40 - touchedRect.x : cols - touchedRect.x;
        //touchedRect.height = (y+40 < rows) ? y + 40 - touchedRect.y : rows - touchedRect.y;
        
        touchedRect.x = (int) (top.x + (int) (cols*0.2));
        touchedRect.y = (int) (top.y + (int) (rows*0.45));

        touchedRect.width = (int) ((int) (cols*0.6 ));
        touchedRect.height = (int) ((int) (rows*0.2));
        
        Mat touchedRegionRgba = mRgba.submat(touchedRect);
        Core.rectangle(mRgba,touchedRect.tl(), touchedRect.br(),FACE_RECT_COLOR,5);

        Mat touchedRegionHsv = new Mat();
        Imgproc.cvtColor(touchedRegionRgba, touchedRegionHsv, Imgproc.COLOR_RGB2HSV_FULL);

        // Calculate average color of touched region
        mBlobColorHsv = Core.sumElems(touchedRegionHsv);
        int pointCount = touchedRegionHsv.width()*touchedRegionHsv.height();
        for (int i = 0; i < mBlobColorHsv.val.length; i++)
            mBlobColorHsv.val[i] /= pointCount;

        mBlobColorRgba = converScalarHsv2Rgba(mBlobColorHsv);

        Log.i(TAG, "Touched rgba color: (" + mBlobColorHsv.val[0] + ", " + mBlobColorHsv.val[1] +
                ", " + mBlobColorHsv.val[2] + ", " + mBlobColorHsv.val[3] + ")");

        mDetector.setHsvColor(mBlobColorHsv);

        Imgproc.resize(mDetector.getSpectrum(), mSpectrum, SPECTRUM_SIZE);
        return;

    }
    
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

    	if ( faceDetectionCounter % faceDetectionDelay == 0){	        
	        if (mAbsoluteFaceSize == 0) {
	            int height = mGray.rows();
	            if (Math.round(height * mRelativeFaceSize) > 0) {
	                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
	            }
	            mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
	        }
	
	        faces = new MatOfRect();
	
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
	        oldFaces=faces;
    	}
	    ++faceDetectionCounter;
	    faces=oldFaces;
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++){

//----------------------------------------------------------------------------------------------
        	
            Mat onlyFace = mRgba.submat(facesArray[i]);
            Core.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
        
            //skinColorCounter % skinColorDelay == 0
            //CalculateSkinColorRange_2(facesArray[i]);
            if (skinColorCounter % skinColorDelay == 0){
                CalculateSkinColorRange_2(facesArray[i]);
            }
            
            mDetector.process(onlyFace);
            List<MatOfPoint> contours = mDetector.getContours();
            Log.e(TAG, "Contours count: " + contours.size());
            if (contours.size()>=1){
            PixelItensity(contours,onlyFace);
            }
            Imgproc.drawContours(onlyFace, contours, -1, CONTOUR_COLOR);
            //skin collor in upper right corner
            Mat colorLabel = mRgba.submat(4, 68, 4, 68);
            colorLabel.setTo(mBlobColorRgba);

            Mat spectrumLabel = mRgba.submat(4, 4 + mSpectrum.rows(), 70, 70 + mSpectrum.cols());
            mSpectrum.copyTo(spectrumLabel);
            
    	    ++skinColorCounter;
        }
//----------------------------------------------------------------------------------------------        
        
        
        return mRgba;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemExit   = menu.add("Exit");
        mItemFace50 = menu.add("Size 50%");
        mItemFace40 = menu.add("Size 40%");
        mItemFace30 = menu.add("Size 30%");
        mItemFace20 = menu.add("Size 20%");
        mCameraType = menu.add(mCameraName[mCamera]);
        mItemType   = menu.add(mDetectorName[mDetectorType]);
        
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if( item == mItemExit)
        	this.onDestroy();
        else if (item == mItemFace50)
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
                mGray.release();
                mRgba.release();
                mDetector = new ColorBlobDetector();
                mSpectrum = new Mat();
                SPECTRUM_SIZE = new Size(200, 64);
                mBlobColorRgba = new Scalar(255);
                mBlobColorHsv = new Scalar(255);
                CONTOUR_COLOR = new Scalar(255,0,0,255);
                mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
                mOpenCvCameraView.setCameraIndex(1);
                //1 -> main camera
                mOpenCvCameraView.setCvCameraViewListener(this);
                skipFirstPixel=true;
                kalman = new Kalman();
                OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
            } else {
                Log.i(TAG, "Front camera");
                //mOpenCvCameraView.setCameraIndex(-1);
                mOpenCvCameraView.disableView();
                mGray.release();
                mRgba.release();
                mDetector = new ColorBlobDetector();
                mSpectrum = new Mat();
                SPECTRUM_SIZE = new Size(200, 64);
                mBlobColorRgba = new Scalar(255);
                mBlobColorHsv = new Scalar(255);
                CONTOUR_COLOR = new Scalar(255,0,0,255);
                mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
                mOpenCvCameraView.setCameraIndex(-1);
                //-1 -> front camera
                mOpenCvCameraView.setCvCameraViewListener(this);
                skipFirstPixel=true;
                kalman = new Kalman();
                OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
            }
        }
        
    }
}
