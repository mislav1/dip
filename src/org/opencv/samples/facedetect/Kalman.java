package org.opencv.samples.facedetect;


import java.util.ArrayList;
import java.util.List;

import jama.Matrix;
import jkalman.JKalman;
import android.util.Log;


public class Kalman {

    private static final String    TAG   = "OCVSample::Activity";
	
    private 	      JKalman kalman;
    private 	      Matrix s; // state [x, y]
	private 	      Matrix c; // corrected state [x, y]
	private           Matrix m; // measurement [x]
	private           Matrix K; // gain matrix
	private boolean   configured = false;
	private boolean   firstValue = false;
	private double    pulse      = 0;
	private double    f          = 0; // frequency
	private double    omega;
	private double    x,x_dt,z;
	private double    pi         = (double) 3.141592654;
	private double    old_z= 0;
	private double    old_x= 0;
	private double    old_x_dt= 0;
	private double    A          = 2;
	private double    fi         = pi;
	private int       k          = 0;
	private double    T          = 1/9;
	private double    frameRate;
	private double    t;
	private FPS       fps;
	
	private List<Double> listT           = new ArrayList<Double>();
    private List         pulseListKalman = new ArrayList();
    private List         pulseList       = new ArrayList();
    private List         BPMcorrect      = new ArrayList();

	public void configure(double red) {
		
		try {

			kalman = new JKalman(3,3);
			
		    omega = 2*pi*f;
		    x = A *Math.sin(omega*k*T);   
		    x_dt = -A * omega *Math.cos(omega*k*T);
		    z=omega*omega;

			s = new Matrix(3, 1);
			c = new Matrix(3, 1);

			m = new Matrix(3, 1);
			
			m.set(0, 0, red);
			m.set(1, 0, 0);			
			m.set(2, 0, 0);
			
			s.set(0, 0, x);
			s.set(1, 0, x_dt);			
			s.set(2, 0, z);
			
			
			
			// transitions for x, y
		    double[][] transition = { {1, T, 0 },           // { {1, 1},   // x
                                    { -z*T, 1, -x*T},             //   {0, 1} }; // dx
                                    {  0, 0, 1 }, };

		    double [][] processNoiseMatrix = {{0,0,0},
		    		                          {0,0.333*x*x*T*T*T*fi,-0.5*x*T*T*fi},
		    		                          {0,-0.5*x*T*T*fi,T*fi}};

		    double errorM = A*0.0001;
		    double [][] measurmentNoiseMatrix = {{errorM,0,0},
                                                 {0,errorM,0},
                                                 {0,0,errorM}};
		    //Matrix tr = new Matrix(transition);
            //kalman.setTransition_matrix(tr);
            kalman.setTransition_matrix(new  Matrix(transition));
            kalman.setMeasurement_noise_cov(new Matrix(measurmentNoiseMatrix));
            //kalman.setMeasurement_noise_cov(kalman.getMeasurement_noise_cov().identity());
            kalman.setError_cov_post(new  Matrix(processNoiseMatrix));
            
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		configured = true;
		
	}
	
	public double getEstimation(double red) {
		
		if(!configured){
			configure(red);
			fps = new FPS();
			return pulse;
		}
        ++k;
		t = fps.getT();
        frameRate = fps.getFPS();
		s = kalman.Predict();
		
		//kalman.setMeasurement_noise_cov(kalman.getMeasurement_noise_cov().identity());
		
		
        T = 1 / frameRate;
        listT.add(T);
        double sum = 0;
        for (double d : listT){
            sum += d;
        }
        double avgT = sum / listT.size();
        T=avgT;
        //T=avgT;
	    omega = 2*pi*f;
	    x = A *Math.sin(omega*t);  
	    x_dt = -A * omega *Math.cos(omega*t);
	    //z=omega*omega;  
	    
		m.set(0, 0, red);
		m.set(1, 0, 0);			
		m.set(2, 0, 0);
		
		s.set(0, 0, x);
		s.set(1, 0, x_dt);			
		s.set(2, 0, z);
		
		fi=z/10;
		
	    double [][] processNoiseMatrix = {{0,                                         0,                    0},
                                          {0,   0.333*s.get(0,0)*s.get(0,0)*T*T*T*fi,  -0.5*s.get(0,0)*T*T*fi},
                                          {0,                 -0.5*s.get(0,0)*T*T*fi,                  T*z*fi}};
		

        kalman.setError_cov_post(new  Matrix(processNoiseMatrix));

	    double[][] transition = { {    1,   T,   0 },                 // { {1, 1},   // x
                                  { -z*T,   1, -x*T},             //   {0, 1} }; // dx
                                  {    0,   0,   1 }, };
        
        kalman.setTransition_matrix(new Matrix(transition));  
        
        c = kalman.Correct(m);
        K = kalman.getGain();
        
        if (firstValue){
        //x = old_x - K.get(0, 0)*( c.get(0,0) - red  );
        //x_dt = old_x_dt- K.get(1, 1)*( c.get(0,0) - red  );
        z = old_z - K.get(2, 2)*(red - s.get(0,0));
        //Log.i(TAG,"K -> "+ z +" = "+ old_z +" - "+ K.get(2, 2)+" * "+"(  "+c.get(0,0)+" - "+red+"  );");
        
        //Log.i(TAG," c -> "+c.get(0,0)+" s -> "+s.get(0,0)+" red  -> "+red);
        
        //Log.i(TAG,"K 0 "+K.get(0,0)+"  "+K.get(0,1)+"  "+K.get(0,2));
        //Log.i(TAG,"K 1 "+K.get(1,0)+"  "+K.get(1,1)+"  "+K.get(1,2));
        //Log.i(TAG,"K 2 "+K.get(2,0)+"  "+K.get(2,1)+"  "+K.get(2,2));
        
    	//old_x= x;
    	//old_x_dt= x_dt;
        old_z=z;
        }
        else {
        	firstValue=true;
    	    //old_x= x;
    	    //old_x_dt= x_dt;
            old_z=z;
        }
        
        f=Math.sqrt(Math.abs(z)/(4*pi*pi));
        pulseListKalman.add(s.get(0,0));    
        pulseList.add(f*60);
        Log.i(TAG,"frames -> "+k);
        Log.i(TAG,"fps -> "+(1/avgT));
        Log.i(TAG,"pulse in BPM->"+pulseList);
        Log.i(TAG,"pulse list kalman->"+pulseListKalman);
        return (f*60);
		
	}

}

