package org.opencv.samples.facedetect;

public class FPS {
	private long timeOfLastFrame;
	private long currentTime;
	private long FPS;
	private long t,t0;
	
	public FPS(){
		timeOfLastFrame=(long) (System.currentTimeMillis()-62.5);
		t0 = System.currentTimeMillis();
	}
	
	public long getFPS(){
		// time is in milliseconds
		currentTime=System.currentTimeMillis();
		FPS=(long) (1000/(currentTime-timeOfLastFrame));
		timeOfLastFrame=currentTime;
		return FPS;
	}
	public long getT(){
		// time is in milliseconds
        t = 1000 * (System.currentTimeMillis()-t0);
		return t;
	}

}
