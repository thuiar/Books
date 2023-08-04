/*
 * Utils.java
 *
 */
package GAssist;

import java.util.ArrayList;

/**
 * Small routines doing trivial stuff here
 */
public class Utils {
	public static double getAverage(ArrayList data) {
		double ave=0;
		int i,size=data.size();

		for(i=0;i<size;i++) {
			ave+=((Double)data.get(i)).doubleValue();
		}

		ave/=(double)size;
		return ave;
	}

	public static double getDeviation(ArrayList data) {
		double ave=getAverage(data),dev=0;
		int i,size=data.size();

		for(i=0;i<size;i++) {
			double val=((Double)data.get(i)).doubleValue();
			dev+=Math.pow(val-ave,2.0);
		}

		dev/=(double)size;
		return Math.sqrt(dev);
	}
}
