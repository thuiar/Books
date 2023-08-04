/*
 * Globals_UBR.java
 *
 */
package GAssist;

import GAssist.Dataset.Attribute;
import GAssist.Dataset.Attributes;

/**
 * This class computes and maintains global information for the UBR KR
 */
public class Globals_UBR {
	
	public static int ruleSize;
	public static int[] size;
	public static int[] offset;
	public static int[] types;
	public static double[] minD;
	public static double[] maxD;
	public static double[] sizeD;


	public static void initialize() {
		ruleSize=0;
		size = new int[Parameters.numAttributes];
		types = new int[Parameters.numAttributes];
		offset = new int[Parameters.numAttributes];
		minD = new double[Parameters.numAttributes];
		maxD = new double[Parameters.numAttributes];
		sizeD = new double[Parameters.numAttributes];

		for(int i=0;i<Parameters.numAttributes;i++) {
			Attribute at=Attributes.getAttribute(i);
			offset[i]=ruleSize;
			if(at.getType()==Attribute.NOMINAL) {
				types[i]=Attribute.NOMINAL;
				size[i]=at.getNumNominalValues();
			} else {
				types[i]=Attribute.REAL;
				size[i]=4;
				minD[i]=at.minAttribute();
				maxD[i]=at.maxAttribute();
				sizeD[i]=maxD[i]-minD[i];
			}
			ruleSize+=size[i];
		}
		ruleSize++;
	}
	public static boolean hasDefaultClass() {
		return true;
	}
}
