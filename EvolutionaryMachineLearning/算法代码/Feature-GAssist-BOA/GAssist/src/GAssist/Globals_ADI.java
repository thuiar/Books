/*
 * Globals_ADI.java
 *
 */
package GAssist;

import GAssist.Dataset.Attribute;
import GAssist.Dataset.Attributes;

/**
 * This class computes and maintains global information for the ADI KR
 */
public class Globals_ADI {
	
	public static int ruleSize;
	public static int[] size;
	public static int[] offset;
	public static int[] types;
	public static ProbabilityManagement probReinit;


	public static void initialize() {
		ruleSize=0;
		size = new int[Parameters.numAttributes];
		types = new int[Parameters.numAttributes];
		offset = new int[Parameters.numAttributes];

		for(int i=0;i<Parameters.numAttributes;i++) {
			Attribute at=Attributes.getAttribute(i);
			offset[i]=ruleSize;
			if(at.getType()==Attribute.NOMINAL) {
				types[i]=Attribute.NOMINAL;
				size[i]=at.getNumNominalValues()+2;
			} else {
				types[i]=Attribute.REAL;
				size[i]=Parameters.maxIntervals*2+2;
			}
			ruleSize+=size[i];
		}
		ruleSize++;
		ruleSize++;

		probReinit = new ProbabilityManagement(
			Parameters.probReinitializeBegin,
			Parameters.probReinitializeEnd,
			ProbabilityManagement.LINEAR);
	}

	public static void nextIteration() {
		if(!Parameters.adiKR) return;

		Parameters.probReinitialize=probReinit.incStep();
	}
	public static boolean hasDefaultClass() {
		return true;
	}
}
