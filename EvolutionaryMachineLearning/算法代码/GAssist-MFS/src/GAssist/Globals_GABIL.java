/*
 * Globals_GABIL.java
 *
 */
package GAssist;

import GAssist.Dataset.*;

/**
 * This class computes and maintains global information for the GABIL KR
 */
public class Globals_GABIL {

	public static int ruleSize;
	public static int[] size;
	public static int[] offset;

	public static void initialize() {
		ruleSize = 0;
		size = new int[Parameters.numAttributes];
		 offset = new int[Parameters.numAttributes];

		for (int i = 0; i < Parameters.numAttributes; i++) {
			Attribute at = Attributes.getAttribute(i);
			 offset[i] = ruleSize;
			 size[i] = at.getNumNominalValues();
			 ruleSize += size[i];
		} ruleSize++;//需要一个int表示class的值

	}

	public static boolean hasDefaultClass() {
		return true;
	}
}
