/*
 * Globals_GABIL.java
 *
 */
package GAssist;

import java.util.ArrayList;
import java.util.List;

import GAssist.Dataset.Attribute;
import GAssist.Dataset.Attributes;

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
		} ruleSize++;

	}

	public static boolean hasDefaultClass() {
		return true;
	}
	
	//根据offset 和  size信息对编码进行分组
	public static List<int[]> dataGroup(){
		List<int[]> arrays =new ArrayList<int[]>();
		for(int i=0;i<offset.length ;i++){
			
			int[] a = new int[size[i]];
			for(int j=0;j<size[i];j++){
				a[j] = offset[i]+j;
			}
			arrays.add(a);
			
		}
		return arrays;
	}
}
