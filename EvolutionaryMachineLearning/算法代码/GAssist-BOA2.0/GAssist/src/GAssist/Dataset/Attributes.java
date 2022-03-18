/*
 * Attributes.java
 *
 */
package GAssist.Dataset;
import java.util.Vector;

import GAssist.LogManager;

/**
 *
 */
public class Attributes {
	
	private static Vector attributes=new Vector();
	private static boolean hasNominal=false;
	private static boolean hasReal=false;
	public static int []type;

	private static int []classCounts;
	public static int numClasses;
	public static int majorityClass;
	public static int minorityClass;
	
	public static void addAttribute(Attribute attr) {
		attributes.addElement(attr);
		if(attr.getType()==Attribute.NOMINAL) hasNominal=true;
		if(attr.getType()==Attribute.REAL) hasReal=true;
	}

	public static void clear(){
		attributes=new Vector();
	}
	
	public static boolean hasNominalAttributes() {
		return hasNominal;
	}
	public static boolean hasRealAttributes() {
		return hasReal;
	}
	
	public static Attribute getAttribute(String _name) {
	   int pos=attributes.indexOf(_name);
	   return (Attribute)attributes.elementAt(pos);
	}
	
	public static Attribute getAttribute(int pos) {
	   return (Attribute)attributes.elementAt(pos);
	}

	public static int getNumAttributes() {
		return attributes.size();
	}

	public static void endOfHeader() {
		Attribute a=(Attribute)attributes.lastElement();
		numClasses=a.getNumNominalValues();
		classCounts=new int[numClasses];
		for(int i=0;i<numClasses;i++) classCounts[i]=0;
	}

	public static void insertClass(int whichClass) {
		classCounts[whichClass]++;
	}

	public static int numInstancesOfClass(int whichClass) {
		return classCounts[whichClass];
	}

	public static void computeStatistics() {
		int nA=attributes.size()-1;
		type= new int[nA];
		for(int i=0;i<nA;i++) {
			Attribute at=(Attribute)attributes.elementAt(i);
			at.computeStatistics();
			type[i]=at.getType();
		}

		int min=classCounts[0];
		int max=classCounts[0];
		majorityClass=0;
		minorityClass=0;
		for(int i=1;i<numClasses;i++) {
			if(classCounts[i]<min) {
				min=classCounts[i];
				minorityClass=i;
			}
			if(classCounts[i]>max) {
				max=classCounts[i];
				majorityClass=i;
			}
		}
		LogManager.println("Major/Minor: "+majorityClass+"/"+minorityClass);
	}
}
