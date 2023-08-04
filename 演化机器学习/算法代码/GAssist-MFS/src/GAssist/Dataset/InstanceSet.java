/*
 * InstanceSet.java
 *
 */

/**
 *
 */
package GAssist.Dataset;

import java.util.*;

public class InstanceSet {
	
	private Instance[] instanceSet;
	private String header;
	
	/** Creates a new instance of InstanceSet */
	public InstanceSet(String fileName,boolean isTrain) {
		ParserARFF parser=new ParserARFF(fileName,isTrain);
		parser.parseHeader();
		header=parser.getHeader();
		String instance;
		Vector tempSet=new Vector(1000,100000);
		while((instance=parser.getInstance())!=null) {
			tempSet.addElement(new Instance(instance,isTrain));
		}
		if(isTrain) Attributes.computeStatistics();

		int sizeInstance=tempSet.size();
		instanceSet=new Instance[sizeInstance];
		for (int i=0; i<sizeInstance; i++) {
			instanceSet[i]=(Instance)tempSet.elementAt(i);
		}
	}
	
	/**
	 *  Moves cursor to the next instance.
	 */
	public int numInstances() {
		return instanceSet.length;
	}
	
	/**
	 *  Gets the instance that is located at the cursor position.
	 */
	public Instance getInstance(int whichInstance) {
		return instanceSet[whichInstance];
	}

	public Instance[] getInstances() {
		return instanceSet;
	}

	public String getHeader() {
		return header;
	}
}
