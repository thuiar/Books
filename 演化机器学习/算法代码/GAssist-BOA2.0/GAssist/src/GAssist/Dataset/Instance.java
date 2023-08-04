package GAssist.Dataset;

import java.util.StringTokenizer;

import GAssist.LogManager;
import GAssist.Parameters;

/*
 * Instance.java
 *
 */

/**
 *
 */
public class Instance {
	int[] nominalValues;
	double []realValues;
	boolean []missing;
	int instanceClass;
	boolean isTrain;
	boolean hasMissing;
	
	public Instance(String def,boolean _isTrain) {
		StringTokenizer st=new StringTokenizer(def,",");
		nominalValues=new int[Parameters.numAttributes+1];
		realValues=new double[Parameters.numAttributes+1];
		missing=new boolean[Parameters.numAttributes+1];
		isTrain=_isTrain;
		hasMissing=false;

		for(int i=0;i<Parameters.numAttributes;i++) missing[i]=false;

		int attributeCount=0;
		while (st.hasMoreTokens()) {
			if(attributeCount>Parameters.numAttributes) {
				LogManager.println("Instance "+def+" has more attributes than defined "+Parameters.numAttributes+"<->"+attributeCount);
				System.exit(1);
			}
			String att=st.nextToken();

			if(att.equalsIgnoreCase("?")) {
				missing[attributeCount]=true;
				hasMissing=true;
			} else if(Attributes.getAttribute(attributeCount).getType()==Attribute.REAL) {
				try {
					realValues[attributeCount]=Double.parseDouble(att);
				} catch(NumberFormatException e) {
					LogManager.println("Attribute "+attributeCount+" of "+def+" is not a real value");
					e.printStackTrace();
					System.exit(1);
				}
			} else if(Attributes.getAttribute(attributeCount).getType()==Attribute.NOMINAL) {
				nominalValues[attributeCount]=Attributes.getAttribute(attributeCount).convertNominalValue(att);
				if(nominalValues[attributeCount]==-1) {
					LogManager.println("Attribute "+attributeCount+" of "+def+" is not a valid nominal value");
					System.exit(1);
				}
			}
			attributeCount++;
		}

		if(attributeCount!=Parameters.numAttributes+1) {
			LogManager.println("Instance "+def+" has less attributes than defined");
			System.exit(1);
		}
		if(missing[Parameters.numAttributes]) {
			LogManager.println("The class of the instance cannot be missing");
			System.exit(1);
		} else {
			instanceClass=nominalValues[Parameters.numAttributes];
		}
		
		if(isTrain) {
			for(int i=0;i<Parameters.numAttributes;i++) {
				if(!missing[i]) {
					if(Attributes.getAttribute(i).getType()==Attribute.REAL) {
						Attributes.getAttribute(i).enlargeBounds(realValues[i],instanceClass);
					} else if(Attributes.getAttribute(i).getType()==Attribute.NOMINAL) {
						Attributes.getAttribute(i).insertNominalValue(nominalValues[i],instanceClass);
					}
				}
			}
			Attributes.insertClass(instanceClass);
		}
	}

	public double getRealAttribute(int attr) {
		return realValues[attr];
	}

	public int getNominalAttribute(int attr) {
		return nominalValues[attr];
	}

	public void setRealAttribute(int attr,double value) {
		realValues[attr]=value;
	}
	public void setNominalAttribute(int attr,int value) {
		nominalValues[attr]=value;
	}


	public int[] getNominalAttributes() {
		return nominalValues;
	}

	public double[] getRealAttributes() {
		return realValues;
	}

	public boolean[] areAttributesMissing() {
		return missing;
	}

	public int getInstanceClass() {
		return instanceClass;
	}

	public boolean hasMissingValues() {
		return hasMissing;
	}

	public String toString() {
		String ins="";
		for(int i=0;i<Parameters.numAttributes;i++) {
			if(Attributes.getAttribute(i).getType()
				== Attribute.REAL) {
				ins+=realValues[i]+",";
			} else if(Attributes.getAttribute(i).getType()
				== Attribute.NOMINAL) {
				ins+=nominalValues[i]+",";
			}
		}
		ins+=instanceClass;
		return ins;
	}
}
