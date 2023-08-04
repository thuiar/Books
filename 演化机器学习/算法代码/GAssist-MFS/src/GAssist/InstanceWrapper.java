package GAssist;

import GAssist.Dataset.*;
import GAssist.DiscretizationAlgorithms.*;

/** 
  * InstanceWrapper.java - Wrapper for the global KEEL Instance class tailored
  * to the needs of GAssist
  */

public class InstanceWrapper {
	int[] nominalValues;
	double []realValues;
	int instanceClass;
	int[][] nominalValuesFromDiscretizers;

	public InstanceWrapper(Instance ins) {
		nominalValues = new int[Parameters.numAttributes];
		realValues = new double[Parameters.numAttributes];

		for(int i=0;i<Parameters.numAttributes;i++) {
			nominalValues[i]=ins.getNominalAttribute(i);
			realValues[i]=ins.getRealAttribute(i);
		}
		instanceClass=ins.getInstanceClass();

		if(Parameters.adiKR) {
			int num=DiscretizationManager.getNumDiscretizers();
			nominalValuesFromDiscretizers=new int[Parameters.numAttributes][];
			for(int i=0;i<Parameters.numAttributes;i++) {
				if(Attributes.getAttribute(i).getType()==Attribute.REAL) {
					nominalValuesFromDiscretizers[i]=new int[num];
					for(int j=0;j<num;j++) {
						nominalValuesFromDiscretizers[i][j]=
							DiscretizationManager.getDiscretizer(j).discretize(i,realValues[i]);
					}
				}
			}
		}
	}

	public int[][] getDiscretizedValues() {
		return nominalValuesFromDiscretizers;
	}

	public int getDiscretizedValue(int attribute,int discretizer) {
		return nominalValuesFromDiscretizers[attribute][discretizer];
	}

	public int[] getNominalValues() {
		return nominalValues;
	}
	public int getNominalValue(int attribute) {
		return nominalValues[attribute];
	}

	public double[] getRealValues() {
		return realValues;
	}
	public double getRealValue(int attribute) {
		return realValues[attribute];
	}

	public int classOfInstance() {
		return instanceClass;
	}
}
