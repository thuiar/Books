/*
 * Attribute.java
 *
 */
package GAssist.Dataset;

import java.util.Vector;

import GAssist.LogManager;
import GAssist.Parameters;

/**
 *
 */
public class Attribute {
        public final static int NOMINAL = 0;
        public final static int REAL = 1;

	int type;
	String name;
	Vector nominalValues;
	double min;
	double max;
	boolean firstTime;
	boolean fixedBounds;
	double globalMean;
	int globalMostFreq;
	int []globalValueFrequencies;
	int globalCount;
	int []countValues;
	int [][]valueFrequencies;
	int []mostFrequentValue;
	double []valueMeans;

	public Attribute() {
		type=-1;
		globalCount=0;
		globalMean=0;
	}

	public void setType(int _type) {
		if(type!=-1) {
			LogManager.println("Type already fixed !!");
			System.exit(1);
		}
		type=_type;
		firstTime=true;

		if(type==NOMINAL) {
			nominalValues=new Vector();
		} else if(type==REAL) {
			fixedBounds=false;
		}
	}

	public int getType() {
		return type;
	}

	public String getName() {
		return name;
	}

	public void setName(String _name) {
		name=new String(_name);
	}

	public void setBounds(double _min,double _max) {
		if(type!=REAL) return;
		fixedBounds=true;
		min=_min;
		max=_max;
	}

	public void enlargeBounds(double value,int instanceClass) {
		if(type!=REAL) return;

		if(firstTime) {
			if(!fixedBounds) {
				min=value;
				max=value;
			}
			firstTime=false;

			valueMeans=new double[Parameters.numClasses];		
			countValues=new int[Parameters.numClasses];		
			for(int i=0;i<Parameters.numClasses;i++) {
				valueMeans[i]=0;
				countValues[i]=0;
			}
		}

		globalMean+=value;
		globalCount++;

		valueMeans[instanceClass]+=value;
		countValues[instanceClass]++;

		if(fixedBounds) return;
		if(value<min) {
			min=value;
		}
		if(value>max) {
			max=value;
		}
	}

	public void insertNominalValue(int value,int instanceClass) {
		if(type!=NOMINAL) return;

		if(firstTime) {
			firstTime=false;
			int numV=nominalValues.size();

			globalCount=0;
			globalValueFrequencies=new int[numV];
			for(int i=0;i<numV;i++) globalValueFrequencies[i]=0;

			valueFrequencies=new int[Parameters.numClasses][];
			countValues=new int[Parameters.numClasses];		
			for(int i=0;i<Parameters.numClasses;i++) { 
				countValues[i]=0;	
				valueFrequencies[i]=new int[numV];
				for(int j=0;j<numV;j++){
					valueFrequencies[i][j]=0;
				}
			}
		}

		globalCount++;
		globalValueFrequencies[value]++;	

		valueFrequencies[instanceClass][value]++;
		countValues[instanceClass]++;
	}

	public double minAttribute() {
		return min;
	}

	public double maxAttribute() {
		return max;
	}

	public void addNominalValue(String value) {
		if(type!=NOMINAL) return;
		nominalValues.addElement(new String(value));
	}

	public int getNumNominalValues() {
		if(type!=NOMINAL) return -1;
		return nominalValues.size();
	}
	public String getNominalValue(int value) {
		if(type!=NOMINAL) return null;
		return (String)nominalValues.elementAt(value);
	}

	public int convertNominalValue(String value) {
		return nominalValues.indexOf(value);
	}

	public boolean equals(Attribute attr) {
		if(!name.equals(attr.name)) return false;
		if(attr.type!=type) return false;
		if(type==NOMINAL) {
			if(!nominalValues.equals(attr.nominalValues)) 
				return false;
		}
		return true;
	}

	public void computeStatistics() {
		if(type==REAL) {
			if(globalCount>0) {
				globalMean/=globalCount;
			} else {
				valueMeans=new double[Parameters.numClasses];	
				countValues=new int[Parameters.numClasses];
			}

			for(int i=0;i<Parameters.numClasses;i++) {
				if(countValues[i]>0) {
					valueMeans[i]/=countValues[i];
				} else {
					valueMeans[i]=globalMean;
				}
			}
		} else if(type==NOMINAL) {
			int numV=nominalValues.size();

			if(globalCount==0) {
				mostFrequentValue=new int[Parameters.numClasses];
				for(int i=0;i<Parameters.numClasses;i++) {
					mostFrequentValue[i]=0;
				}
			} else {
				int max=globalValueFrequencies[0];
				int posMax=0;
				for(int i=1;i<numV;i++) {
					if(globalValueFrequencies[i]>max) {
						max=globalValueFrequencies[i];
						posMax=i;
					}
				}
				globalMostFreq=posMax;
	
				mostFrequentValue=new int[Parameters.numClasses];
				for(int i=0;i<Parameters.numClasses;i++) {
					max=valueFrequencies[i][0];
					posMax=0;
					for(int j=1;j<numV;j++){
						if(valueFrequencies[i][j]>max) {
							max=valueFrequencies[i][j];
							posMax=j;
						}
					}
					if(max>0) {
						mostFrequentValue[i]=posMax;
					} else {
						mostFrequentValue[i]=globalMostFreq;
					}
				}
			}
		}
	}

	public double getGlobalMean() {
		return globalMean;
	}

	public double getGlobalMostFreq() {
		return globalMostFreq;
	}

	public double getMeanForClass(int _class) {
		return valueMeans[_class];
	}

	public int getMostFrequentValueForClass(int _class) {
		return mostFrequentValue[_class];
	}
}
