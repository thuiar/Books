package GAssist;

import java.util.*;

/**
 *	Windowing.java - Class that manages the subset of training instances
 *	that is used at each iteration to perform the fitness computations
 */

/**
 * 滑动窗口类，用于定义在某轮迭代中用于fitness评估的训练样本集
 */
public class Windowing {
	InstanceWrapper []is;
	InstanceWrapper [][]strata;
	int numStrata;
	int currentIteration;
	boolean lastIteration;

	public Windowing(InstanceWrapper []_is) {
		is = _is;
		numStrata = Parameters.numStrata;
		strata = new InstanceWrapper[numStrata][];
		currentIteration=0;
		lastIteration=false;

		createStrata();
	}

	private void createStrata() {
		Vector []tempStrata = new Vector[numStrata];
		Vector []instancesOfClass = new Vector[Parameters.numClasses];

		for(int i=0;i<numStrata;i++) tempStrata[i]=new Vector();
		for(int i=0;i<Parameters.numClasses;i++) 
			instancesOfClass[i]=new Vector();

		int numInstances=is.length;
		for(int i=0;i<numInstances;i++) {
			int cl=is[i].classOfInstance();
			instancesOfClass[cl].addElement(is[i]);
		}

		for(int i=0;i<Parameters.numClasses;i++) {
			int stratum=0;
			int count=instancesOfClass[i].size();
			while(count>=numStrata) {
				int pos=Rand.getInteger(0,count-1);
				tempStrata[stratum].addElement(instancesOfClass[i].elementAt(pos));
				instancesOfClass[i].removeElementAt(pos);
				stratum=(stratum+1)%numStrata;
				count--;
			}
			while(count>0) {
				stratum=Rand.getInteger(0,numStrata-1);
				tempStrata[stratum].addElement(instancesOfClass[i].elementAt(0));
				instancesOfClass[i].removeElementAt(0);
				count--;
			}
		}

		for(int i=0;i<numStrata;i++) {
			int num=tempStrata[i].size();
			strata[i]=new InstanceWrapper[num];
			for(int j=0;j<num;j++) {
				strata[i][j]=(InstanceWrapper)tempStrata[i].elementAt(j);
			}
		}
	}

	public boolean newIteration() {
		currentIteration++;
		if(currentIteration==Parameters.numIterations) 
			lastIteration=true;

		if(numStrata>1) return true;
		return false;
	}

	public InstanceWrapper []getInstances() {
		if(lastIteration) {
			return is;
		}
		return strata[currentIteration%numStrata];
	}

	public int numVersions() {
		if(lastIteration) return 1;
		return numStrata;
	}

	public int getCurrentVersion() {
		if(lastIteration) return 0;
		return currentIteration%numStrata;
	}
		
}	
		
