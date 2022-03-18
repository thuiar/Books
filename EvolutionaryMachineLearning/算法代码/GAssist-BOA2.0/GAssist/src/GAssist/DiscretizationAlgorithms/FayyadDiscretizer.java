/*
 * FayyadDiscretizer.java
 *
 */

/**
 *
 */

package GAssist.DiscretizationAlgorithms;

import java.util.Vector;

import GAssist.Parameters;

public class FayyadDiscretizer extends Discretizer {
	protected Vector discretizeAttribute(int attribute,int []values,int begin,int end) {
		Vector cd=classDistribution(attribute,values,begin,end);
		if(cd.size()==1) return new Vector();
		int numValues=sumValues(cd);
		double entAll=computeEntropy(cd,numValues);

		Vector candidateCutPoints = getCandidateCutPoints(attribute,values,begin,end);
		if(candidateCutPoints.size()==0) return new Vector();

		int posMin=((Integer)candidateCutPoints.elementAt(0)).intValue();
		double entMin=computePartitionEntropy(attribute,values,begin,posMin,end);
		for(int i=1,size=candidateCutPoints.size();i<size;i++) {
			int pos=((Integer)candidateCutPoints.elementAt(i)).intValue();
			double ent=computePartitionEntropy(attribute,values,begin,pos,end);
			if(ent<entMin) {
				entMin=ent;
				posMin=pos;
			}
		}

		if(entMin<entAll && doCutPoint(attribute,values,begin,posMin,end)) {
			Vector res1=discretizeAttribute(attribute,values,begin,posMin-1);
			double cutPoint=(realValues[attribute][values[posMin-1]]+realValues[attribute][values[posMin]])/2.0;
			res1.addElement(new Double(cutPoint));
			Vector res2=discretizeAttribute(attribute,values,posMin,end);
			res1.addAll(res2);
			return res1;
		}
		return new Vector();
	}

	boolean doCutPoint(int attribute,int []values,int begin,int midPoint,int end) {
		Vector cd=classDistribution(attribute,values,begin,end);
		Vector cd1=classDistribution(attribute,values,begin,midPoint-1);

		Vector cd2=classDistribution(attribute,values,midPoint,end);

		int numValues=sumValues(cd);
		int numValues1=sumValues(cd1);
                int numValues2=sumValues(cd2);

		double entAll=computeEntropy(cd,numValues);
		double ent1=computeEntropy(cd1,numValues1);
		double ent2=computeEntropy(cd2,numValues2);
		double entPart=((double)numValues1*ent1+(double)numValues2*ent2)/(double)(numValues1+numValues2);

		double gain=entAll-entPart;
		double incPart=log2(Math.pow(3,cd.size())-2)-(cd.size()*entAll-cd1.size()*ent1-cd2.size()*ent2);
		if(gain<(log2(numValues-1)+incPart)/(double)numValues) {
			return false;
		}
		return true;
	}

	public double log2(double value) {
		return Math.log(value)/Math.log(2);
	}

	double computePartitionEntropy(int attribute,int []values,int begin,int midPoint,int end) {
		Vector cd1=classDistribution(attribute,values,begin,midPoint-1);
		Vector cd2=classDistribution(attribute,values,midPoint,end);

		int numValues1=sumValues(cd1);
		int numValues2=sumValues(cd2);

		double ent1=computeEntropy(cd1,numValues1);
		double ent2=computeEntropy(cd2,numValues2);
		return ((double)numValues1*ent1+(double)numValues2*ent2)/(double)(numValues1+numValues2);
	}

	double computeEntropy(Vector v,int numValues) {
		double ent=0;

		for(int i=0,size=v.size();i<size;i++) {
			double prob=((Integer)v.elementAt(i)).intValue();
			prob/=(double)numValues;
			ent+=prob*Math.log(prob)/Math.log(2);
		}
		return -ent;
	}

	int sumValues(Vector v) {
		int sum=0;
		for(int i=0,size=v.size();i<size;i++) {
			sum+=((Integer)v.elementAt(i)).intValue();
		}
		return sum;
	}

	Vector getCandidateCutPoints(int attribute,int []values,int begin,int end) {
		Vector cutPoints = new Vector();
		double valueAnt=realValues[attribute][values[begin]];

		for(int i=begin;i<=end;i++) {
			double val=realValues[attribute][values[i]];
			if(val!=valueAnt) cutPoints.addElement(new Integer(i));
			valueAnt=val;
		}
		return cutPoints;
	}


	Vector classDistribution(int attribute,int []values,int begin,int end) {
		int []classCount = new int[Parameters.numClasses];
		for(int i=0;i<Parameters.numClasses;i++) classCount[i]=0;

		for(int i=begin;i<=end;i++) classCount[classOfInstances[values[i]]]++;
		
		Vector res= new Vector();
		for(int i=0;i<Parameters.numClasses;i++) {
			if(classCount[i]>0) res.addElement(new Integer(classCount[i]));
		}

		return res;
	}
		
}
