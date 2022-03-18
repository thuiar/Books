/*
 * USDDiscretizer.java
 *
 */

/**
 *
 */

package GAssist.DiscretizationAlgorithms;

import java.util.Vector;

import GAssist.Parameters;

public class USDDiscretizer extends Discretizer {
	private class Interval {
		int attribute;
		int begin;
		int end;
		int []values;
		int []cd;
		int majority;
		double goodness;
		boolean pure;

		public Interval(int _attribute,int []_values,int _begin,int _end) {
			attribute=_attribute;
			begin=_begin;
			end=_end;
			values=_values;

			computeIntervalRatios();
		}

		void computeIntervalRatios() {
			cd=classDistribution(attribute,values,begin,end);
			int max=-1;
			int maxC=-1;
			boolean tie=false;
			int count=0;
			for(int i=0;i<Parameters.numClasses;i++) {
				if(cd[i]>max) {
					maxC=i;
					max=cd[i];
					tie=false;
				} else if(cd[i]==max) {
					tie=true;
				}
				count+=cd[i];
			}
			if(!tie) majority=maxC;
			else majority=-1;

			if(max==count) pure=true;
			else pure=false;

			goodness=max/(1.0+(count-max));
		}

		public void enlargeInterval(int newEnd) {
			end=newEnd;
			computeIntervalRatios();
		}
	}


	protected Vector discretizeAttribute(int attribute,int []values,int begin,int end) {
		Vector intervals=mergeEqualValues(attribute,values,begin,end);
		createInitialIntervals(intervals);

		boolean thereAreUnions=true;
		while(thereAreUnions) {
			thereAreUnions=false;
			int bestUnion=-1;
			double bestGoodness=0;
			Interval bestInterval=null;
			for(int i=0;i<intervals.size()-1;i++) {
				Interval int1=(Interval)intervals.elementAt(i);
				Interval int2=(Interval)intervals.elementAt(i+1);
				if(int1.majority==int2.majority || int1.majority==-1 || int2.majority==-1) {
					Interval res=new Interval(attribute,values,int1.begin,int2.end);
					if(res.goodness>(int1.goodness+int2.goodness)/2.0) {
						thereAreUnions=true;
						if(bestUnion==-1 || res.goodness>bestGoodness) {
							bestUnion=i;
							bestGoodness=res.goodness;
							bestInterval=res;
						}
					}
				}
			}
			if(thereAreUnions) {
				intervals.removeElementAt(bestUnion);
				intervals.removeElementAt(bestUnion);
				intervals.insertElementAt(bestInterval,bestUnion);
			}
		}

		Vector cutPoints=new Vector();
		for(int i=0;i<intervals.size()-1;i++) {
			Interval int1=(Interval)intervals.elementAt(i);
			Interval int2=(Interval)intervals.elementAt(i+1);
			double cutPoint=(realValues[attribute][values[int1.end]]+realValues[attribute][values[int2.begin]])/2.0;
			cutPoints.addElement(new Double(cutPoint));
		}
		return cutPoints;
	}

	void createInitialIntervals(Vector intervals) {
		int index=0;
		while(index<intervals.size()-1) {
			Interval int1=(Interval)intervals.elementAt(index);
			Interval int2=(Interval)intervals.elementAt(index+1);
			if(int1.majority==int2.majority &&
				int1.majority!=-1 && int2.majority!=-1 &&
				int1.pure && int2.pure) {
				int1.enlargeInterval(int2.end);
				intervals.removeElementAt(index+1);
			} else {
				index++;
			}
		}
	}		
				

	Vector mergeEqualValues(int attribute,int []values,int begin,int end) {
		Vector intervals = new Vector();
		int beginAnt=begin;
		double valueAnt=realValues[attribute][values[begin]];

		for(int i=begin+1;i<=end;i++) {
			double val=realValues[attribute][values[i]];
			if(val!=valueAnt) {
				intervals.addElement(new Interval(attribute,values,beginAnt,i-1));
				beginAnt=i;
				valueAnt=val;
			}
		}
		intervals.addElement(new Interval(attribute,values,beginAnt,end));
		return intervals;
	}


	int []classDistribution(int attribute,int []values,int begin,int end) {
		int []classCount = new int[Parameters.numClasses];
		for(int i=0;i<Parameters.numClasses;i++) classCount[i]=0;

		for(int i=begin;i<=end;i++) classCount[classOfInstances[values[i]]]++;
		return classCount;	
	}
		
}
