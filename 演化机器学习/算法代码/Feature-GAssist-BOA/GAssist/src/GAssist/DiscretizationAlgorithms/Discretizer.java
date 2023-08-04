/*
 * ChiMergeDiscretizer.java
 *
 */

/**
 *
 */

package GAssist.DiscretizationAlgorithms;

import java.util.ArrayList;
import java.util.Vector;

import GAssist.Parameters;
import GAssist.Dataset.Attribute;
import GAssist.Dataset.Attributes;
import GAssist.Dataset.Instance;
import GAssist.Dataset.InstanceSet;
import boa.util.MyUtil;

abstract public class Discretizer {
	double [][]cutPoints;
	double [][]realValues;
	boolean []realAttributes;
	int []classOfInstances;
	
	public void buildCutPoints(InstanceSet is) {
		Instance []instances=is.getInstances();

		classOfInstances= new int[instances.length];
		for(int i=0;i<instances.length;i++) 
			classOfInstances[i]=instances[i].getInstanceClass();
		
		cutPoints=new double[Parameters.numAttributes][];
		realAttributes = new boolean[Parameters.numAttributes];
		realValues = new double[Parameters.numAttributes][];
		for(int i=0;i<Parameters.numAttributes;i++) {
			Attribute at=Attributes.getAttribute(i);
			if(at.getType()==Attribute.REAL) {
				realAttributes[i]=true;

				realValues[i] = new double[instances.length];
				int []points= new int[instances.length];
				int numPoints=0;
				for(int j=0;j<instances.length;j++) {
					if(!instances[j].areAttributesMissing()[i]) {
						points[numPoints++]=j;
						realValues[i][j]=instances[j].getRealAttribute(i);
					}
				}

				if(numPoints>0) {

					sortValues(i,points,0,numPoints-1);
	
					Vector cp=discretizeAttribute(i,points,0,numPoints-1); 
					if(cp.size()>0) {
						cutPoints[i]=new double[cp.size()];
						for(int j=0;j<cutPoints[i].length;j++) {
							cutPoints[i][j]=((Double)cp.elementAt(j)).doubleValue();
							//LogManager.println("Cut point "+j+" of attribute "+i+" : "+cutPoints[i][j]);
						}
					} else {
						cutPoints[i]=null;
					}
					//LogManager.println("Number of cut points of attribute "+i+" : "+cp.size());
				} else {
					cutPoints[i]=null;
				}
			} else {
				realAttributes[i]=false;
			}
		}
	}

	protected void sortValues(int attribute,int []values,int begin,int end) {
		double pivot;
		int temp;
		int i,j;

		i=begin;j=end;
		pivot=realValues[attribute][values[(i+j)/2]];
		do {
			while(realValues[attribute][values[i]]<pivot) i++;
			while(realValues[attribute][values[j]]>pivot) j--;
			if(i<=j) {
				if(i<j) {
					temp=values[i];
					values[i]=values[j];
					values[j]=temp;
				}
				i++; j--;
			}
		} while(i<=j);
		if(begin<j) sortValues(attribute,values,begin,j);
		if(i<end) sortValues(attribute,values,i,end);
	}

	public int getNumIntervals(int attribute) {
		if(cutPoints[attribute]==null) return 1;
		return cutPoints[attribute].length+1;
	}

	public double getCutPoint(int attribute,int cp) {
		return cutPoints[attribute][cp];
	}

	protected abstract Vector discretizeAttribute(int attribute,int []values,int begin,int end) ;

	public int discretize(int attribute,double value) {
		if(cutPoints[attribute]==null) return 0;
		for(int i=0;i<cutPoints[attribute].length;i++)
			if(value<cutPoints[attribute][i]) return i;
		return cutPoints[attribute].length;
	}
	
	public double computeChi(int attribute, int startInterval, int endInterval,InstanceSet is){
		Instance []instances=is.getInstances();
		
		
		
		int numClass = 2;
		int numInterval=getNumIntervals(attribute);
		
		double[][]  matrix= new double[numClass][numInterval];
		double[]	classMatrix= new double[numClass];
		double[]	intervalMatrix= new double[numInterval];
		int counter=0;
		//Joint Probability
		for(int i=0;i<instances.length ;i++){
			Instance instance=instances[i];
			int cls =instance.getInstanceClass();
			int interval = discretize(attribute,instance.getRealAttribute(attribute));
			matrix[cls][interval]+=1;
			counter++;
		}
		if(counter!=instances.length) System.out.println("MI ERROR");
		
		//Marginal Probability
		//Cj
		for(int i=0;i<numClass;i++){
			for(int j=startInterval;j<=endInterval;j++){
				classMatrix[i]+=matrix[i][j];
			}
		}
		int N = 0;
		for(int i=startInterval;i<=endInterval;i++){
			for(int j=0;j<numClass;j++){
				intervalMatrix[i]+=matrix[j][i];
			}
			N += intervalMatrix[i];
		}
		 
		//Chi
		double chi = 0 ;
		for(int i=0;i<numClass;i++){
				for(int j=startInterval;j<=endInterval;j++){
					double e = 0.1;
					if(classMatrix[i]!=0 && intervalMatrix[j]!=0){
						e =	 (double)classMatrix[i]*(double)intervalMatrix[j] / (double)N;
					}
					chi += ((double)matrix[i][j] - e)*((double)matrix[i][j] - e)/e;
				}
		}
		double c = Parameters.chi[numClass-1];
//		System.out.print("("+c /amount/2+")");
//		System.out.print(Math.abs(mutualInfo)> c /(amount*2));
//		System.out.print(" ");
		//return mutualInfo/jointEntropy;
		return chi/c ;
	}
	
	
	public double computeIR(int attribute, int startInterval, int endInterval,InstanceSet is,int numInterval){
		Instance []instances=is.getInstances();
		
		double mutualInfo=0, jointEntropy=0;
		int amount=0;
		
		int numClass = 2;
//		int numInterval=getNumIntervals(attribute);
		
		double[][]  matrix= new double[numClass][numInterval];
		double[]	classMatrix= new double[numClass];
		double[]	intervalMatrix= new double[numInterval];
		int counter=0;
		//Joint Probability
		for(int i=0;i<instances.length ;i++){
			Instance instance=instances[i];
			int cls =instance.getInstanceClass();
			int interval = discretize(attribute,instance.getRealAttribute(attribute));
			
			matrix[cls][interval]+=1/(double)instances.length;
			counter++;
		}
		if(counter!=instances.length) System.out.println("MI ERROR");
		
		//Marginal Probability
		for(int i=0;i<numClass;i++){
			for(int j=startInterval;j<=endInterval;j++){
				classMatrix[i]+=matrix[i][j];
				amount+=(int)(matrix[i][j]*instances.length+0.00001);
			}
		}
		for(int i=startInterval;i<=endInterval;i++){
			for(int j=0;j<numClass;j++){
				intervalMatrix[i]+=matrix[j][i];
			}
		}
		
		//Mutual Information
		for(int i=0;i<numClass;i++){
			if(classMatrix[i]!=0){
				for(int j=startInterval;j<=endInterval;j++){
					if(intervalMatrix[j]!=0&&matrix[i][j]!=0){
			
						mutualInfo+= matrix[i][j]* 
							Math.log(matrix[i][j]/(classMatrix[i]*intervalMatrix[j])) ;
				
						jointEntropy+= -matrix[i][j]* 
							Math.log(matrix[i][j]) ;
					}
				}
			}
		}
		double c = Parameters.chi[numInterval-1];
//		System.out.print("("+c /amount/2+")");
//		System.out.print(Math.abs(mutualInfo)> c /(amount*2));
//		System.out.print(" ");
		//return mutualInfo/jointEntropy;
		return mutualInfo / (c /(amount*2)) ;
		
	}
	
	
	public double computeCAIM(int attribute,InstanceSet is,int numInterval){
		Instance []instances=is.getInstances();
		int numClass = 2;
		int[][]  matrix= new int[numClass][numInterval];
		int[]	classMatrix= new int[numClass];
		int[]	intervalMatrix= new int[numInterval];
		
		//Joint Probability
		for(int i=0;i<instances.length ;i++){
			Instance instance=instances[i];
			int cls =instance.getInstanceClass();
			int interval = discretize(attribute,instance.getRealAttribute(attribute));
			matrix[cls][interval]+=1;
		}
		
		//Marginal Probability
		for(int i=0;i<numClass;i++){
			for(int j=0;j<numInterval;j++){
				classMatrix[i]+=matrix[i][j];
			}
		}
		for(int i=0;i<numInterval;i++){
			for(int j=0;j<numClass;j++){
				intervalMatrix[i]+=matrix[j][i];
			}
		}
		double CAIM = 0;
		for(int i=0;i<numInterval;i++){
			int max = Integer.MIN_VALUE;
			int classIndex = -1;
			for(int j=0;j<numClass;j++){
				if(matrix[j][i] > max){
					max = matrix[j][i];
					classIndex = j;
				}
			}
			CAIM += (double)max /(double)intervalMatrix[i];
		}
		return CAIM/numInterval;
	}

	public void CAIMmergeIntervals(int attribute, InstanceSet is){
		int orignalLength = cutPoints[attribute].length;
		
		double[] CAIM =new double[cutPoints[attribute].length-1];
		double max = Integer.MIN_VALUE;
		int flag = -1, reduce =0;
		boolean terminal = false;
		
		while(!terminal){
			int currentLength = orignalLength- reduce;
			double[] orginalCP =  new double[currentLength];
			System.arraycopy(cutPoints[attribute], 0, orginalCP, 0, currentLength);
			terminal = true;
			for(int i=0;i<cutPoints[attribute].length-1;i++){
				cutPoints[attribute] =new double[currentLength-1];
				
				System.arraycopy(orginalCP,0,cutPoints[attribute],0,i);
				System.arraycopy(orginalCP,i+1,cutPoints[attribute],i, currentLength-i-1);
				
				CAIM[i] = computeCAIM(attribute, is, currentLength);
				if(CAIM[i]>max){
					max =CAIM[i];
					flag = i; 
					terminal = false;
				}
				System.out.println(i+":"+CAIM[i]);	
			}
			//
			System.out.println(flag+":"+CAIM[flag]);	
			cutPoints[attribute] =new double[currentLength-1];
			System.arraycopy(orginalCP,0,cutPoints[attribute],0,flag);
			System.arraycopy(orginalCP,flag+1,cutPoints[attribute],flag, currentLength-flag-1);
			reduce++;
		}
	}
	
	
	
	public void mergeInterval(int attribute, int interval){
		double[] currentCutPoints =cutPoints[attribute];
		double[] newCutPoints =new double[cutPoints[attribute].length-1];
		if(interval>=1)
			System.arraycopy(currentCutPoints, 0, newCutPoints, 0, interval);
		if(interval<currentCutPoints.length-1)
			System.arraycopy(currentCutPoints, interval+1, newCutPoints, interval, currentCutPoints.length-1-interval);
		
		cutPoints[attribute] = newCutPoints;
		
	}
	
//	public void computeIR(InstanceSet is){
//		for(int i=0;i<Attributes.getNumAttributes()-1;i++){
//			for(int j=0;j<getNumIntervals(i)-1;j++){
//				double ir = computeIR(i,j,j+1,is);
//			}
//			System.out.println();
//		}
//	}

	
	
	public void mergeIntervals(int attr,InstanceSet is){
		if(getNumIntervals(attr)<3) return;
		
		ArrayList<Double> IRList =new ArrayList<Double>();
/////////////////////////////////////////
		int numInterval=getNumIntervals(attr);
//		System.out.println(attr+" "+attr);
/////////////////////////////////////////

		for(int j=0;j<getNumIntervals(attr)-1;j++){
			//double ir = computeIR(attr,j,j+1,is,numInterval);
			double ir = computeChi(attr,j,j+1,is);
			IRList.add(ir);
		}
		
		double min=Double.MAX_VALUE;	
		double tempCP[];
		do{
			min=Double.MAX_VALUE;
			int index =-1;
			for(int i=0;i<IRList.size();i++){
				if(IRList.get(i)<min){
					min = IRList.get(i); index = i;
				}
			}
			if(index==-1){
				break;
			}else{
	//			System.out.println("merge ");
			}
			mergeInterval(attr,index);
			numInterval--;
			
			if(index+1<IRList.size())	
				IRList.remove(index+1);
			IRList.remove(index);
			if(index-1>=0)
				IRList.remove(index-1);
			if(index-1>=0)
				IRList.add(index-1,computeChi(attr, index-1, index,is));
			if(index<getNumIntervals(attr)-1)
				IRList.add(index,computeChi(attr, index, index+1,is));
		}while(min<1 && min>=0 && IRList.size()>1);
		
		
	}
	
	public void mergeIntervals(InstanceSet is){
		for(int i=0;i<Attributes.getNumAttributes()-1;i++){
			//CAIMmergeIntervals(i,is);
			mergeIntervals(i,is);
			
		}
		System.out.println(consistency(is));
	}
	
	public void adptivemergeIntervals(InstanceSet is){
		double[][] bakCP;
		do{
			//bakCP =
			for(int i=0;i<Attributes.getNumAttributes()-1;i++){
				//CAIMmergeIntervals(i,is);
				mergeIntervals(i,is);
			}
		}while(consistency(is));
		//CP = bakCP;
	}
	
	
	
	public double[][] getCutPoints(){
		return cutPoints;
	}
	
	public boolean consistency(InstanceSet is){
		int counter=0;
		Instance []instances=is.getInstances();
		int indexOfClass = Attributes.getNumAttributes()-1;
		int[][] data=new int[instances.length][Attributes.getNumAttributes()];
		for(int i=0;i<instances.length ;i++){
			Instance instance=instances[i];
			int cls =instance.getInstanceClass();
			data[i][indexOfClass]=cls;
			for(int j=0;j<Attributes.getNumAttributes()-1;j++){
				int interval = discretize(j,instance.getRealAttribute(j));
				data[i][j]=interval;
			}
		}
		for(int i=0;i<instances.length-1 ;i++){
			for(int j=i+1;j<instances.length ;j++){
				if(MyUtil.isSameArray(data[i],data[j],indexOfClass)){
					if(data[i][indexOfClass]!=data[j][indexOfClass]){
						//MyUtil.printArray(data[i]);
						//MyUtil.printArray(data[j]);
						
						System.out.println(i+"\t"+j);
						System.out.print(i+"\t"+instances[i]);
						System.out.println("\t"+j+"\t"+instances[j]);
						
						//System.out.println(MyUtil.isSameInstance(instances[i],instances[j]));
						counter++;
						//return false;
					}
				}
			}
		}
		System.out.println("Inconsistancy\t"+counter);
		return true;
	}
}
