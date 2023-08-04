/**
 * classifier.java
 *
 * This is the base class for all classifiers (knowledge representations)
 */

package GAssist;

abstract public class Classifier  implements Comparable { 
	
	protected boolean isEvaluated;
	
	protected boolean bloatControlDone;
	
	protected double accuracy;
	
	protected double fitness;

	protected double exceptionsLength;

	protected double theoryLength;
	
	protected int numAliveRules;
	
	protected int positionRuleMatch;
	
	protected int numRules;
	
	/****************************************/
	protected double[] cover;
	protected double[] uncover;
	protected double[] posCorrenct;
	protected double[] negCorrenct;
	protected double[] accuracies;
	/****************************************/
	
	public abstract void initRandomClassifier();
	
	public abstract int doMatch(InstanceWrapper ins);
	
	public abstract int getNumRules();

	public abstract void deleteRules(int []whichRules);
	
	public abstract Classifier[] crossoverClassifiers(Classifier _parent2);
	   
	public abstract void doMutation();
	
	public abstract Classifier copy();
	
	public abstract void printClassifier();
	
	public static long numMatch =0;
	
	public static long numMetaMatch =0;
	
	public static void clearNumMatch() {
		numMatch =0;
		numMetaMatch = 0;
	}
	
	/******************************************************************/
	public void initAliveFlags() {
		cover = new double[numRules];
		uncover  = new double[numRules];
		posCorrenct = new double[numRules];
		negCorrenct = new double[numRules];
		accuracies =  new double[numRules];
	}
	
	
	public double getRuleAccuracies(int index){
		return accuracies[index];
	}
		
	public void setRuleAccuracies() {
		for(int index=0;index<numRules;index++){
			double a =  (posCorrenct[index]+negCorrenct[index])/PopulationWrapper.is.numInstances();
			
			double aPlus =  0;
			if(cover[index]!=0){
				aPlus = posCorrenct[index]/cover[index];
			}
			
			double aMinus =  0;
			if(uncover[index]!=0){
				aMinus = negCorrenct[index]/uncover[index]; 
			}
			//accuracies[index] =  aPlus;
			//accuracies[index] =  aPlus * cover[index] / numOfinstance;
			accuracies[index] = 0.7*aPlus+0.3*aMinus;
		}
	}
	/******************************************************************/
	
	
	public boolean getIsEvaluated() {
		return isEvaluated;
	}
	
	public void setIsEvaluated(boolean _isEvaluated) {
		isEvaluated=_isEvaluated;
	}
	
	double getAccuracy() {
		return accuracy;
	}
	
	public void setAccuracy(double _accuracy) {
		accuracy = _accuracy;
	}
	
	public double getFitness() {
		return fitness;
	}
	
	public void setFitness(double _fitness) {
		fitness = _fitness;
	}

	public double getExceptionsLength() {
		return exceptionsLength;
	}
	public void setExceptionsLength(double _exceptionsLength) {
		exceptionsLength=_exceptionsLength;
	}
   
	public int getNumAliveRules() {
		return numAliveRules;
	}
	
	public void setNumAliveRules(int _numAliveRules) {
		numAliveRules = _numAliveRules;
	}
	
	public void resetPerformance() {
		accuracy=0;
		fitness=0;
		numAliveRules=0;
		isEvaluated=false;
	}
	
	public void computePerformance() {
		accuracy=PerformanceAgent.getAccuracy();
		fitness=PerformanceAgent.getFitness(this);
		numAliveRules=PerformanceAgent.getNumAliveRules();
		isEvaluated=true;
	}

	/**
	  *	 positionRuleMatch contains the position within the classifier
	  *	 (e.g. the rule) that matched the last classified input
	  *	 instance
	  */
	public int getPositionRuleMatch() {
		return positionRuleMatch;
	}
	
	public void setPositionRuleMatch(int _positionRuleMatch) {
		positionRuleMatch = _positionRuleMatch;
	}

	public abstract double getLength();

	public double getTheoryLength() {
		return theoryLength;
	}

	public abstract double computeTheoryLength();

	/**
	 * This function returns true if this individual is better than
	 * the the individual passed as a parameter. This comparison can
	 * be based on accuracy or a combination of accuracy and size
	 */
	public boolean compareToIndividual(Classifier ind) {
		double l1=getLength();
		double l2=ind.getLength();
		double f1=getFitness();
		double f2=ind.getFitness();

		if(Parameters.doHierarchicalSelection) {
			if(Math.abs(f1-f2) <= Parameters.hierarchicalSelectionThreshold) {
				if(l1<l2) return true;
				if(l1>l2) return false;
			}
		}

		if(Parameters.useMDL==false) {
			if(f1>f2) return true;
			if(f1<f2) return false;
			if(Rand.getReal()<0.5) return true;
			return false;
		}

		if(f1<f2) return true;
		if(f1>f2) return false;
		if(Rand.getReal()<0.5) return true;
		return false;
	}
	
	public int compareTo(Object o){
		Classifier other = (Classifier) o;
		if(compareToIndividual(other)) return 1;
		else if(!compareToIndividual(other)) return -1;
		return 0;
	}
	

	public abstract int getNiche();
	public abstract int getNumNiches();

	public abstract int numSpecialStages();
	public abstract void doSpecialStage(int stage);
	
	
}
