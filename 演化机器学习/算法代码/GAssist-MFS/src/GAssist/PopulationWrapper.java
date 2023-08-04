
package GAssist;

import GAssist.Dataset.Attribute;
import GAssist.Dataset.Attributes;
import GAssist.Dataset.Instance;
import GAssist.Dataset.InstanceSet;
import GAssist.DiscretizationAlgorithms.DiscretizationManager;
import MAFS.Chromosome;

/*
 * PopulationWrapper.java
 *
 * This class contains methods that manipulate the population in various
 * ways: classifying the training set for the fitness computations, checking
 * if there are improved solutions in the population and performing the
 * test stage (generating the output files)
 */

/**
 * This class contains methods that manipulate the population in various ways.<br>
 * population的包装类
 */
public class PopulationWrapper {
	
	static public InstanceSet is;
	static public Windowing ilas;
	static public InstanceWrapper[] allInstances;

	static public int[][] instancesByClass;
	static public Sampling []samplesOfClasses;
	static public boolean smartInit;
	static public boolean cwInit;

	public static int getCurrentVersion() {
		return ilas.getCurrentVersion();
	}
	public static int numVersions() {
		return ilas.numVersions();
	}
	
	public static void initInstancesEvaluation() {
		//Added by WY, 01/08/2011
		Attributes.clear();
		
		is=new InstanceSet(Parameters.trainFile,true);
		replaceMissing(is);
		if(Parameters.adiKR) DiscretizationManager.init();
		allInstances=createWrapperInstances(is);
		ilas = new Windowing(allInstances);

		if(Parameters.initMethod!=null) {
			if(Parameters.initMethod.equalsIgnoreCase("smart")) {
				smartInit=true;
				cwInit=false;
			} else if(Parameters.initMethod.equalsIgnoreCase("cwInit")) {
				smartInit=true;
				cwInit=true;
			} else {
				smartInit=false;
				cwInit=false;
			}
		} else {
			smartInit=false;
			cwInit=false;
		}

		if(smartInit) {
			int nc=Parameters.numClasses;
			int classCounts[]= new int[nc];
			instancesByClass= new int[nc][];
			samplesOfClasses= new Sampling[nc];

			for(int i=0;i<nc;i++) {
				int num=Attributes.numInstancesOfClass(i);
				instancesByClass[i]=new int[num];
				samplesOfClasses[i]=new Sampling(num);
				classCounts[i]=0;
			}
			for(int i=0;i<allInstances.length;i++) {
				int cl=allInstances[i].classOfInstance();
				instancesByClass[cl][classCounts[cl]++]=i;
			}
		}
	}

	public static InstanceWrapper getInstanceInit(int forbiddenCL) {
		if(cwInit) {
			int cl;
			do {
				cl=Rand.getInteger(0,Parameters.numClasses-1);
			} while(cl==forbiddenCL 
					|| instancesByClass[cl].length==0);
			int pos=samplesOfClasses[cl].getSample();
			return allInstances[instancesByClass[cl][pos]];
		}

		int count[]=new int[Parameters.numClasses];
		int total=0;
		for(int i=0;i<count.length;i++) {
			if(i==forbiddenCL) count[i]=0;
			else {
				count[i]=samplesOfClasses[i].numSamplesLeft();
				total+=count[i];
			}
		}

		int pos=Rand.getInteger(0,total-1);
		int acum=0;
		for(int i=0;i<count.length;i++) {
			acum+=count[i];
			if(pos<acum) {
				int inst=samplesOfClasses[i].getSample();
				return allInstances[instancesByClass[i][inst]];
			}
		}

		LogManager.printErr("We should not be here !!!");
		System.exit(1);
		return null;
	}

	public static InstanceWrapper[] createWrapperInstances(InstanceSet is){
		InstanceWrapper []iw 
			= new InstanceWrapper[is.numInstances()];
		for(int i=0;i<iw.length;i++) {
			iw[i]=new InstanceWrapper(is.getInstance(i));
		}
		return iw;
	}

	public static boolean initIteration() {
		return ilas.newIteration();
	}

	public static void evaluateClassifier(Classifier ind) {
		int predicted, real;
		InstanceWrapper []instances=ilas.getInstances();

		ind.resetPerformance();
		PerformanceAgent.resetPerformance(ind.getNumRules());
		for(int i=0;i<instances.length;i++) {
			real=instances[i].classOfInstance();
			predicted=ind.doMatch(instances[i]);
			PerformanceAgent.addPrediction(predicted,real
				,ind.getPositionRuleMatch());
		}
				
		ind.computePerformance();

		if (Parameters.doRuleDeletion) {
			ind.deleteRules(PerformanceAgent.controlBloatRuleDeletion());
		}
	}
	
	//Added by WY, 06/01/2011
	public static double evaluateClassifierFS(Classifier cls, Chromosome fss) {
		int predicted, real;
		InstanceWrapper []instances=ilas.getInstances();

		cls.resetPerformance();
		PerformanceAgent.resetPerformance(cls.getNumRules());
		for(int i=0;i<instances.length;i++) {
			real=instances[i].classOfInstance();
			predicted=cls.doMatch(instances[i],fss);
			PerformanceAgent.addPrediction(predicted,real
				,cls.getPositionRuleMatch());
		}
				
		cls.computePerformance();

//		if (Parameters.doRuleDeletion) {
//			cls.deleteRules(PerformanceAgent.controlBloatRuleDeletion());
//		}
		
		return cls.accuracy;
	}

	public static void doEvaluation(Classifier[] _population) {
		Chronometer.startChronEvaluation();
		int popSize=_population.length;
		
		for (int i=0; i<popSize; i++) {
			if (!_population[i].getIsEvaluated()) {
				evaluateClassifier(_population[i]);
			}
		}
		
		Chronometer.stopChronEvaluation();
	}
	
	//Added by WY, 06/01/2011
	public static void doEvaluationFSS(Chromosome[] FSpopulation, Classifier bestCLS) {
		Chronometer.startChronEvaluation();
		int popSize=FSpopulation.length;
		
		for (int i=0; i<popSize; i++) {
			double acc = evaluateClassifierFS(bestCLS, FSpopulation[i]);
			FSpopulation[i].calFitness(acc);
		}
		
		Chronometer.stopChronEvaluation();
	}
	
	//Added by WY, 06/01/2011
	public static void doEvaluationCLS(Classifier[] CLpopulation, Chromosome bestFSS) {
		Chronometer.startChronEvaluation();
		int popSize=CLpopulation.length;
		
		for (int i=0; i<popSize; i++) {
			evaluateClassifierFS(CLpopulation[i], bestFSS);
		}
		
		Chronometer.stopChronEvaluation();
	}
	
	/**
	 *  Obtains the best classifier of population.
	 *  @return Best classifier of population.
	 */
	public static Classifier getBest(Classifier[] _population) {
		int sizePop=_population.length;
		int posWinner=0;
		
		for (int i=1; i<sizePop; i++) {
			if(_population[i].compareToIndividual(_population[posWinner])) posWinner=i;
		}
		return _population[posWinner];
	}

	/**
	 *  Obtains the worst classifier of population.
	 *  @return Worst classifier of population.
	 */
	public static int getWorst(Classifier[] _population) {
		int sizePop=_population.length;
		int posWorst=0;
		
		for (int i=1; i<sizePop; i++) {
			if(!_population[i].compareToIndividual(_population[posWorst])) {
				posWorst=i;
			}
		}
		return posWorst;
	}
	
	public static void setModified(Classifier[] _population) {
		int sizePop=_population.length;
		
		for (int i=0; i<sizePop; i++) {
			_population[i].setIsEvaluated(false);
		}
	}

	
	public static void testClassifier(Classifier ind,String typeOfTest,String testInputFile) {
		InstanceSet testSet = new InstanceSet(testInputFile,false);
		replaceMissing(testSet);
		InstanceWrapper []instances=createWrapperInstances(testSet);
		double numInstances=instances.length;
		int real, predicted;
		PerformanceAgent.resetPerformanceTest(ind.getNumRules());
		ind.resetPerformance();

		for(int i=0;i<numInstances;i++) {
			real=instances[i].classOfInstance();
			predicted=ind.doMatch(instances[i]);
			PerformanceAgent.addPredictionTest(predicted,real
				,ind.getPositionRuleMatch());
		}

		LogManager.println("\nStatistics on "+typeOfTest+" file");
		PerformanceAgent.dumpStats(typeOfTest);
		LogManager.println("");
	}
	
	//Added by WY, 06/01/2011
	public static void testClassifierFS(Classifier ind, Chromosome fss, String typeOfTest,String testInputFile) {
		InstanceSet testSet = new InstanceSet(testInputFile,false);
		replaceMissing(testSet);
		InstanceWrapper []instances=createWrapperInstances(testSet);
		double numInstances=instances.length;
		int real, predicted;
		PerformanceAgent.resetPerformanceTest(ind.getNumRules());
		ind.resetPerformance();

		for(int i=0;i<numInstances;i++) {
			real=instances[i].classOfInstance();
			predicted=ind.doMatch(instances[i], fss);
			PerformanceAgent.addPredictionTest(predicted,real
				,ind.getPositionRuleMatch());
		}

		LogManager.println("\nStatistics on "+typeOfTest+" file");
		PerformanceAgent.dumpStats(typeOfTest);
		LogManager.println("");
	}

	static void replaceMissing(InstanceSet is) {
		int [][]mostFreq = new int[Parameters.numAttributes][Parameters.numClasses];
		double [][]means = new double[Parameters.numAttributes][Parameters.numClasses];
		int []types = new int[Parameters.numAttributes];

		for(int i=0;i<Parameters.numAttributes;i++) {
			Attribute attr=Attributes.getAttribute(i);
			types[i]=attr.getType();
			if(types[i]==Attribute.NOMINAL) {
				for(int j=0;j<Parameters.numClasses;j++) {
					mostFreq[i][j]=attr.getMostFrequentValueForClass(j);
				}
			} else {
				for(int j=0;j<Parameters.numClasses;j++) {
					means[i][j]=attr.getMeanForClass(j);
				}
			}
		}

		Instance []inst=is.getInstances();
		for(int i=0;i<inst.length;i++) {
			if(inst[i].hasMissingValues()) {
				int cl=inst[i].getInstanceClass();
				boolean []miss=inst[i].areAttributesMissing();
				for(int j=0;j<Parameters.numAttributes;j++) {
					if(miss[j]) {
						if(types[j]==Attribute.NOMINAL) {
							inst[i].setNominalAttribute(j,mostFreq[j][cl]);
						} else {
							inst[i].setRealAttribute(j,means[j][cl]);
						}
					}
				}
			}
		}

	}
}
