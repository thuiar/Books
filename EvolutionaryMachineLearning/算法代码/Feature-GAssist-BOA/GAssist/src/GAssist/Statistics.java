/**
 * Class that computes and stores several statistics about the learning process
 */

package GAssist;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

public class Statistics {

	public static double[] averageFitness;
	public static double[] averageAccuracy;
	public static double[] bestAccuracy;
	public static double[] bestRules;
	public static double[] bestAliveRules;
	public static double[] averageNumRules;
	public static double[] averageNumRulesUtils;

	public static int iterationsSinceBest = 0;
	public static double bestFitness;
	public static double last10IterationsAccuracyAverage;

	public static int countStatistics = 0;

	public static double[] statAttr;

	public static void resetBestStats() {
		iterationsSinceBest = 0;
	}

	public static int getIterationsSinceBest() {
		return iterationsSinceBest;
	}

	public static void bestOfIteration(double itBestFit) {
		if (iterationsSinceBest == 0) {
			bestFitness = itBestFit;
			iterationsSinceBest++;
		} else {
			boolean newBest = false;
			if (Parameters.useMDL) {
				if (itBestFit < bestFitness)
					newBest = true;
			} else {
				if (itBestFit > bestFitness)
					newBest = true;
			}

			if (newBest) {
				bestFitness = itBestFit;
				iterationsSinceBest = 1;
			} else {
				iterationsSinceBest++;
			}
		}

		int i = countStatistics - 9;
		if (i < 0)
			i = 0;
		int max = countStatistics + 1;
		int num = max - i;
		last10IterationsAccuracyAverage = 0;
		for (; i < max; i++)
			last10IterationsAccuracyAverage += bestAccuracy[i];
		last10IterationsAccuracyAverage /= (double) num;
	}

	public static void initStatistics() {
		countStatistics = 0;

		Chronometer.startChronStatistics();

		int numStatistics = Parameters.numIterations;

		averageFitness = new double[numStatistics];
		averageAccuracy = new double[numStatistics];
		bestAccuracy = new double[numStatistics];
		bestRules = new double[numStatistics];
		bestAliveRules = new double[numStatistics];
		averageNumRules = new double[numStatistics];
		averageNumRulesUtils = new double[numStatistics];

		Chronometer.stopChronStatistics();
	}

	public static void statisticsToFile() {
		FileManagement file = new FileManagement();
		int length = countStatistics;
		String line;
		String lineToWrite = "";
		try {
			// file.initWrite("NumRules.txt");

			// TODO

			// file.closeWrite();
		} catch (Exception e) {
			LogManager.println("Error in statistics file");
		}
	}

	public static void computeStatistics(Classifier[] _population) {
		Chronometer.startChronStatistics();
		int populationLength = Parameters.popSize;
		Classifier classAct;
		double sumFitness = 0;
		double sumAccuracy = 0;
		double sumNumRules = 0;
		double sumNumRulesUtils = 0;

		for (int i = 0; i < populationLength; i++) {
			classAct = _population[i];
			sumFitness += classAct.getFitness();
			sumAccuracy += classAct.getAccuracy();
			sumNumRules += classAct.getNumRules();
			sumNumRulesUtils += classAct.getNumAliveRules();
		}
		
//		if (countStatistics % Parameters.REDUCE_INTERVAL == Parameters.REDUCE_INTERVAL-1
//				||countStatistics == Parameters.numIterations -2) {
//			computeRedundancy(_population);
//		}
		 
		
		sumFitness = sumFitness / populationLength;
		sumAccuracy = sumAccuracy / populationLength;
		sumNumRules = sumNumRules / populationLength;
		sumNumRulesUtils = sumNumRulesUtils / populationLength;

		if (countStatistics >= averageFitness.length) {
			System.out.println();
		}
		Statistics.averageFitness[countStatistics] = sumFitness;
		Statistics.averageAccuracy[countStatistics] = sumAccuracy;
		Statistics.averageNumRules[countStatistics] = sumNumRules;
		Statistics.averageNumRulesUtils[countStatistics] = sumNumRulesUtils;

		Classifier best = PopulationWrapper.getBest(_population);
		LogManager.println("Best of iteration " + countStatistics + " : "
				+ best.getAccuracy() + " " + best.getFitness() + " "
				+ best.getNumRules() + "(" + best.getNumAliveRules() + ")\tAve:"+sumAccuracy);
	
		if (countStatistics % Parameters.PRINT_INTERVAL == 0) {
			LogManager.println_file("Best of iteration " + countStatistics
					+ " : " + best.getAccuracy() + " " + best.getFitness()
					+ " " + best.getNumRules() + "(" + best.getNumAliveRules()
					+ ") " + Classifier.numMatch + " "+Classifier.numMetaMatch);
		}

		Statistics.bestAccuracy[countStatistics] = best.getAccuracy();
		Statistics.bestRules[countStatistics] = best.getNumRules();
		Statistics.bestAliveRules[countStatistics] = best.getNumAliveRules();
		bestOfIteration(best.getFitness());

		countStatistics++;
		Chronometer.stopChronStatistics();
	}

	public static void computeRedundancy(Classifier[] _population) {
		int nrRule = 0, base = 0;
		statAttr = new double[Globals_GABIL.size.length];
		for (int i = 0; i < Parameters.popSize; i++) {
			nrRule += _population[i].numRules;
			for (int j = 0; j < _population[i].numRules; j++) {
				base = j * Globals_GABIL.ruleSize;
				int interval = 0;
				for (int k = 0; k < Globals_GABIL.size.length; k++) {
					boolean suc = true;
					for (int l = 1; l < Globals_GABIL.size[k]; l++) {
						if (((ClassifierGABIL) _population[i]).crm[base
								+ interval + l] != ((ClassifierGABIL) _population[i]).crm[base
								+ interval + l - 1]) {
							suc = false;
							break;
						}
					}
					if (suc) {
						statAttr[k]++;
					}
					interval += Globals_GABIL.size[k];
				}
			}
		}

		DecimalFormat df = new DecimalFormat("0.000");
		// for (int j = 0; j < length; j++) {
		// statBit[j] /= nrRule;
		// }

		for (int j = 0; j < Globals_GABIL.size.length; j++) {
			statAttr[j] /= nrRule;
			System.out.print(df.format(statAttr[j]) + "\t");
			LogManager.print_file(df.format(statAttr[j]) + "\t");
		}
		System.out.println();
		LogManager.println_file("");

		// if(countStatistics == Parameters.numIterations -2){
		// for (int j = 0; j < Globals_GABIL.size.length; j++) {
		// LogManager.print_file(df.format(statAttr[j]) + "\t");
		// }
		// LogManager.print_file("\n");
		// }
	}
	
	public static int uselessAttr() {
		double max = Double.MIN_VALUE;
		int attr=-1;
		for(int i=0;i<statAttr.length ;i++){
			if(statAttr[i]>max && statAttr[i]>Parameters.attributeThrehold){
				max = statAttr[i]; 
				attr =  i;
			}
		}
		//LogManager.println_file(attr+"\t"+max);
		//System.out.println(attr+"\t"+max);
		return attr;
	}

	public static int[] uselessAttrs() {
		List<Integer> attrs =new ArrayList<Integer>();
		for(int i=0;i<statAttr.length ;i++){
			if(statAttr[i]>Parameters.attributesThrehold){
				attrs.add(i);
			}
		}
		//LogManager.println_file(attr+"\t"+max);
		//System.out.println(attr+"\t"+max);
		int length = attrs.size();
		int[] uselessattrs =new int[length];
		int i=0;
		for(Integer index:attrs){
			uselessattrs[i++]=index;
		}
		return uselessattrs;
	}
	
	public static void clearStatistics() {
		
		Chronometer.startChronStatistics();

		int numStatistics = Parameters.numIterations;

		averageFitness = new double[numStatistics];
		averageAccuracy = new double[numStatistics];
		bestAccuracy = new double[numStatistics];
		bestRules = new double[numStatistics];
		bestAliveRules = new double[numStatistics];
		averageNumRules = new double[numStatistics];
		averageNumRulesUtils = new double[numStatistics];
		
		statAttr = new double[Globals_GABIL.size.length];

		Chronometer.stopChronStatistics();
	}
}
