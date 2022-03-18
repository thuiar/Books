/**
 * Class that computes and stores several statistics about the learning process
 */

package GAssist;

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
		//Added by WY, 08/01/2011
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
		sumFitness = sumFitness / populationLength;
		sumAccuracy = sumAccuracy / populationLength;
		sumNumRules = sumNumRules / populationLength;
		sumNumRulesUtils = sumNumRulesUtils / populationLength;

		Statistics.averageFitness[countStatistics] = sumFitness;
		Statistics.averageAccuracy[countStatistics] = sumAccuracy;
		Statistics.averageNumRules[countStatistics] = sumNumRules;
		Statistics.averageNumRulesUtils[countStatistics] = sumNumRulesUtils;

		Classifier best = PopulationWrapper.getBest(_population);
		LogManager.println("Best of iteration " + countStatistics + " : "
				+ best.getAccuracy() + " " + best.getFitness() + " "
				+ best.getNumRules() + "(" + best.getNumAliveRules() + ")");
		
		//Added by WY, 07/01/2011
		if (countStatistics % 10 == 0) {
			LogManager.println_file("Best of iteration " + countStatistics
					+ " : " + best.getAccuracy() + " " + best.getFitness()
					+ " " + best.getNumRules() + "(" + best.getNumAliveRules()
					+ ")");
		}
		
		Statistics.bestAccuracy[countStatistics] = best.getAccuracy();
		Statistics.bestRules[countStatistics] = best.getNumRules();
		Statistics.bestAliveRules[countStatistics] = best.getNumAliveRules();
		bestOfIteration(best.getFitness());

		countStatistics++;
		Chronometer.stopChronStatistics();
	}

}
