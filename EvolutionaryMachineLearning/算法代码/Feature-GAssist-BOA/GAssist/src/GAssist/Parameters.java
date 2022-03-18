/*
 * Parameters.java
 *
 * This class contains all the parameters of the system
 */
package GAssist;

public class Parameters {
	public static String algorithmName;

	public static double confidenceThreshold;
	public static int numIntervals;

	public static int numClasses;
	public static int numAttributes;

	public static int popSize;
	public static int initialNumberOfRules;
	public static double probCrossover;
	public static double probMutationInd;
	public static int tournamentSize;
	public static int numIterations;
	public static double percentageOfLearning;

	public static boolean useMDL;
	public static int iterationMDL;
	public static double initialTheoryLengthRatio;
	public static double weightRelaxFactor;
	public static double theoryWeight;

	public static double probOne;

	public static String trainFile;
	public static String testFile;
	public static long seed = -1;

	public static boolean doRuleDeletion = false;
	public static int iterationRuleDeletion;
	public static int ruleDeletionMinRules;
	public static boolean doHierarchicalSelection = false;
	public static int iterationHierarchicalSelection;
	public static double hierarchicalSelectionThreshold;
	public static int sizePenaltyMinRules;

	public static int numStrata;

	public static String discretizer1;
	public static String discretizer2;
	public static String discretizer3;
	public static String discretizer4;
	public static String discretizer5;
	public static String discretizer6;
	public static String discretizer7;
	public static String discretizer8;
	public static String discretizer9;
	public static String discretizer10;
	public static String CHI_SQUARE;
	public static int maxIntervals;
	public static double probSplit;
	public static double probMerge;
	public static double probReinitialize;
	public static double probReinitializeBegin;
	public static double probReinitializeEnd;
	public static boolean adiKR = false;

	public static String defaultClass;
	public static String initMethod;

	public static int MAX_NODE_INCOMING;
	public static int BOA_INTERVAL;
	public static int PRINT_INTERVAL;
	public static int REDUCE_INTERVAL;

	public static boolean TEMP_doRuleDeletion;
	public static boolean store_flag;

	public static double attributesThrehold;
	public static double attributeThrehold;
	
	
	public static void reset() {
		doRuleDeletion = false;
		doHierarchicalSelection = false;
		adiKR = false;
	}

	public static void forbidRuleDeletion() {
		TEMP_doRuleDeletion = doRuleDeletion;
		doRuleDeletion = false;
		store_flag = true;
	}

	public static void restoreRuleDeletion() {
		doRuleDeletion = TEMP_doRuleDeletion;
		store_flag = false;
	}

	public static double[] chi;

	public static double[] chi_10 = new double[] { 0, 0.0157908 };
	public static double[] chi_20 = new double[] { 0, 0.0641847 };
	public static double[] chi_30 = new double[] { 0, 0.148472 };
	public static double[] chi_40 = new double[] { 0, 0.274996 };
	public static double[] chi_50 = new double[] { 0, 0.454936 };
	public static double[] chi_60 = new double[] { 0, 0.708326 };
	public static double[] chi_70 = new double[] { 0, 1.07419 };

	public static double[] chi_75 = new double[] { 0, 1.32, 2.77, 4.11, 5.39,
			6.63, 7.84, 9.04, 10.22, 11.39, 12.55, 13.7, 14.85, 15.98, 17.12,
			18.25, 19.37, 20.49, 21.6, 22.72, 23.83, 24.93, 26.04, 27.14,
			28.24, 29.34, 30.43, 31.53, 32.62, 33.71, 34.8, 45.62, 56.33,
			66.98, 88.13, 109.1 };

	public static double[] chi_80 = new double[] { 0, 1.64, 3.22, 4.64, 5.99,
			7.29, 8.56, 9.8, 11.03, 12.24, 13.44, 14.63, 15.81, 16.98, 18.15,
			19.31, 20.47, 21.61, 22.76, 23.9, 25.04, 26.17, 27.3, 28.43, 29.55,
			30.68, 31.79, 32.91, 34.03, 35.14, 36.25, 47.27, 58.16, 68.97,
			90.41, 111.7 };

	public static double[] chi_85 = new double[] { 0, 2.07, 3.79, 5.32, 6.74,
			8.12, 9.45, 10.75, 12.03, 13.29, 14.53, 15.77, 16.99, 18.2, 19.41,
			20.6, 21.79, 22.98, 24.16, 25.33, 26.5, 27.66, 28.82, 29.98, 31.13,
			32.28, 33.43, 34.57, 35.71, 36.85, 37.99 };

	public static double[] chi_90 = new double[] { 0, 2.71, 4.61, 6.25, 7.78,
			9.24, 10.64, 12.02, 13.36, 14.68, 15.99, 17.28, 18.55, 19.81,
			21.06, 22.31, 23.54, 24.77, 25.99, 27.2, 28.41, 29.62, 30.81,
			32.01, 33.2, 34.38, 35.56, 36.74, 37.92, 39.09, 40.26 };

	public static double[] chi_95 = new double[] { 0, 3.84, 5.99, 7.82, 9.49,
			11.07, 12.59, 14.07, 15.51, 16.92, 18.31, 19.68, 21.03, 22.36,
			23.69, 25.00, 26.30, 27.59, 28.87, 30.14, 31.41 };

	public static double[] chi_99 = new double[] { 0, 6.64, 9.21, 11.35, 13.28,
			15.09, 16.81, 18.48, 20.09, 21.67, 23.21, 24.73, 26.22, 27.69,
			29.14, 30.58, 32.00, 33.41, 34.81, 36.19, 37.57 };

	public static double[] chi_999 = new double[] { 0, 10.83, 13.82, 16.27,
			18.47, 20.52, 22.46, 24.32, 26.13, 27.88, 29.59, 31.26, 32.91,
			34.53, 36.12, 37.70, 39.25, 40.79, 42.31, 43.82, 45.32 };

	public static double[] chi_9995 = new double[] { 0, 12.12, 15.2, 17.73, 20,
			22.11, 24.1, 26.02, 27.87, 29.67, 31.42, 33.14, 34.82, 36.48,
			38.11, 39.72, 41.31, 42.88, 44.43, 45.97, 47.5 };

	public static double[] chi_9999 = new double[] { 0, 15.14, 18.42, 21.11,
			23.61, 25.74, 27.86, 29.88, 31.83, 33.72, 35.57, 37.37, 39.13,
			40.87, 42.58, 44.26, 45.92, 47.57, 49.19, 50.80, 52.39, 53.96,
			55.52, 57.08, 58.61, 60.14, 61.66, 63.17, 64.66, 66.15, 67.63,
			69.11, 70.57, 72.03, 73.48, 74.93, 76.37, 77.80, 79.22, 80.65,
			82.06, 83.47, 84, 89, 86.28, 87.68, 89.07, 90.46, 91.84, 93.22,
			94.60, 95.97, 97.34, 98.70, 100.06, 101.42, 102.78, 104.13, 105.48,
			106.82, 108.16, 109.50, 110.84, 113.17, 113.51, 114.83, 116.16,
			117.48, 118.81, 120.12, 121.44, 122.76, 124.38, 125.38, 126.68,
			127.99, 129.29, 130.60, 131.89, 133.19, 134.49, 135.78, 137.08,
			138.37, 139.66, 140.94, 142.22, 143.51, 144.79, 146.07, 147.35,
			148.63, 149.90, 151.18, 152.45, 153.72, 154.99, 156.26, 157.53,
			158.79, 160.06, 161.32 };
	
	public static double[] chi_99999 = new double[] { 0, 19.51, 23.03};
	
	public static double[] chi_999999 = new double[] { 0, 23.93, 27.63};
	
	public static double[] chi_9999999 = new double[] { 0,28.37,32.24};

	public static void setChiQuare() {
		if (CHI_SQUARE.matches("10")) {
			chi = chi_10;
		} else if (CHI_SQUARE.matches("20")) {
			chi = chi_20;
		} else if (CHI_SQUARE.matches("30")) {
			chi = chi_30;
		} else if (CHI_SQUARE.matches("40")) {
			chi = chi_40;
		} else if (CHI_SQUARE.matches("50")) {
			chi = chi_50;
		} else if (CHI_SQUARE.matches("60")) {
			chi = chi_60;
		} else if (CHI_SQUARE.matches("70")) {
			chi = chi_70;
		} else if (CHI_SQUARE.matches("75")) {
			chi = chi_75;
		} else if (CHI_SQUARE.matches("80")) {
			chi = chi_80;
		} else if (CHI_SQUARE.matches("85")) {
			chi = chi_85;
		} else if (CHI_SQUARE.matches("90")) {
			chi = chi_90;
		} else if (CHI_SQUARE.matches("95")) {
			chi = chi_95;
		} else if (CHI_SQUARE.matches("99")) {
			chi = chi_99;
		} else if (CHI_SQUARE.matches("999")) {
			chi = chi_999;
		} else if (CHI_SQUARE.matches("9995")) {
			chi = chi_9995;
		} else if (CHI_SQUARE.matches("9999")) {
			chi = chi_9999;
		} else if (CHI_SQUARE.matches("999999")) {
			chi = chi_999999;
		} else if (CHI_SQUARE.matches("9999999")) {
			chi = chi_9999999;
		} else {
			chi = chi_9999;
		}
	}
}
