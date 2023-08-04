package GAssist;


public class Globals_MDL {
	static double theoryWeight;
	static boolean activated = false;
	static boolean fixedWeight = false;

	public static boolean newIteration(int iteration, Classifier[]pop) {
		if (!Parameters.useMDL)
			return false;

		Classifier ind = PopulationWrapper.getBest(pop);

		boolean updateWeight = false;
		if (iteration == Parameters.iterationMDL) {
			LogManager.println("Iteration " + iteration +
					   " :MDL fitness activated");
			activated = true;
			double error = ind.getExceptionsLength();
			double theoryLength = ind.getTheoryLength();
			 theoryLength *= Parameters.numClasses;
			 theoryLength /= ind.getNumAliveRules();

			 theoryWeight =
			    (Parameters.initialTheoryLengthRatio /
			     (1.0 - Parameters.initialTheoryLengthRatio))
			    * (error / theoryLength);
			 updateWeight = true;
		}

		if (activated && !fixedWeight &&
		    Statistics.last10IterationsAccuracyAverage == 1.0) {
			fixedWeight = true;
		}

		if (activated && !fixedWeight) {
			if (ind.getAccuracy() != 1.0) {
				if (Statistics.getIterationsSinceBest() ==
				    10) {
					theoryWeight *=
					    Parameters.weightRelaxFactor;
					updateWeight = true;
				}
			}
		}

		if (updateWeight) {
			Statistics.resetBestStats();
			return true;
		}

		return false;
	}

	public static double mdlFitness(Classifier ind) {
		double fit = 0;
		ind.computeTheoryLength();
		if (activated) {
			fit = ind.getTheoryLength() * theoryWeight;
		}
		double exceptionsLength =
		    105.00 - PerformanceAgent.getAccuracy() * 100.0;
		ind.setExceptionsLength(exceptionsLength);
		fit += exceptionsLength;
		return fit;
	}
}
