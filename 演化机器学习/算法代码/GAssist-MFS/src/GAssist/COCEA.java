package GAssist;

import java.util.*;
import java.lang.*;
import MAFS.*;

/*
 * GA.java
 *
 */

public class COCEA {

	public Classifier[] population;
	public Classifier[] bestIndiv;
	public Classifier bestCLS;
	public int numVersions;

	// public int nFeature;

	public Chromosome[] FSpopulation;
	public Chromosome bestFSS;

	/** Creates a new instance of GA */
	public COCEA() {
	}

	/**
	 * Prepares GA for a new run.
	 */
	public void initCOCEA() {
		// Init CLSpopulation
		population = new Classifier[Parameters.popSize];
		PopulationWrapper.initInstancesEvaluation();

		FSParameters.numAttributes = Parameters.numAttributes;

		numVersions = PopulationWrapper.numVersions();
		bestIndiv = new Classifier[numVersions];

		Factory.initialize();
		initPopulation(population);
		Statistics.initStatistics();

		PopulationWrapper.doEvaluation(population);
		// Arrays.sort(population);
		bestCLS = PopulationWrapper.getBest(population).copy();

		// Init FSSpopulation
		FSpopulation = new Chromosome[FSParameters.popSize];

		for (int i = 0; i < FSParameters.popSize; i++) {
			FSpopulation[i] = new Chromosome(FSParameters.numAttributes);
			System.out.println(FSpopulation[i].toString());
			// FSpopulation[i].evaluate();
			double acc = PopulationWrapper.evaluateClassifierFS(bestCLS,
					FSpopulation[i]);
			FSpopulation[i].calFitness(acc);
		}

		Arrays.sort(FSpopulation);
		bestFSS = new Chromosome(FSpopulation[0].getGenes());

	}

	/**
	 * Inits a new population.
	 * 
	 * @param _population
	 *            Population to init.
	 */
	private void initPopulation(Classifier[] _population) {
		for (int i = 0; i < Parameters.popSize; i++) {
			_population[i] = Factory.newClassifier();
			_population[i].initRandomClassifier();
		}
	}

	public void checkBestIndividual() {
		Classifier best = PopulationWrapper.getBest(population);
		int currVer = PopulationWrapper.getCurrentVersion();

		if (bestIndiv[currVer] == null) {
			bestIndiv[currVer] = best.copy();
		} else {
			if (best.compareToIndividual(bestIndiv[currVer])) {
				bestIndiv[currVer] = best.copy();
			}
		}
	}

	/**
	 * Executes a number of iterations of GA.
	 * 
	 * @param _numIterations
	 *            Number of iterations to do.
	 */
	public void run() {
		Classifier[] offsprings;
		
		Chromosome[] newPop;

		PopulationWrapper.doEvaluation(population);

		int numIterations = Parameters.numIterations;

		for (int iteration = 0; iteration < numIterations; iteration++) {
			boolean lastIteration = (iteration == numIterations - 1);
			Parameters.percentageOfLearning = (double) iteration
					/ (double) numIterations;
			boolean res1 = PopulationWrapper.initIteration();
			boolean res2 = Timers.runTimers(iteration, population);
			if (res1 || res2) {
				PopulationWrapper.setModified(population);
			}

			// GA cycle
			population = doTournamentSelection(population);
			offsprings = doCrossover(population);
			doMutation(offsprings);
			PopulationWrapper.doEvaluation(offsprings);
			population = replacementPolicy(offsprings, lastIteration);

			// FS cycle
			FSpopulation = MA.doTournamentSelection(FSpopulation);
			newPop = MA.doCrossover(FSpopulation);
			MA.doMutation(newPop);
			PopulationWrapper.doEvaluationFSS(newPop, bestCLS);
			FSpopulation = MA.replacementPolicy(FSpopulation, newPop);
			Arrays.sort(FSpopulation);

//			Arrays.sort(population);
			Classifier tempBestCLS = PopulationWrapper.getBest(population).copy();			
			Chromosome tempBestFSS = new Chromosome(FSpopulation[0].getGenes());
			
			double acc1 = PopulationWrapper.evaluateClassifierFS(bestCLS, bestFSS); 
			double acc2 = PopulationWrapper.evaluateClassifierFS(tempBestCLS, tempBestFSS);
			double acc3 = PopulationWrapper.evaluateClassifierFS(bestCLS, tempBestFSS);
			double acc4 = PopulationWrapper.evaluateClassifierFS(tempBestCLS, bestFSS);
			if(acc1 < acc2){
				if(acc2 < acc3){
					if(acc3 < acc4){
						bestCLS = tempBestCLS.copy(); 
					}
					else{
						bestFSS = new Chromosome(tempBestFSS.getGenes());
					}
				}else{
					if(acc2 > acc4){
						bestCLS = tempBestCLS.copy(); 
						bestFSS = new Chromosome(tempBestFSS.getGenes());
					}else{
						bestCLS = tempBestCLS.copy(); 
					}
				}
			}else{
				if(acc1 < acc3){
					if(acc3 < acc4){
						bestCLS = tempBestCLS.copy(); 
					}else{
						bestFSS = new Chromosome(tempBestFSS.getGenes());
					}
				}else{
					if(acc1 > acc4){
						
					}else{
						bestCLS = tempBestCLS.copy(); 
					}
				}
			}
			

			Statistics.computeStatistics(population);
			Timers.runOutputTimers(iteration, population);
		}

		Statistics.statisticsToFile();
		
//		Classifier best = PopulationWrapper.getBest(population).copy();
//		Arrays.sort(FSpopulation);
//		bestFSS = new Chromosome(FSpopulation[0].getGenes());
		
		LogManager.println("\nPhenotype: ");
		bestCLS.printClassifier();
		LogManager.println(bestFSS.toString());
		LogManager.println_file(bestFSS.toString());

		PopulationWrapper
				.testClassifierFS(bestCLS, bestFSS, "training", Parameters.trainFile);
		PopulationWrapper.testClassifierFS(bestCLS, bestFSS, "test", Parameters.testFile);
	}

	Classifier[] doCrossover(Classifier[] _population) {
		Chronometer.startChronCrossover();

		int i, j, k, countCross = 0;
		int numNiches = _population[0].getNumNiches();
		ArrayList[] parents = new ArrayList[numNiches];
		Classifier parent1, parent2;
		Classifier[] offsprings = new Classifier[2];
		Classifier[] offspringPopulation = new Classifier[Parameters.popSize];

		for (i = 0; i < numNiches; i++) {
			parents[i] = new ArrayList();
			parents[i].ensureCapacity(Parameters.popSize);
		}

		for (i = 0; i < Parameters.popSize; i++) {
			int niche = _population[i].getNiche();
			parents[niche].add(new Integer(i));
		}

		for (i = 0; i < numNiches; i++) {
			int size = parents[i].size();
			Sampling samp = new Sampling(size);
			int p1 = -1;
			for (j = 0; j < size; j++) {
				if (Rand.getReal() < Parameters.probCrossover) {
					if (p1 == -1) {
						p1 = samp.getSample();
					} else {
						int p2 = samp.getSample();
						int pos1 = ((Integer) parents[i].get(p1)).intValue();
						int pos2 = ((Integer) parents[i].get(p2)).intValue();
						parent1 = _population[pos1];
						parent2 = _population[pos2];

						offsprings = parent1.crossoverClassifiers(parent2);
						offspringPopulation[countCross++] = offsprings[0];
						offspringPopulation[countCross++] = offsprings[1];
						p1 = -1;
					}
				} else {
					int pos = ((Integer) parents[i].get(samp.getSample()))
							.intValue();
					offspringPopulation[countCross++] = _population[pos].copy();
				}
			}
			if (p1 != -1) {
				int pos = ((Integer) parents[i].get(p1)).intValue();
				offspringPopulation[countCross++] = _population[pos].copy();
			}
		}

		Chronometer.stopChronCrossover();
		return offspringPopulation;
	}

	private int selectNicheWOR(int[] quotas) {
		int num = quotas.length;
		if (num == 1)
			return 0;

		int total = 0, i;
		for (i = 0; i < num; i++)
			total += quotas[i];
		if (total == 0) {
			return Rand.getInteger(0, num - 1);
		}
		int pos = Rand.getInteger(0, total - 1);
		total = 0;
		for (i = 0; i < num; i++) {
			total += quotas[i];
			if (pos < total) {
				quotas[i]--;
				return i;
			}
		}

		LogManager.printErr("We should not be here");
		System.exit(1);
		return -1;
	}

	private void initPool(ArrayList pool, int whichNiche,
			Classifier[] _population) {
		if (Globals_DefaultC.nichingEnabled) {
			for (int i = 0; i < Parameters.popSize; i++) {
				if (_population[i].getNiche() == whichNiche) {
					pool.add(new Integer(i));
				}
			}
		} else {
			for (int i = 0; i < Parameters.popSize; i++) {
				pool.add(new Integer(i));
			}
		}
	}

	private int selectCandidateWOR(ArrayList pool, int whichNiche,
			Classifier[] _population) {
		if (pool.size() == 0) {
			initPool(pool, whichNiche, population);
			if (pool.size() == 0) {
				return Rand.getInteger(0, Parameters.popSize - 1);
			}
		}

		int pos = Rand.getInteger(0, pool.size() - 1);
		int elem = ((Integer) pool.get(pos)).intValue();
		pool.remove(pos);
		return elem;
	}

	/**
	 * Does Tournament Selection without replacement.
	 */
	public Classifier[] doTournamentSelection(Classifier[] _population) {
		Chronometer.startChronSelection();

		Classifier[] selectedPopulation;
		selectedPopulation = new Classifier[Parameters.popSize];
		int i, j, winner, candidate;
		int numNiches;
		if (Globals_DefaultC.nichingEnabled) {
			numNiches = _population[0].getNumNiches();
		} else {
			numNiches = 1;
		}

		ArrayList[] pools = new ArrayList[numNiches];
		for (i = 0; i < numNiches; i++)
			pools[i] = new ArrayList();
		int[] nicheCounters = new int[numNiches];
		int nicheQuota = Parameters.popSize / numNiches;
		for (i = 0; i < numNiches; i++)
			nicheCounters[i] = nicheQuota;

		for (i = 0; i < Parameters.popSize; i++) {
			// There can be only one
			int niche = selectNicheWOR(nicheCounters);
			winner = selectCandidateWOR(pools[niche], niche, _population);
			for (j = 1; j < Parameters.tournamentSize; j++) {
				candidate = selectCandidateWOR(pools[niche], niche, _population);
				if (_population[candidate]
						.compareToIndividual(_population[winner])) {
					winner = candidate;
				}
			}
			selectedPopulation[i] = _population[winner].copy();
		}
		Chronometer.stopChronSelection();
		return selectedPopulation;
	}

	public void doMutation(Classifier[] _population) {
		Chronometer.startChronMutation();
		int popSize = Parameters.popSize;
		double probMut = Parameters.probMutationInd;

		for (int i = 0; i < Parameters.popSize; i++) {
			if (Rand.getReal() < probMut) {
				_population[i].doMutation();
			}
		}

		doSpecialStages(_population);

		Chronometer.stopChronMutation();
	}

	void sortedInsert(ArrayList set, Classifier cl) {
		for (int i = 0, max = set.size(); i < max; i++) {
			if (cl.compareToIndividual((Classifier) set.get(i))) {
				set.add(i, cl);
				return;
			}
		}
		set.add(cl);
	}

	public Classifier[] replacementPolicy(Classifier[] offspring,
			boolean lastIteration) {
		int i;

		Chronometer.startChronReplacement();

		if (lastIteration) {
			for (i = 0; i < numVersions; i++) {
				if (bestIndiv[i] != null) {
					PopulationWrapper.evaluateClassifier(bestIndiv[i]);
				}
			}
			ArrayList set = new ArrayList();
			for (i = 0; i < Parameters.popSize; i++) {
				sortedInsert(set, offspring[i]);
			}
			for (i = 0; i < numVersions; i++) {
				if (bestIndiv[i] != null) {
					sortedInsert(set, bestIndiv[i]);
				}
			}

			for (i = 0; i < Parameters.popSize; i++) {
				offspring[i] = (Classifier) set.get(i);
			}
		} else {
			boolean previousVerUsed = false;
			int currVer = PopulationWrapper.getCurrentVersion();
			if (bestIndiv[currVer] == null && currVer > 0) {
				previousVerUsed = true;
				currVer--;
			}

			if (bestIndiv[currVer] != null) {
				PopulationWrapper.evaluateClassifier(bestIndiv[currVer]);
				int worst = PopulationWrapper.getWorst(offspring);
				offspring[worst] = bestIndiv[currVer].copy();
			}
			if (!previousVerUsed) {
				int prevVer;
				if (currVer == 0) {
					prevVer = numVersions - 1;
				} else {
					prevVer = currVer - 1;
				}
				if (bestIndiv[prevVer] != null) {
					PopulationWrapper.evaluateClassifier(bestIndiv[prevVer]);
					int worst = PopulationWrapper.getWorst(offspring);
					offspring[worst] = bestIndiv[prevVer].copy();
				}
			}
		}

		Chronometer.stopChronReplacement();
		return offspring;
	}

	public void doSpecialStages(Classifier[] population) {
		int numStages = population[0].numSpecialStages();

		for (int i = 0; i < numStages; i++) {
			for (int j = 0; j < population.length; j++) {
				population[j].doSpecialStage(i);
			}
		}
	}
}
