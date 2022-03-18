package MAFS;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.StringTokenizer;

import GAssist.Chronometer;
import GAssist.Classifier;
import GAssist.Globals_DefaultC;
import GAssist.Parameters;
import GAssist.Rand;

public class MA {
	/* Own parameters of the algorithm */
//	private long seed;
//	private int nFeature;
//	private double crossProb;
//	private double mutationProb;
//	private int popSize;
//	private int evaluations;
//	private int maxEvaluations;
//	private double beta;
//	private int k;
//	private boolean elitism;
//	private Chromosome ace;
//
//	private Chromosome population[];
//	private Chromosome newPop[];
//
//	public static Chromosome bestInd;

//	/**
//	 * Builder.
//	 * 
//	 * @param script
//	 *            Configuration script
//	 */
//	public MA(String script) {
//
//		evaluations = 0;
//
//		// Inicialization of auxiliar structures
//
//		Chromosome.setK(k);
//		Chromosome.setMutationProb(mutationProb);
//		Chromosome.setBeta(beta);
//
//		population = new Chromosome[popSize];
//		newPop = new Chromosome[popSize];
//		for (int i = 0; i < popSize; i++) {
//			population[i] = new Chromosome(nFeature);
//			population[i].evaluate();
//		}
//
//		Arrays.sort(population);
//
//		if (elitism == true) {
//			ace = new Chromosome(population[0].getGenes(), population[0]
//					.getFitness());
//		}
//
//		// Initialization of random generator
//
//		// Initialization stuff ends here. So, we can start time-counting
//
//	}

//	/**
//	 * Executes the GGA
//	 */
//	public void execute() {
//
//		int candidate1, candidate2;
//		int selected1, selected2;
//
//		while (evaluations < maxEvaluations) {
//
//			for (int i = 0; i < popSize; i += 2) {
//
//				// Binary tournament selection: First candidate
//
//				candidate1 = Rand.getInteger(0, popSize - 1);
//				do {
//					candidate2 = Rand.getInteger(0, popSize - 1);
//				} while (candidate2 == candidate1);
//
//				if (population[candidate1].getFitness() > population[candidate2]
//						.getFitness()) {
//					selected1 = candidate1;
//				} else {
//					selected1 = candidate2;
//				}
//
//				// Binary tournament selection: Second candidate
//
//				candidate1 = Rand.getInteger(0, popSize - 1);
//				do {
//					candidate2 = Rand.getInteger(0, popSize - 1);
//				} while (candidate2 == candidate1);
//
//				if (population[candidate1].getFitness() > population[candidate2]
//						.getFitness()) {
//					selected2 = candidate1;
//				} else {
//					selected2 = candidate2;
//				}
//
//				// Cross operator
//
//				if (Rand.getReal() < crossProb) {
//					newPop[i] = new Chromosome(population[selected1].getGenes());
//					newPop[i + 1] = new Chromosome(newPop[i]
//							.crossPMX(population[selected2].getGenes()));
//				} else { // there is not cross
//					newPop[i] = new Chromosome(
//							population[selected1].getGenes(),
//							population[selected1].getFitness());
//					newPop[i + 1] = new Chromosome(population[selected2]
//							.getGenes(), population[selected2].getFitness());
//				}
//
//				// Mutation operator
//				newPop[i].mutation();
//				newPop[i + 1].mutation();
//
//				/* Evaluation of the offspring */
//				if (newPop[i].getValid() == false) {
//					newPop[i].evaluate();
//					evaluations++;
//				}
//				if (newPop[i + 1].getValid() == false) {
//					newPop[i + 1].evaluate();
//					evaluations++;
//				}
//			}
//
//			/* Replace the population */
//
//			System.arraycopy(newPop, 0, population, 0, popSize);
//			Arrays.sort(population);
//
//			if (elitism == true) {
//				if (population[0].getFitness() > ace.getFitness()) {
//					ace = new Chromosome(population[0].getGenes(),
//							population[0].getFitness());
//				} else {
//					population[popSize - 1] = new Chromosome(ace.getGenes(),
//							ace.getFitness());
//				}
//			}
//		}
//
//		Arrays.sort(population);
//
//		// Features selected
//		int featSelected[] = new int[nFeature];
//
//		featSelected = population[0].getGenes();
//
//		// System.out.println(name+" "+ relation + " Train " +
//		// (double)(System.currentTimeMillis()-initialTime)/1000.0 + "s");
//		// OutputFS.writeTrainOutput(outFile[0], trainReal, trainNominal,
//		// trainNulls, trainOutput,featSelected, inputs, output, inputAtt,
//		// relation);
//		// OutputFS.writeTestOutput(outFile[1], test, featSelected, inputs,
//		// output, inputAtt, relation);
//		// OutputIS.escribeSalidaAux(outFile[1]+".txt",((double)(System.currentTimeMillis()-initialTime)/1000.0),1.0-((double)population[0].getNGenes()/(double)trainData[0].length),relation);
//
//	}

	public static Chromosome[] doTournamentSelection(Chromosome[] population) {
		// Chronometer.startChronSelection();

		Chromosome[] selectedPopulation;
		selectedPopulation = new Chromosome[FSParameters.popSize];
		int candidate1, candidate2;
		int selected1, selected2;

		for (int i = 0; i < FSParameters.popSize; i += 2) {

			// Binary tournament selection: First candidate

			candidate1 = Rand.getInteger(0, FSParameters.popSize - 1);
			do {
				candidate2 = Rand.getInteger(0, FSParameters.popSize - 1);
			} while (candidate2 == candidate1);

			if (population[candidate1].getFitness() > population[candidate2]
					.getFitness()) {
				selected1 = candidate1;
			} else {
				selected1 = candidate2;
			}

			// Binary tournament selection: Second candidate

			candidate1 = Rand.getInteger(0, FSParameters.popSize - 1);
			do {
				candidate2 = Rand.getInteger(0, FSParameters.popSize - 1);
			} while (candidate2 == candidate1);

			if (population[candidate1].getFitness() > population[candidate2]
					.getFitness()) {
				selected2 = candidate1;
			} else {
				selected2 = candidate2;
			}

			selectedPopulation[i] = new Chromosome(population[selected1]
					.getGenes());
			selectedPopulation[i + 1] = new Chromosome(population[selected2]
					.getGenes());
		}

		// Chronometer.stopChronSelection();
		return selectedPopulation;
	}

	public static Chromosome[] doCrossover(Chromosome[] population) {
		Chromosome[] newPop = new Chromosome[FSParameters.popSize];
		for (int i = 0; i < FSParameters.popSize; i += 2) {
			if (Rand.getReal() < FSParameters.probCrossover) {
				newPop[i] = new Chromosome(population[i].getGenes());
				newPop[i + 1] = new Chromosome(newPop[i]
						.crossPMX(population[i + 1].getGenes()));
			} else { // there is not cross
				newPop[i] = new Chromosome(population[i].getGenes(),
						population[i].getFitness());
				newPop[i + 1] = new Chromosome(population[i + 1].getGenes(),
						population[i + 1].getFitness());
			}
		}

		return newPop;
	}

	public static void doMutation(Chromosome[] pop) {
		for (int i = 0; i < FSParameters.popSize; i++) {
			pop[i].mutation();
		}
	}
	
	public static Chromosome[] replacementPolicy(Chromosome[] population, Chromosome[] newPop){
		System.arraycopy(newPop, 0, population, 0, FSParameters.popSize);
//		Arrays.sort(population);
		return population;
	}

}// end-class
