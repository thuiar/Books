package GAssist;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import GAssist.Dataset.Attributes;
import boa.BOA;
import boa.Chromosome;
import boa.Population;
import boa.util.MyUtil;

/*
 * GA.java
 *
 */

public class GA {
	
	Classifier[] population;
	Classifier[] bestIndiv;
	int numVersions;
	BOA boa;
	int premature=0;
	
	/** Creates a new instance of GA */
	public GA() {
	}
	
	/**
	 *  Prepares GA for a new run.
	 */
	public void initGA() {
		//Init population
		population=new Classifier[Parameters.popSize];
		

		numVersions=PopulationWrapper.numVersions();
		bestIndiv = new Classifier[numVersions];
			
		Factory.initialize();
		initPopulation(population);
		Statistics.initStatistics();
		
		
		boa = new BOA();
		premature = Parameters.iterationRuleDeletion ;
	}
	
	
	public void restartGA(int iter) {
		//Init population
		
		
		population=new Classifier[Parameters.popSize];
		numVersions=PopulationWrapper.numVersions();
		bestIndiv = new Classifier[numVersions];
		Factory.initialize();
		initPopulation(population);
		boa = new BOA();
		Parameters.iterationRuleDeletion = iter + premature;

		
		
	}
	
	public void halfRestartGA(int iter){
		List<Classifier> classifierList = new ArrayList<Classifier>();
		for (int i = 0; i < population.length; i++) {
			classifierList.add(population[i]);
		}
		Collections.sort(classifierList);
		population=new Classifier[Parameters.popSize];
		int index = 0;
		for (int  i = classifierList.size() - 1; index < Parameters.popSize/2; index++, i--) {
			population[index] = classifierList.get(i);
		}
		Factory.initialize();
		for (; index<Parameters.popSize; index++) {
				population[index]=Factory.newClassifier();
				population[index].initRandomClassifier();
			}
		boa = new BOA();
		Parameters.iterationRuleDeletion += iter;
	}
	
	/**
	 *  Inits a new population.
	 *  @param _population Population to init.
	 */
	private void initPopulation(Classifier[] _population) {
		for (int i=0; i<Parameters.popSize; i++) {
		   _population[i]=Factory.newClassifier();
		   _population[i].initRandomClassifier();
		}
	}

	public void checkBestIndividual() {
		Classifier best=PopulationWrapper.getBest(population);
		int currVer=PopulationWrapper.getCurrentVersion();

		if(bestIndiv[currVer] == null) {
			bestIndiv[currVer] = best.copy();
		} else {
			if(best.compareToIndividual(bestIndiv[currVer])) {
				bestIndiv[currVer] = best.copy();
			}
		}
	}
	
	public Population initRulePopulation() {
		Population rulePopulation = new Population();
		for (int i = 0; i < population.length; i++) {
			ClassifierGABIL c = (ClassifierGABIL) population[i];
			int ruleSize = Globals_GABIL.ruleSize;
			for (int j = 0; j < c.numRules; j++) {
				rulePopulation.addIndividual(new Chromosome(c.crm, j* ruleSize, ruleSize, c.getRuleAccuracies(j)));
			}
		}
		rulePopulation.sort();
		return rulePopulation;
	}
	
	
		
	
   /**
	 *  Executes a number of iterations of GA.
	 *  @param _numIterations Number of iterations to do.
	 */
	public void run(boolean isBOA) {
		
		boolean deleted = false;
		boolean justDeleted = false;
		int deleteItera = -1;
		double beforeDeleteAve = -1, beforeDeleteBest = -1;
		
		Classifier.clearNumMatch();
		Classifier[] offsprings=null;
		Population rulePopulation = null, ruleParent=null;
		
		
		
		int numIterations=Parameters.numIterations;
		
		rulePopulation =  initRulePopulation();
		ruleParent = new Population();
		PopulationWrapper.doEvaluation(population);
		
		
//		for (int i = 0; i < population.length; i++) {
//			ClassifierGABIL c = (ClassifierGABIL) population[i];
//			int ruleSize = Globals_GABIL.ruleSize;
//		
//			for (int j = 0; j < c.numRules; j++) {
//				rulePopulation.addIndividual(new Chromosome(c.crm, j* ruleSize, ruleSize, c.getRuleAccuracies(j)));
//			}
//		}
		rulePopulation.sort();
		
		
		double bestAcc = 0;
		for (int iteration=0; iteration<numIterations && bestAcc<1 ; iteration++) {
			boolean lastIteration=(iteration==numIterations-1);
			Parameters.percentageOfLearning = (double)iteration
				/(double)numIterations;
			boolean res1=PopulationWrapper.initIteration();
			boolean res2=Timers.runTimers(iteration,population);
			if(res1 || res2) {
				PopulationWrapper.setModified(population);
			}
		
			// GA cycle	
			//PopulationWrapper.doEvaluation(population);
			
			/***************** BOA for offsprings ************************************/
			if (isBOA&&iteration%Parameters.BOA_INTERVAL==0||justDeleted) {
				
				justDeleted = false;
				
				
				LogManager.print("BOA iteration : ");
				double aveAccuracy = 0;
				
				
				
				aveAccuracy /= rulePopulation.getSize();
				ruleParent=rulePopulation.truncationSelection();
				//ruleParent=rulePopulation.wheelSelection();
				
				
				
				//boa.constructTheNetwork(ruleParent);
				boa.constructTheNetwork_MDL(ruleParent);
				
				int rn=0;
				for (int i = 0; i < population.length; i++) {
					ClassifierGABIL c = (ClassifierGABIL) population[i];
					rn+=c.numRules;
				}
				
				
//				int offspringSize = rn;
				int offspringSize = ruleParent.getSize();
				if(offspringSize*2<rn){
					offspringSize =  rn-ruleParent.getSize();
					System.out.println("-------"+offspringSize);
				}
//				int offspringSize = rn-ruleParent.getSize();
//				int offspringSize = ruleParent.getSize();
				
				
				
				
				Population ruleOffspring = boa.generateOffsprings(offspringSize);
//				ruleOffspring.check(group);
				ruleParent.append(ruleOffspring);
				
				
				
				List<Chromosome> chrosPool = ruleParent.getChromosomes();
			
				
				Chromosome[] allCS = chrosPool.toArray(new Chromosome[chrosPool.size()]);
				ClassifierGABIL temp[] = new ClassifierGABIL[allCS.length];
				for(int i=0;i<temp.length;i++){
					temp[i] =  new ClassifierGABIL(new Chromosome[]{allCS[i]});
				}
				PopulationWrapper.doEvaluation(temp);
				
				int ruleSize = Globals_GABIL.ruleSize;
				Population rulePool =  new Population();;
				for(int i=0;i<temp.length ;i++){
					for (int j = 0; j < temp[i].numRules; j++) {
						rulePool.addIndividual(new Chromosome(temp[i].crm, j* ruleSize, ruleSize, temp[i].getRuleAccuracies(j)));
					}
				}
				rulePool.sort();
				
				List<Chromosome> chros =new ArrayList<Chromosome>();
				System.out.println(rulePool.getChromosomes().size()+" "+ ruleParent.getChromosomes().size()+"  "+rn);
				for(int i = 0; i<rn ; i++){
					chros.add(rulePool.getChromosomes().get(rulePool.getChromosomes().size()-i-1));
				}
				
				while(rulePool.getChromosomes().size()>rn){
					rulePool.getChromosomes().remove(0);
				}
				System.out.println(rulePool.getChromosomes().size()+" "+ ruleParent.getChromosomes().size()+"  "+rn);
				
				
				offsprings = new Classifier[Parameters.popSize];
				int index = 0;
				for (int i = 0; i < population.length; i++) {
					Chromosome[] cs = new Chromosome[population[i].numRules];
					for (int j = 0; j < cs.length; j++, index++) {
						cs[j] = chros.get(index);
					}
					offsprings[i] = new ClassifierGABIL(cs);
				}
				//rulePopulation = ruleParent;
				rulePopulation = rulePool;
				
				LogManager.println("Parent Rule Ave Fit: "+ruleParent.getAveFitness()+" ");
				LogManager.println("Pool Rule Ave Fit: "+rulePool.getAveFitness()+" ");
			
				
				PopulationWrapper.doEvaluation(offsprings);
				population = replacementPolicy(population, offsprings, lastIteration);
			
				
				
				
			}/**************************************************************/
			else{
//				if(isBOA&&iteration%Parameters.BOA_INTERVAL>5){
//					Parameters.forbidRuleDeletion();
//					
//				}
				population=doTournamentSelection(population);
				offsprings=doCrossover(population);
				doMutation(offsprings);
				PopulationWrapper.doEvaluation(offsprings);
				population=replacementPolicy(offsprings,lastIteration);
				
//				if(isBOA&&iteration%Parameters.BOA_INTERVAL==1&&Parameters.store_flag){
//					Parameters.restoreRuleDeletion();
//				}
			}
			
		
			Statistics.computeStatistics(population);
				
		
			if(deleted){
//				if(Statistics.bestAccuracy[iteration]>=beforeDeleteBest){
//					System.out.println("Best\t"+Statistics.bestAccuracy[iteration]+"\t"+beforeDeleteBest);
//					deleted = false;
//				}else if(Statistics.averageAccuracy[iteration]>=beforeDeleteAve){
//					System.out.println("Ave\t"+Statistics.averageAccuracy[iteration]+"\t"+beforeDeleteAve);
//					deleted = false;
//				}else if(Statistics.bestAccuracy[iteration]==Statistics.bestAccuracy[iteration-1]){
//					deleted = false;
//				}
			

				if(Statistics.bestAccuracy[iteration]>=Statistics.bestAccuracy[iteration-2]){
					deleted = false;
				}

				
				
			}
			
			if(iteration%Parameters.REDUCE_INTERVAL==Parameters.REDUCE_INTERVAL-1&&!deleted){
				
				
				Statistics.computeRedundancy(population);
//				rulePopulation = new Population();
//				for (int i = 0; i < population.length; i++) {
//					ClassifierGABIL c = (ClassifierGABIL) population[i];
//					int ruleSize = Globals_GABIL.ruleSize;
//
//					for (int j = 0; j < c.numRules; j++) {
//						rulePopulation.addIndividual(new Chromosome(c.crm,
//								j * ruleSize, ruleSize, c.getRuleAccuracies(j)));
//					}
//				}
				
//				Bayesian bayesian = new Bayesian(rulePopulation);
//				bayesian.constructTheNetwork();
//				List<ArrayList<Integer>> array = bayesian.printPC();
//				deletedAttr = rulePopulation.computeMI();
				if(beforeDeleteAve < Statistics.averageAccuracy[iteration]){
					beforeDeleteAve = Statistics.averageAccuracy[iteration];
				}
				if(beforeDeleteBest < Statistics.bestAccuracy[iteration]){
					beforeDeleteBest = Statistics.bestAccuracy[iteration];
				}
			/*************************************/
				deleted=deleteAttribute();
				if(deleted){
					justDeleted = true;
					deleteItera = iteration;					
					//halfRestartGA(iteration);
					restartGA(iteration);
					PopulationWrapper.doEvaluation(population);
					rulePopulation =  initRulePopulation();					//initGA();
				}
			/*************************************/
				
//				deleteAttribute(array);
			}	
				
		}


		Statistics.statisticsToFile();
		Classifier best=PopulationWrapper.getBest(population);

		LogManager.println("\nPhenotype: ");
		best.printClassifier();
		PopulationWrapper.testClassifier(best,"training",Parameters.trainFile);	   
		PopulationWrapper.testClassifier(best,"test",Parameters.testFile);	   
	}
	
	
	public boolean deleteAttribute() {
		
		
		int delete = Statistics.uselessAttr();
		if(delete ==-1) return false;
		
		
		/**********************/
//		System.out.println("Before Delete:"+Attributes.getNumAttributes());
//		deleteAttribute(new int[]{delete});
//		System.out.println("After Delete:"+Attributes.getNumAttributes());
//		return true;
		/**********************/
		
//		int[] delete = Statistics.uselessAttrs();
//		if(delete.length == 0) return false;
//		int r = (int)(Math.random() * delete.length);  
//		deleteAttribute(new int[]{delete[r]});
//		System.out.println("Cand  Delete:"+delete.length);
		
		

		
		int[] deletes = Statistics.uselessAttrs();
		if(deletes.length!= 0){
			deleteAttribute(deletes);
			System.out.println("Before Delete:"+Attributes.getNumAttributes());
			System.out.println("After Delete:"+Attributes.getNumAttributes());
			return true;
		}
		return false;
		
	}
	
	public void deleteAttribute(List<ArrayList<Integer>> array) {
		System.out.println("Before Delete:"+Attributes.getNumAttributes());
		int[] deletesCand = Statistics.uselessAttrs();
		List<Integer> delete = new ArrayList<Integer>();
		for(int i=0;i<deletesCand.length;i++){
			System.out.print(deletesCand[i]+" ");
			delete.add(deletesCand[i]);
		}
		System.out.println();
		for(int i=0;i<delete.size();i++){
			int temp = delete.get(i);
			List<Integer> PC = array.get(temp);
			for(int j=i+1;j<delete.size();j++){
				if(PC.contains(delete.get(j))){
					delete.remove(j);
				}
			}
		}

		int[] realDelete = new int[delete.size()];
		for(int i=0;i<delete.size();i++){
			realDelete[i]=delete.get(i);
		}
		if(realDelete.length!= 0){
			deleteAttribute(realDelete);

		}
		System.out.println("After Delete:"+Attributes.getNumAttributes());
	}
	
	

	public void deleteAttribute(int[] attrs) {
		
		System.out.print("Delete:\t");
		LogManager.print_file("Delete:\t");
		for(int i=0;i<attrs.length;i++){
			System.out.print(attrs[i]+", ");
			LogManager.print_file(attrs[i]+", ");
		}
		System.out.println();
		LogManager.println_file("");
		
		for (int i = 0; i < population.length; i++) {
			((ClassifierGABIL) population[i]).deleteAttribute(attrs);
		}
		MyUtil.deleteAttribute(Parameters.trainFile, "delete-train.arff",attrs);
		MyUtil.deleteAttribute(Parameters.testFile, "delete-test.arff", attrs);

		Parameters.trainFile = "delete-train.arff";
		Parameters.testFile = "delete-test.arff";
		PopulationWrapper.initInstancesEvaluation();
		Factory.initialize();
		Statistics.clearStatistics();

	}


	Classifier[] doCrossover(Classifier[] _population) {
		Chronometer.startChronCrossover();

		int i,j,k,countCross=0;
		int numNiches=_population[0].getNumNiches();
		ArrayList[] parents=new ArrayList[numNiches];
		Classifier parent1, parent2;
		Classifier[] offsprings=new Classifier[2];
		Classifier[] offspringPopulation=new Classifier[Parameters.popSize];

		for(i=0;i<numNiches;i++) {
			parents[i]=new ArrayList();
			parents[i].ensureCapacity(Parameters.popSize);
		}

		for(i=0;i<Parameters.popSize;i++) {
			int niche=_population[i].getNiche();
			parents[niche].add(new Integer(i));
		}

		for(i=0;i<numNiches;i++) {
			int size=parents[i].size();
			Sampling samp = new Sampling(size);
			int p1=-1;
			for(j=0;j<size;j++) {
				if (Rand.getReal()<Parameters.probCrossover) {
					if(p1==-1) {
						p1=samp.getSample();
					} else {
						int p2=samp.getSample();
						int pos1=((Integer)parents[i].get(p1)).intValue();
						int pos2=((Integer)parents[i].get(p2)).intValue();
						parent1=_population[pos1];
						parent2=_population[pos2];

						//offsprings=parent1.crossoverClassifiers(parent2);
						offsprings = ((ClassifierGABIL)parent1).crossoverRules(parent2);
						
						
						offspringPopulation[countCross++]=offsprings[0];
						offspringPopulation[countCross++]=offsprings[1];
						p1=-1;
					}
				} else {
					int pos=((Integer)parents[i].get(samp.getSample())).intValue();
					offspringPopulation[countCross++]=_population[pos].copy();
				}
			}
			if(p1!=-1) {
				int pos=((Integer)parents[i].get(p1)).intValue();
				offspringPopulation[countCross++]=_population[pos].copy();
			}
		}
						
		Chronometer.stopChronCrossover();
		return offspringPopulation;
	}

	private int selectNicheWOR(int []quotas) {
		int num=quotas.length;
		if(num==1) return 0;

		int total=0,i;
		for(i=0;i<num;i++) total+=quotas[i];
		if(total==0) {
			return Rand.getInteger(0,num-1);
		}
		int pos=Rand.getInteger(0,total-1);
		total=0;
		for(i=0;i<num;i++) {
			total+=quotas[i];
			if(pos<total) {
				quotas[i]--;
				return i;
			}
		}

                LogManager.printErr("We should not be here");
                System.exit(1);
		return -1;
	}

	private void initPool(ArrayList pool,int whichNiche,Classifier[] _population) {
		if(Globals_DefaultC.nichingEnabled) {
			for(int i=0;i<Parameters.popSize;i++) {
				if(_population[i].getNiche()==whichNiche) {
					pool.add(new Integer(i));
				}
			}
		} else {
			for(int i=0;i<Parameters.popSize;i++) {
				pool.add(new Integer(i));
			}
		}
	}

	private int selectCandidateWOR(ArrayList pool,int whichNiche,Classifier[] _population) {
		if(pool.size()==0) {
			initPool(pool,whichNiche,population);
			if(pool.size()==0) {
				return Rand.getInteger(0,Parameters.popSize-1);
			}
		}

		int pos=Rand.getInteger(0,pool.size()-1);
		int elem=((Integer)pool.get(pos)).intValue();
		pool.remove(pos);
		return elem;
	}

	/**
	 *  Does Tournament Selection without replacement.
	 */
	public Classifier[] doTournamentSelection(Classifier[] _population) {
		Chronometer.startChronSelection();

		Classifier[] selectedPopulation;
		selectedPopulation=new Classifier[Parameters.popSize];
		int i,j,winner,candidate;
		int numNiches;
		if(Globals_DefaultC.nichingEnabled) {
			numNiches=_population[0].getNumNiches();
		} else {
			numNiches=1;
		}

		ArrayList[] pools=new ArrayList[numNiches];
		for(i=0;i<numNiches;i++) pools[i]=new ArrayList();
		int []nicheCounters = new int[numNiches];
		int nicheQuota=Parameters.popSize/numNiches;
		for(i=0;i<numNiches;i++) nicheCounters[i]=nicheQuota;

		for(i=0;i<Parameters.popSize;i++) {
			// There can be only one
			int niche=selectNicheWOR(nicheCounters);
			winner=selectCandidateWOR(pools[niche],niche
					,_population);
			for(j=1;j<Parameters.tournamentSize;j++) {
				candidate=selectCandidateWOR(pools[niche]
						,niche,_population);
				if(_population[candidate].compareToIndividual(_population[winner])) {
					winner=candidate;
				}
			}
			selectedPopulation[i]=_population[winner].copy();
		}
		Chronometer.stopChronSelection();
		return selectedPopulation;
	}

	public void doMutation(Classifier[] _population) {
		Chronometer.startChronMutation();
		int popSize=Parameters.popSize;
		double probMut=Parameters.probMutationInd;
		
		for (int i=0; i<Parameters.popSize; i++) {
			if (Rand.getReal()<probMut) {
				_population[i].doMutation();
			}
		}

		doSpecialStages(_population);

		Chronometer.stopChronMutation();
	}

	void sortedInsert(ArrayList set,Classifier cl) {
		for(int i=0,max=set.size();i<max;i++) {
			if(cl.compareToIndividual((Classifier)set.get(i))) {
				set.add(i,cl);
				return;
			}
		}
		set.add(cl);
	}
			

	public Classifier[] replacementPolicy(Classifier[] offspring
			,boolean lastIteration) {
		int i;

		Chronometer.startChronReplacement();

		if(lastIteration) {
			for(i=0;i<numVersions;i++) {
				if(bestIndiv[i] != null) {
					PopulationWrapper.evaluateClassifier(
							bestIndiv[i]);
				}
			}
			ArrayList set=new ArrayList();
			for(i=0;i<Parameters.popSize;i++) {
				sortedInsert(set,offspring[i]);
			}
			for(i=0;i<numVersions;i++) {
				if(bestIndiv[i] != null) {
					sortedInsert(set,bestIndiv[i]);
				}
			}

			for(i=0;i<Parameters.popSize;i++) {
				offspring[i]=(Classifier)set.get(i);
			}
		} else {
			boolean previousVerUsed=false;
			int currVer=PopulationWrapper.getCurrentVersion();
			if(bestIndiv[currVer] == null && currVer>0) {
				previousVerUsed=true;
				currVer--;
			}

			if(bestIndiv[currVer] != null) {
				PopulationWrapper.evaluateClassifier(bestIndiv[currVer]);
				int worst=PopulationWrapper.getWorst(offspring);
				offspring[worst]=bestIndiv[currVer].copy();
			}
			if(!previousVerUsed) {
				int prevVer;
				if(currVer==0) {
					prevVer=numVersions-1;
				} else {
					prevVer=currVer-1;
				}
				if(bestIndiv[prevVer]!=null) {
					PopulationWrapper.evaluateClassifier(bestIndiv[prevVer]);
					int worst=PopulationWrapper.getWorst(offspring);
					offspring[worst]=bestIndiv[prevVer].copy();
				}
			}
		}
				
	        Chronometer.stopChronReplacement();
	        return offspring;
	}

	public Classifier[] replacementPolicy(Classifier[] parent,Classifier[] offspring, boolean lastIteration) {
		Classifier[] pop = new Classifier[Parameters.popSize];
		List<Classifier> classifierList = new ArrayList<Classifier>();
		for (int i = 0; i < parent.length; i++) {
			classifierList.add(parent[i]);
		}
		for (int i = 0; i < offspring.length; i++) {
			classifierList.add(offspring[i]);
		}
		Collections.sort(classifierList);

		for (int index = 0, i = classifierList.size() - 1; index < pop.length; index++, i--) {
			pop[index] = classifierList.get(i);
		}
		return pop;

	}
	
	public void doSpecialStages(Classifier []population) {
		int numStages=population[0].numSpecialStages();

		for(int i=0;i<numStages;i++) {
			for(int j=0;j<population.length;j++) {
				population[j].doSpecialStage(i);
			}
		}
	}
}
