package GAssist;

import java.util.ArrayList;

import GAssist.Dataset.Attributes;


public class Globals_DefaultC {
	static int defaultClassPolicy;
	static int defaultClass;
	static boolean enabled;
	static int numClasses;

	static boolean nichingEnabled;
	static int numNiches;
	static ArrayList[] accDefaultRules;
	

	public final static int DISABLED	= 1;
	public final static int MINOR		= 2;
	public final static int MAJOR		= 3;
	public final static int AUTO		= 4;

	public static void init(boolean hasDefaultClass) {
		nichingEnabled=false;
		
		if(!hasDefaultClass) {
			defaultClassPolicy=DISABLED;
			enabled=false;
			numClasses=Parameters.numClasses;
			return;
		}
		if(Parameters.defaultClass == null) {
			defaultClassPolicy=DISABLED;
			enabled=false;
			numClasses=Parameters.numClasses;
			return;
		}
		if(Parameters.defaultClass.equalsIgnoreCase("disabled")) {
			defaultClassPolicy=DISABLED;
			enabled=false;
			numClasses=Parameters.numClasses;
			
			
			
		} else if(Parameters.defaultClass.equalsIgnoreCase("major")) {
			defaultClassPolicy=MAJOR;
			defaultClass=Attributes.majorityClass;
			numClasses=Parameters.numClasses-1;
			enabled=true;
		} else if(Parameters.defaultClass.equalsIgnoreCase("minor")) {
			defaultClassPolicy=MINOR;
			defaultClass=Attributes.minorityClass;
			numClasses=Parameters.numClasses-1;
			enabled=true;
		} else if(Parameters.defaultClass.equalsIgnoreCase("auto")) {
			defaultClassPolicy=AUTO;
			numClasses=Parameters.numClasses-1;
			enabled=true;

			nichingEnabled=true;
			numNiches=Parameters.numClasses;
			accDefaultRules=new ArrayList[numNiches];
			for(int i=0;i<numNiches;i++) 
				accDefaultRules[i]=new ArrayList();
			
			defaultClass=0;
			
		} else {
			System.err.println("Unknown default class option "
					+Parameters.defaultClass);
			System.exit(1);
		}
	}

	static void checkNichingStatus(int iteration,Classifier[] population) {
		if(nichingEnabled) {
			int i;
			int[] counters=new int[numNiches];
			double []nicheFit=new double[numNiches];
			for(i=0;i<numNiches;i++) {
				counters[i]=0;
				nicheFit[i]=0;
			}

			for(i=0;i<population.length;i++) {
				int niche=population[i].getNiche();
				counters[niche]++;
				double indAcc=population[i].getAccuracy();
				if(indAcc>nicheFit[niche]) 
					nicheFit[niche]=indAcc;
			}

			if(accDefaultRules[0].size()==15) {
				for(i=0;i<numNiches;i++) 
					accDefaultRules[i].remove(0);
			}

			for(i=0;i<numNiches;i++) {
				accDefaultRules[i].add(new Double(nicheFit[i]));
			}

			if(accDefaultRules[0].size()==15) {
				ArrayList aves=new ArrayList();
				for(i=0;i<numNiches;i++) {
					double aveN=Utils.getAverage(
							accDefaultRules[i]);
					aves.add(new Double(aveN));
				}
				double dev=Utils.getDeviation(aves);
				if(dev<0.005) {
					LogManager.println("Iteration "+iteration+",niching disabled");
					nichingEnabled=false;
				}
			}
		}
	}
}
