package run;

import GAssist.Chronometer;
import GAssist.GA;
import GAssist.LogManager;
import GAssist.Parameters;
import GAssist.ParserParameters;
import GAssist.PopulationWrapper;
import GAssist.Rand;

public class BOAGAssist {
	public static void run(String confile, String trainfile, String testfile, boolean isBOA, int iter, String lonfile){
		long t1=0,t2=0;
		
		ParserParameters.doParse(confile);
		Parameters.trainFile=trainfile;
		Parameters.testFile=testfile;
		LogManager.initLogManager(lonfile);
		
		LogManager.println_file(trainfile);
		
		LogManager.println_file("BOA: "+ isBOA);
		
		Rand.initRand();
		
		
		t2=System.currentTimeMillis();
		for(int i=0;i<iter;i++){
			PopulationWrapper.initInstancesEvaluation();
			
			
			t1=System.currentTimeMillis();
			while((t1-t2)/1000<1){
				t1=System.currentTimeMillis();
			}
			
			Parameters.reset();
			ParserParameters.doParse(confile);
			
			Rand.initRand();
			
			LogManager.println("Experiment: "+i);
			LogManager.println_file("Experiment: "+i);
			t1=System.currentTimeMillis();
			GA ga=new GA();
			ga.initGA();
			ga.run(isBOA);
			t2=System.currentTimeMillis();
			LogManager.println(Chronometer.getChrons());
			LogManager.println("Total time: "+((t2-t1)/1000.0));   
			LogManager.println_file("Total time: "+((t2-t1)/1000.0));
			LogManager.println("");
		}
		
		

		LogManager.closeLog();
	}
}
