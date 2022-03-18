package run;

import GAssist.Chronometer;
import GAssist.GA;
import GAssist.LogManager;
import GAssist.Parameters;
import GAssist.ParserParameters;
import GAssist.PopulationWrapper;
import GAssist.Rand;

public class BOAGAssist {
	public static void run(String confile, String _trainfile, String _testfile, boolean isBOA, int iter, String lonfile){
		long t1=0,t2=0;
		
		//String trainfile="training.arff", testfile="test.arff";
		String trainfile="nominal_"+_trainfile, testfile="nominal_"+_testfile;
		
		
		
		ParserParameters.doParse(confile);
		
		Parameters.trainFile=_trainfile;
		Parameters.testFile=_testfile;
		Parameters.setChiQuare();
		LogManager.initLogManager(lonfile);
		
		//LogManager.println_file(trainfile);
		
		LogManager.println_file("BOA: "+ isBOA);
		
		Rand.initRand();
		
//		System.out.println("init---------------------------");
		PopulationWrapper.initInstancesEvaluation();
		
		/*transfer real to nomination*******************************************/
//		Transform.transform(_trainfile, trainfile, PopulationWrapper.getIntervals());
//		Transform.transform(_testfile, testfile, PopulationWrapper.getIntervals());
		
//		Parameters.trainFile=trainfile;
//		Parameters.testFile=testfile;
		/**********************************************************************/
		
		
		t2=System.currentTimeMillis();
		for(int i=0;i<iter;i++){
			Parameters.trainFile=_trainfile;
			Parameters.testFile=_testfile;
			
//			System.out.println("trans---------------------------");
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
