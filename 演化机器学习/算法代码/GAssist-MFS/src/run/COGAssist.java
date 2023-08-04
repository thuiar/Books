package run;

import GAssist.COCEA;
import GAssist.Chronometer;
import GAssist.GA;
import GAssist.LogManager;
import GAssist.Parameters;
import GAssist.ParserParameters;
import GAssist.PopulationWrapper;
import GAssist.Rand;
import MAFS.FSParserParameters;

public class COGAssist {
	public static void run(String confile, String fsconfile, String trainfile, String testfile, int iter, String lonfile){
		long t1=0,t2=0;
		
		
		LogManager.initLogManager(lonfile);
		LogManager.println_file(trainfile);
		Rand.initRand();
		
		t2=System.currentTimeMillis();
		for(int i=0;i<iter;i++){
			
//			ParserParameters.doParse(confile);
//			Parameters.trainFile=trainfile;
//			Parameters.testFile=testfile;
//			PopulationWrapper.initInstancesEvaluation();
//			t1=System.currentTimeMillis();
//			while((t1-t2)/1000<1){
//				t1=System.currentTimeMillis();
//			}
//			
//			Parameters.reset();
//			ParserParameters.doParse(confile);
			

//			GA ga=new GA();
//			ga.initGA();
//			ga.run();
//			System.out.println("------------------------------ReRun------------------------------");
//			int attr = (int)(Parameters.numAttributes*Math.random());
//			System.out.println(attr);
//			int[] attrs=new int[]{attr};
//			ga.deleteAttribute(attrs);
//			ga.deleteAttribute();
//			ga.run();
			t2=System.currentTimeMillis();
			LogManager.println(Chronometer.getChrons());
			LogManager.println("Total time: "+((t2-t1)/1000.0));   
			LogManager.println_file("Total time: "+((t2-t1)/1000.0));
			LogManager.println("");
			
//			long t1=System.currentTimeMillis();
			
//			String confile= "config.txt";
//			String fsconfile = "fsconfig.txt";
//			String trainfile= "training1.arff";
//			String testfile= "test1.arff";
			
			
			ParserParameters.doParse(confile);
			FSParserParameters.doParse(fsconfile);
			Parameters.trainFile=trainfile;
			Parameters.testFile=testfile;
			LogManager.initLogManager();
			Rand.initRand();
			LogManager.println("Experiment: "+i);
			LogManager.println_file("Experiment: "+i);
			t1=System.currentTimeMillis();

			COCEA cocea = new COCEA();
			cocea.initCOCEA();
			cocea.run();
			LogManager.println(Chronometer.getChrons());
//			LogManager.println(Chronometer.getChrons());
			t2=System.currentTimeMillis();
			LogManager.println("Total time: "+((t2-t1)/1000.0));

//			LogManager.closeLog();
		}
		
		

		LogManager.closeLog();
	}
}
