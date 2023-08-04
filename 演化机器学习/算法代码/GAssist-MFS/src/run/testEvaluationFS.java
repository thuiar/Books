package run;

import GAssist.COCEA;
import GAssist.Chronometer;
import GAssist.LogManager;
import GAssist.Parameters;
import GAssist.ParserParameters;
import GAssist.Rand;
import MAFS.FSParserParameters;

public class testEvaluationFS {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		long t1=System.currentTimeMillis();
		
		String confile= "config.txt";
		String fsconfile = "fsconfig.txt";
		String trainfile= "training1.arff";
		String testfile= "test1.arff";
		
		
		ParserParameters.doParse(confile);
		FSParserParameters.doParse(fsconfile);
		Parameters.trainFile=trainfile;
		Parameters.testFile=testfile;
		LogManager.initLogManager();
		Rand.initRand();
	  
//		GA ga=new GA();
//		ga.initGA();
//		ga.run();
		
		COCEA cocea = new COCEA();
		cocea.initCOCEA();
		cocea.run();
	   
		LogManager.println(Chronometer.getChrons());
//		LogManager.println(Chronometer.getChrons());
		long t2=System.currentTimeMillis();
		LogManager.println("Total time: "+((t2-t1)/1000.0));

		LogManager.closeLog();
	}

}
