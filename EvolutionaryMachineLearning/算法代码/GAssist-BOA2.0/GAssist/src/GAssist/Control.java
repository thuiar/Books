/*
 * Control.java
 *
 */

package GAssist;

public class Control {
	
	/** Creates a new instance of Control */
	public Control() {
	}
	
	/**
	 * @param args the command line arguments
	 */
	public static void main(String[] args) {
		long t1=System.currentTimeMillis();
		
		String confile= "config.txt";
		String trainfile= "training.arff";
		String testfile= "test.arff";
		
		
		ParserParameters.doParse(confile);
		Parameters.trainFile=trainfile;
		Parameters.testFile=testfile;
		LogManager.initLogManager();
		Rand.initRand();
	  
		
		
		GA ga=new GA();
		ga.initGA();
		ga.run(true);
	   
		LogManager.println(Chronometer.getChrons());
		long t2=System.currentTimeMillis();
		LogManager.println("Total time: "+((t2-t1)/1000.0));

		LogManager.closeLog();
	}
	
}
