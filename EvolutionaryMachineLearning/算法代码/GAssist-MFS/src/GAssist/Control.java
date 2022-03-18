/*
 * Control.java
 *
 */

package GAssist;

public class Control {
	
	/** Creates a new instance of Control
	 *  主控程序 */
	public Control() {
	}
	
	/**
	 * @param args the command line arguments
	 * 配置文件、训练集文件、测试集文件
	 */
	public static void main(String[] args) {
		long t1=System.currentTimeMillis();
		
		String confile= "config.txt";
		String trainfile= "training1.arff";
		String testfile= "test1.arff";
		
		
		ParserParameters.doParse(confile);
		Parameters.trainFile=trainfile;
		Parameters.testFile=testfile;
		LogManager.initLogManager();
		Rand.initRand();
	  
		GA ga=new GA();
		ga.initGA();
		ga.run();
	   
		LogManager.println(Chronometer.getChrons());
		long t2=System.currentTimeMillis();
		LogManager.println("Total time: "+((t2-t1)/1000.0));

		LogManager.closeLog();
	}
	
}
