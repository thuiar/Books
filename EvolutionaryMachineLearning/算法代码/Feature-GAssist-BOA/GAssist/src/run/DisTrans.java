package run;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;

import GAssist.LogManager;
import GAssist.Parameters;
import GAssist.ParserParameters;
import GAssist.PopulationWrapper;
import GAssist.Rand;
import boa.util.Transform;

public class DisTrans {
	public static void run(String confile, String _trainfile, String _testfile, boolean isBOA, int iter, String lonfile){
		long t1=0,t2=0;
		
		//String trainfile="training.arff", testfile="test.arff";
		String trainfile=Parameters.CHI_SQUARE+"_"+_trainfile, testfile=Parameters.CHI_SQUARE+"_"+_testfile;
		
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
		Transform.transform(_trainfile, trainfile, PopulationWrapper.getIntervals());
		Transform.transform(_testfile, testfile, PopulationWrapper.getIntervals());
		
	}
	
	public static void main(String[] args){
        String file="command.txt";
		try{
			InputStream is = new FileInputStream(file);
	        BufferedReader reader = new BufferedReader(new InputStreamReader(is));
	        
	    
	        String line = reader.readLine();       // 读取第一行
	        while (line != null) {          // 如果 line 为空说明读完了
	        	System.out.println(line);
	        	String[] parameter =new String [6];
	        	int indexOfPara =0;
	        	while(line.contains("\t")){
	        		int index = line.indexOf("\t");
	        		parameter[indexOfPara++]=line.substring(0,index);
	        		System.out.println(parameter[indexOfPara-1]);
	        		line = line.substring(index+1);
	        	}   
	        	parameter[indexOfPara++]=line;
	        	System.out.println(parameter[indexOfPara-1]);
	        	
	        	
	        	DisTrans.run(parameter[0], parameter[1], parameter[2], Boolean.parseBoolean(parameter[3]), Integer.parseInt(parameter[4]),parameter[5]);
	        	
	        	line = reader.readLine();
	        }
		}catch(Exception e){
			e.printStackTrace();
		}
	}
}
