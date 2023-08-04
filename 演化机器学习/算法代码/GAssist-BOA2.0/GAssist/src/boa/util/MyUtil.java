package boa.util;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;


public class MyUtil {
	private static List<Double> logRecord = new ArrayList<Double>();
	private static double logSum=0;
	
	
	public static int indexedBinaryToInt(char[] string, int[] pos){
		int value =0;
		for(int index=0;index<pos.length;index++){
			//value=(value<<1)+string[pos[index]]-'0';
			value=(value*2)+string[pos[index]]-'0';
		}
		return value;
	}
	
	
	public static int indexedBinaryToInt(char[] string, List<Integer> pos){
		int value =0;
		for(int index=0;index< pos.size();index++){
			//value=(value<<1)+string[pos.get(index)]-'0';
			value=value*2+string[pos.get(index)]-'0';
		}
		return value;
	}
	
	public static double  logSum(int n){
		double sum=0;
		for(int i=1;i<=n;i++){
			sum +=Math.log((double)i);
	    }
	    return sum;

	}
	
	public static double logSumDifference(int m, int n){
		if(logRecord.size()==0)
			logRecord.add(0.0);
		if(logRecord.size()<=n){
			for(int i=logRecord.size();i<=n;i++){
				logSum+=Math.log((double)i);
				logRecord.add(logSum);
			}
		}
		return logRecord.get(n)-logRecord.get(m);
		
		
		// n > m: LogSum(n) -LogSum(m)
//		double sum=0, subtracted = 0;
//		for(int i=1;i<=n;i++){
//			sum +=Math.log((double)i);
//			if(i==m){
//				subtracted = sum;
//			} 
//	    }
//	    return sum-subtracted;
	}
	
	public static int roulettewheelSelection(double[] wheel){
		double r = Math.random();
		double sum=0;
		for(int i=0;i<wheel.length ;i++){
			sum+=wheel[i];
			if(sum>r)  return i;
		}
		return wheel.length-1;
		
	}
	
	public static void splitFile(String file, int seed) {
		
		String trainfile= seed+"-training-"+file;
		String testfile=  seed+"-test-"+file;
		
		
		try{
			InputStream is = new FileInputStream(file);
			
			OutputStream osTrain = new FileOutputStream(trainfile);
			OutputStream osTest = new FileOutputStream(testfile);
			
			PrintStream psTrain = new PrintStream(osTrain);   
			PrintStream psTest = new PrintStream(osTest);     
			
			
			StringBuffer buffer =new StringBuffer();
			String line;        // 用来保存每行读取的内容
	        BufferedReader reader = new BufferedReader(new InputStreamReader(is));
	        
	        boolean dataFlag =false;
	        int counter=1;
	        
	        line = reader.readLine();       // 读取第一行
	        while (line != null) {          // 如果 line 为空说明读完了
	        	System.out.println(line);
	            buffer.append(line);        // 将读到的内容添加到 buffer 中
	            buffer.append("\n");        // 添加换行符
	            
	            if(!dataFlag){
	            	psTrain.println(line);
	            	psTest.println(line);
	            	if(line.matches("@data")){
	            		dataFlag = true;
	            	}
	            }else{
	            	if(counter % 5 ==seed)
	            		psTest.println(line);
	            	else
	            		psTrain.println(line);
	            	counter++;
	            }
	            
	            line = reader.readLine();   // 读取下一行
	            
	        }
	        is.close();
	        osTrain.close();
	        osTest.close();
	        
	        
	        
		}catch(Exception e){
			e.printStackTrace();
		}

	}
	
	public static void iter_match(String input, String output){
		
		double lastAcc=-1,lastMatch = -1;
		
		try{
			InputStream is = new FileInputStream(input);
			BufferedReader reader = new BufferedReader(new InputStreamReader(is));
			
			OutputStream os = new FileOutputStream(output);
			PrintStream ps = new PrintStream(os);   
			
			
	        
	        boolean dataFlag =false;
	        int counter=1;
	        
	        String line = reader.readLine();       // 读取第一行
	        while (line != null) {
	        	String[] para = line.split("\t");
	        	double acc = Double.parseDouble(para[0]);
	        	double match = Double.parseDouble(para[1]);
	        	
	        	if(lastAcc!=-1){
	        		int begin = (int) lastMatch;
	        		int end = (int) match;
	        		if(begin!=end){
	        			for(int i=begin+1;i<=end;i++){
	        				double temp = lastAcc+(i-lastMatch)/(match-lastMatch)*(acc-lastAcc);
	        				System.out.println(i+"\t"+temp);
	        				if(i%10==0){
	        					ps.println(i+"\t"+temp);
	        				}
	        			}
	        		}
	        		
	        	}
	        	lastAcc = acc;
	        	lastMatch = match;
	        	line = reader.readLine();
	        }
	        is.close();
	        ps.close();
	        os.close();
	        
		}catch(Exception ex){
			ex.printStackTrace();
		}
		
	}
}
