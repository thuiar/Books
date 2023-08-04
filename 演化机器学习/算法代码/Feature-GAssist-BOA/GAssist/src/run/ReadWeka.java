package run;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;

public class ReadWeka {
	
	
	public static void main(String[] arg){
		ReadWeka.readSingle("G://Program Files/Weka-3-6/0.out");
		//System.out.println();
		ReadWeka.readSingle("G://Program Files/Weka-3-6/1.out");
		//System.out.println();
		ReadWeka.readSingle("G://Program Files/Weka-3-6/2.out");
		//System.out.println();
		ReadWeka.readSingle("G://Program Files/Weka-3-6/3.out");
		//System.out.println();
		ReadWeka.readSingle("G://Program Files/Weka-3-6/4.out");
	}
		
	public static void readSingle(String file){
		int type =-1;
		try{
			InputStream is = new FileInputStream(file);
	        BufferedReader reader = new BufferedReader(new InputStreamReader(is));
	        
	    
	        String line = reader.readLine();       // 读取第一行
	        while (line != null) {
	        	if(line.contains("Error on training data")){
	        		type++;
	        	}else if(line.contains("Correctly Classified Instances")){
	        		int index = line.indexOf("%");
	        		String value =  line.substring(index-10, index).trim();
	        		if(type==0){//Train
	        			System.out.print(value+"\t");
	        		}else if(type==1){//Test
	        			System.out.print(value+"\t");
	        		}
	        	}
	        	line = reader.readLine();
	        }
		}catch(Exception ex){
			ex.printStackTrace();
		}
	       
	}
}
