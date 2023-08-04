package run;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;


public class Batch{
	public static void main(String[] args){
        String file="command.txt";
        if(args.length==2){
        	file=args[0];
        }
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
	        	
	        	if(args.length==1){
	        		parameter[5] = args[0]; 
	    		}else if(args.length==2){
	    			parameter[5] = args[1];
	    		}
	        	
	        	BOAGAssist.run(parameter[0], parameter[1], parameter[2], Boolean.parseBoolean(parameter[3]), Integer.parseInt(parameter[4]),parameter[5]);
	        	
	        	line = reader.readLine();
	        }
		}catch(Exception e){
			e.printStackTrace();
		}
	}
}
