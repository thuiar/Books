package boa.util;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

import GAssist.Dataset.Attributes;
import GAssist.Dataset.Instance;


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
		
		int index =  file.indexOf("/");
		String filename = file.substring(index+1);
		
		if(index==-1) filename=file;
		
		String trainfile= "data/"+seed+"-training-"+filename;
		String testfile=  "data/"+seed+"-test-"+filename;
		
		
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
	
	public static boolean arrayEqual(char[] a, char[] b){
		if(a.length!=b.length) return false;
		for(int i=0;i<a.length;i++){
			if(a[i]!=b[i]){
				return false;
			}
		}
		return true;
	}
	
	public static void printlArray(char[] a){
		for(int i=0;i<a.length ;i++){
			System.out.print(a[i]);
		}
		System.out.println();
	}
	
	public static void printArray(double[] a){
		for(int i=0;i<a.length ;i++){
			System.out.print(a[i]+"\t");
		}
		System.out.println();
	}
	
	public static void printArray(int[] a){
		for(int i=0;i<a.length ;i++){
			System.out.print(a[i]+"\t");
		}
		System.out.println();
	}
	
	
	
	
	
	public static String[] stepStat(String file,int numExp ){
		String[] results =new String[9];
		
		try{
			
			
			InputStream is = new FileInputStream(file);
			
			int counter = -1;
			double[] step  = new double[numExp];
			double[] trainAcc  = new double[numExp];
			double[] testAcc  = new double[numExp];
			int[] numTotalRules  = new int[numExp];
			int[] numFuncRules  = new int[numExp];
			
			double[] numDelete =   new double[numExp];
			double[] time = new double[numExp];
			
			long[] match = new long[numExp]; 
			
			boolean flag =false;
			double lastAcc = 0, delete = 0; ;
			
			
			
			String line;        // 用来保存每行读取的内容
	        BufferedReader reader = new BufferedReader(new InputStreamReader(is));
	        
	        
	        line = reader.readLine();       // 读取第一行
	        while (line != null) {          // 如果 line 为空说明读完了
	        	//System.out.println(line);
	        	if(line.contains("Experiment:")){
	        		counter++;
	        		flag = true;
	        		lastAcc = 0;
	        		delete =0;
	        	}else if(line.contains("Match:")){
	        		int split = line.indexOf("\t");
					match[counter] = Long.parseLong(line.substring(split+1).trim());	
	        	}else if(line.contains("Delete")){
	        		String p[] = line.split(",");
	        		delete+=p.length-1;
	        	}else if(line.contains("training accuracy")){
	        		int split = line.indexOf("\t");
	        		trainAcc[counter] = Double.parseDouble(line.substring(split));
	        		numDelete[counter] = delete;
	        		flag = false;
	        	}else if(line.contains("test accuracy")){
	        		int split = line.indexOf("\t");
	        		testAcc[counter] = Double.parseDouble(line.substring(split));
	        		flag = false;
	        		//System.out.println(testAcc[counter]);
	        	}else if (line.contains("Total time:")){
					int split = line.indexOf(":");
					time[counter] = Double.parseDouble(line.substring(split + 1));	
				}else if(flag&&line.contains("iteration")){
	        		int split1 = line.indexOf("iteration ");
	        		int split2 = line.indexOf(" :");
	        		int iteration = Integer.parseInt(line.substring(split1+10, split2));
	        		
	        		String subline = line.substring(split2+3);
	        		int split3 = subline.indexOf(" ");
	        		//System.out.println(subline);
	        		Double tempAcc = Double.parseDouble(subline.substring(0, split3));
	        		if(lastAcc!=tempAcc){
	        			step[counter]=iteration;
	        			lastAcc=tempAcc;
	//        			System.out.println(step[counter]);
	        		}
	        		
	        		int split4 = line.indexOf("(");
	        		String numRules = line.substring(split4-3);
	        		split4 = numRules.indexOf(" ");
	        		numRules = numRules.substring(split4+1);
	        		int leftBracket  = numRules.indexOf("(");
	        		int rightBracket = numRules.indexOf(")");
	        		//numTotalRules[counter] = Integer.parseInt(numRules.substring(0,leftBracket));
	        		//numFuncRules[counter] = Integer.parseInt(numRules.substring(leftBracket+1,rightBracket));
	        		
	        	}
	        	line = reader.readLine(); 
	        }
	        is.close();
	        
	        
	        double aveStep=0,aveTrainAcc=0,aveTestAcc=0, aveTotalRules=0, aveFuncRules=0, aveDelete = 0, aveTime=0, aveMatch=0;
	        for(int i=0;i<step.length ;i++){
	        	aveStep+=step[i];
	        	aveTrainAcc+=trainAcc[i];
	        	aveTestAcc+=testAcc[i];
	        	aveTotalRules+=numTotalRules[i];
	        	aveFuncRules+=numFuncRules[i];
	        	
	        	aveDelete+=numDelete[i];
	        	aveTime+=time[i];
	        	
	        	aveMatch+=match[i];
	        	
	        }
	        
	        aveStep/=step.length ;
	        aveTrainAcc/=step.length ;
	        aveTestAcc/=step.length ;
	        aveTotalRules/=step.length ;
	        aveFuncRules/=step.length ;
	        
	        aveDelete/=step.length ;
	        aveMatch/=step.length ;
	        
	        double stdStep=0,stdTrainAcc=0,stdTestAcc=0,stdTotalRules=0, stdFuncRules=0, stdDelete = 0, stdTime = 0, stdMatch = 0;
	        for(int i=0;i<step.length ;i++){
	        	stdStep+=(step[i]-aveStep)*(step[i]-aveStep);
	        	stdTrainAcc+=(trainAcc[i]-aveTrainAcc)*(trainAcc[i]-aveTrainAcc);
	        	stdTestAcc+=(testAcc[i]-aveTestAcc)*(testAcc[i]-aveTestAcc);	
	        	stdTotalRules+=(numTotalRules[i]-aveTotalRules)*(numTotalRules[i]-aveTotalRules);
	        	stdFuncRules+=(numFuncRules[i]-aveFuncRules)*(numFuncRules[i]-aveFuncRules);
	        	
	        	stdDelete+=(numDelete[i]-aveDelete)*(numDelete[i]-aveDelete);
	        	stdTime += (time[i] - aveTime)	* (time[i] - aveTime);
	        	stdMatch+= (match[i] - aveMatch)* (match[i] - aveMatch);
	        	
	        }
	        
	        stdStep=Math.sqrt(stdStep/(step.length-1));
	        stdTrainAcc=Math.sqrt(stdTrainAcc/(step.length-1));
	        stdTestAcc=Math.sqrt(stdTestAcc/(step.length-1));
	        stdTotalRules=Math.sqrt(stdTotalRules/(step.length-1));
	        stdFuncRules=Math.sqrt(stdFuncRules/(step.length-1));
	        
	        stdDelete=Math.sqrt(stdDelete/(step.length-1));
	        stdTime = Math.sqrt(stdTime / (step.length - 1));
	        stdMatch = Math.sqrt(stdMatch / (step.length - 1));
	        
	        StringBuffer buf = new StringBuffer();
	        
	        System.out.println(counter);
	        results[0] =  Integer.toString(counter) + "\t";
	        
	        buf = new StringBuffer();
	        System.out.print("Step:");
	        System.out.print("\t");
	        System.out.print(aveStep);buf.append(aveStep);
	        System.out.print("\t");buf.append("\t");
	        System.out.print(stdStep);buf.append(stdStep);
	        System.out.print("\n");
	        results[1] = buf.toString();
	        
	        buf = new StringBuffer();
	        System.out.print("TrainAcc:");
	        System.out.print("\t");
	        System.out.print(aveTrainAcc);buf.append(aveTrainAcc);
	        System.out.print("\t");buf.append("\t");
	        System.out.print(stdTrainAcc);buf.append(stdTrainAcc);
	        System.out.print("\n");
	        results[2] = buf.toString();
	        
	        buf = new StringBuffer();
	        System.out.print("TestAcc:");
	        System.out.print("\t");
	        System.out.print(aveTestAcc);buf.append(aveTestAcc);
	        System.out.print("\t");buf.append("\t");
	        System.out.print(stdTestAcc);buf.append(stdTestAcc);
	        System.out.print("\n");
	        results[3] = buf.toString();
	                
	        DecimalFormat f = new DecimalFormat("0.000");
	      
	        buf = new StringBuffer();
	        System.out.print("TotalRules:");
	        System.out.print("\t");
	        System.out.print(f.format(aveTotalRules));buf.append(f.format(aveTotalRules));
	        System.out.print("\t");buf.append("\t");
	        System.out.print(f.format(stdTotalRules));buf.append(f.format(stdTotalRules));
	        System.out.print("\n");
	        results[4] = buf.toString();
	        
	        
	        buf = new StringBuffer();
	        System.out.print("FuncRules:");
	        System.out.print("\t");
	        System.out.print(f.format(aveFuncRules));buf.append(f.format(aveFuncRules));
	        System.out.print("\t");buf.append("\t");
	        System.out.print(f.format(stdFuncRules));buf.append(f.format(stdFuncRules));
	        System.out.print("\n"); 
	        results[5] = buf.toString();
	        
	        buf = new StringBuffer();
	        System.out.print("Delete:");
	        System.out.print("\t");
	        System.out.print(f.format(aveDelete));buf.append(f.format(aveDelete));
	        System.out.print("\t");buf.append("\t");
	        System.out.print(f.format(stdDelete));buf.append(f.format(stdDelete));
	        System.out.print("\n");
	        results[6] = buf.toString();
	        
	        buf = new StringBuffer();
	        System.out.print("Match:");
	        System.out.print("\t");
	        System.out.print(f.format(aveMatch));buf.append(f.format(aveMatch));
	        System.out.print("\t");buf.append("\t");
	        System.out.print(f.format(stdMatch));buf.append(f.format(stdMatch));
	        System.out.print("\n");
	        results[7] = buf.toString();
	        
	        buf = new StringBuffer();
	        System.out.print("Time:");
	        System.out.print("\t");
			System.out.print(f.format(aveTime));buf.append(f.format(aveTime));
			System.out.print("\t");buf.append("\t");
			System.out.print(f.format(stdTime));buf.append(f.format(stdTime));
			System.out.println("\n");
			results[8] = buf.toString();
	        
	        
		}catch(Exception e){
			e.printStackTrace();
		}
		
		
		return results;
	}
	
	public static void steplineStat(String file,int numExp, int numStep ){
		
		int interval = 	10; 
		
		try{
			
			
			InputStream is = new FileInputStream(file);
			
			int counter = -1;

			

			double[] stepAcc = new double[numStep/interval];
			for(int i=0;i<stepAcc.length;i++){
				//stepAcc[i]=1;
				
			}
			double[] matchNum = new double[numStep/interval];
			int index = 0;
			boolean flag =false;

			
			String line;        // 用来保存每行读取的内容
	        BufferedReader reader = new BufferedReader(new InputStreamReader(is));
	        
	        
	        line = reader.readLine();       // 读取第一行
	        while (line != null) {          // 如果 line 为空说明读完了
	        	//System.out.println(line);
	        	if(line.contains("Experiment:")){
	        		counter++;
	        		flag = true;
	        		index = 0;
	        	}else if(line.contains("training accuracy")){
	        		flag = false;
	        		if(index<numStep-interval){
	        			index+=interval;
	        			for(;index<numStep;index+=interval){
	        				stepAcc[index/interval]+=1;
	        			}
	        		}
	        	}else if(line.contains("test accuracy")){
	        		flag = false;
	        		//System.out.println(testAcc[counter]);
	        	}else if(flag){
	        		String[] para = line.split(" ");
	        		if(Integer.parseInt(para[3])%interval!=0){
	        			line = reader.readLine(); 
	        			continue;
	        		}
	        		
	        		int split1 = line.indexOf("iteration ");
	        		int split2 = line.indexOf(" :");
	        		int iteration = Integer.parseInt(line.substring(split1+10, split2));
	        		index = iteration;
	        		String subline = line.substring(split2+3);
	        		int split3 = subline.indexOf(" ");
	        		//System.out.println(subline);
	        		Double tempAcc = Double.parseDouble(subline.substring(0, split3));
	        		stepAcc[iteration/interval]+=tempAcc;
	        		
	        		Double tempMatch = Double.parseDouble(para[para.length-1]);
	        		matchNum[iteration/interval]+=tempMatch;
	        		
	        	}
	        	line = reader.readLine(); 
	        }
	        is.close();
	        
	        for(int i=0;i<numStep/interval;i++){
	        	stepAcc[i]/=numExp;
	        	
	        	matchNum[i]/=numExp;
	        	
	        	//System.out.print(i);
	        	//System.out.print("\t");
	        	//if(i%2==1)
	        		System.out.println(i+"\t"+stepAcc[i]+"\t"+matchNum[i]);
	        	
	        	
	        	
	        }
	      
	        
	        	        
	        
	        
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	
	
	public static void linkStat(String file,int numExp){
		int counter = -1;
		try{
			
			
			InputStream is = new FileInputStream(file);
			
			
			boolean flag =false;
			
			double twin=0,group=0,link=0,total=0;
			
			
			String line;        // 用来保存每行读取的内容
	        BufferedReader reader = new BufferedReader(new InputStreamReader(is));
	        
	        
	        line = reader.readLine();       // 读取第一行
	        while (line != null) {          // 如果 line 为空说明读完了
	        	//System.out.println(line);
	        	if(line.contains("Experiment:")){
	        		counter++;
	        		flag = true;
	        		
	        	}else if(line.contains("Children")){	        		
	        		int split1 = line.indexOf("of");
	        		int parentIndex = Integer.parseInt(line.substring(split1+3,line.length()-1));
	        		int parentGroup=parentIndex/4;
	        		line = reader.readLine(); 
	        		line =line.substring(1);
	        		while(line.contains(" ")){
	        			int split = line.indexOf(" ");
	        			String temp = line.substring(0, split);
	        			int childIndex = Integer.parseInt(temp);
	        			int childGroup = childIndex/4;
	        			if(childIndex/2==parentIndex/2){
	        				twin++;
	        			}else if(childGroup==parentGroup){
	        				group++;
	        			}else if(childGroup<2||parentGroup<2){
	        				link++;
	        			}
	        			total++;
	        			line = line.substring(split+1);
	        		}
	        		
	        		
	        	}else if(line.contains("training accuracy")){
	        		flag = false;
	        	}else if(line.contains("iteration")){	        		
	        		flag = false;
	        		System.out.print(counter+":\t");
	        		System.out.print(total+"\t");
	     	        System.out.print(twin/total+"\t");
	     			System.out.print(group/total+"\t");
	     			System.out.print(link/total+"\t");
	     			System.out.println(twin/total+group/total+link/total);
	     			twin=0;group=0;link=0;total=0;
	        	}else if(flag){
	        			}
	        	line = reader.readLine(); 
	        }
	        is.close();
	        
	        System.out.println(counter);
	       
		
		}catch(Exception ex){
			System.out.println(counter);
			ex.printStackTrace();
		}
			
	}
	
	public  static int discretizer(double data, double[] split){ 
		for(int i=0;i<split.length;i++){
			if(data<split[i]) return i; 
		}
		return split.length;
	}
	
	public static String getArray(int max){
		StringBuffer buf =new StringBuffer();
		buf.append("{");
		for(int i=0;i<max;i++){
			buf.append(i);
			buf.append(",");
		}
		buf.append(max);
		buf.append("}");
		return buf.toString();
		
	}
	
	public static boolean isSameArray(int[] a, int[] b, int length){
		for(int i=0;i<length;i++){
			if(a[i]!=b[i]) return false;
		}
		return true;
	}
	
	public static boolean isSameInstance(Instance a, Instance b){
		for(int i=0;i<Attributes.getNumAttributes()-1;i++){
			if(a.getRealAttribute(i)!=
				b.getRealAttribute(i)) return false;
		}
		return true;
	}
	
	public static void deleteAttribute(String srcFile, String desFile,	int[] attrs) {
		try {
			
			copyFile(srcFile);
			
			InputStream is = new FileInputStream("temp.txt");
			OutputStream os = new FileOutputStream(desFile);
			PrintStream ps = new PrintStream(os);

			StringBuffer buffer = new StringBuffer();
			String line; // 用来保存每行读取的内容
			BufferedReader reader = new BufferedReader(
					new InputStreamReader(is));

			int attrCounter = 0;
			int attrDelCounter = 0;

			line = reader.readLine(); // 读取第一行
			while (line != null) { // 如果 line 为空说明读完了
				// System.out.println(line);
				if (line.contains("@attribute")) {
					if (attrCounter != attrs[attrDelCounter]) {
						ps.println(line);
					} else if (attrDelCounter < attrs.length - 1) {
						attrDelCounter++;
					}
					attrCounter++;
				} else if (line.contains(",")) {
					attrDelCounter = 0;
					String[] values = line.split(",");
					buffer = new StringBuffer();
					for (int i = 0; i < values.length; i++) {
						if (i != attrs[attrDelCounter]) {
							if (buffer.length() != 0)
								buffer.append(",");
							buffer.append(values[i]);
						} else if (attrDelCounter < attrs.length - 1) {
							attrDelCounter++;
						}
					}
					ps.println(buffer.toString());
				} else {
					ps.println(line);
				}
				line = reader.readLine(); // 读取下一行

			}
			is.close();
			os.close();

		} catch (Exception e) {
			e.printStackTrace();
		}

	}
	
	public static void copyFile(String srcFile) {
		try {
			InputStream is = new FileInputStream(srcFile);
			OutputStream os = new FileOutputStream("temp.txt");
			PrintStream ps = new PrintStream(os);
			BufferedReader reader = new BufferedReader(
					new InputStreamReader(is));
			String line; // 用来保存每行读取的内容

			line = reader.readLine(); // 读取第一行
			while (line != null) { // 如果 line 为空说明读完了
				ps.println(line);
				line = reader.readLine(); // 读取下一行

			}
			is.close();
			os.close();

		} catch (Exception e) {
			e.printStackTrace();
		}

	}
	
	
	
	
	public static String[] iterationStat(String file,int numExp ){
		String[] results =new String[9];
		
		long[] match = new long[numExp-1];
		long[] metaMatch = new long[numExp-1];
		long[] iter = new long[numExp-1];
		try{
			
			
			InputStream is = new FileInputStream(file);
			
			int counter = -1;
			double[] step  = new double[numExp];
			
			boolean flag =false;
			double lastAcc = 0, delete = 0; ;
			String lastLine = "";
			
			
			String line;        // 用来保存每行读取的内容
	        BufferedReader reader = new BufferedReader(new InputStreamReader(is));
	        
	        
	        line = reader.readLine();       // 读取第一行
	        while (line != null) {          // 如果 line 为空说明读完了
	        	//System.out.println(line);
	        	if(line.contains("Experiment:")){
	        		String[] ps = lastLine.split(" ");
	        		for(int i=0;i<ps.length;i++){
	        			System.out.print(ps[i]+"\t");
	        		}
	        		System.out.println();
	        		if(!lastLine.matches("")){
	        			iter[counter] = Long.parseLong(ps[3]);
	        			match[counter] = Long.parseLong(ps[ps.length-2]);
	        			metaMatch[counter] = Long.parseLong(ps[ps.length-1]);
	        			
	        		}
	        		counter++;
	        		flag = true;
	        		lastAcc = Double.MIN_VALUE;
	        	}else if(line.contains("Delete")){
	        		lastAcc = Double.MIN_VALUE;
	        	}else if(line.contains("training accuracy")){
	        		flag = false;
				}else if(flag&&line.contains("iteration")){
	        		int split1 = line.indexOf("iteration ");
	        		int split2 = line.indexOf(" :");
	        		int iteration = Integer.parseInt(line.substring(split1+10, split2));
	        		
	        		String subline = line.substring(split2+3);
	        		int split3 = subline.indexOf(" ");
	        		//System.out.println(subline);
	        		Double tempAcc = Double.parseDouble(subline.substring(0, split3));
	        		if(lastAcc<tempAcc){
	        			step[counter]=iteration;
	        			lastAcc=tempAcc;
	        			lastLine =  line;
	//        			System.out.println(step[counter]);
	        		}
	        			
	        	}
	        	line = reader.readLine(); 
	        }
	        is.close();
	        
	        
		}catch(Exception e){
			e.printStackTrace();
		}
		
		double matchAve = 0, matchStd = 0;
		double metaMatchAve = 0, metaMatchStd = 0;
		double iterAve = 0, iterStd = 0;
		for(int i=0;i<match.length;i++){
			//System.out.print("\t"+iter[i]);
			matchAve+=match[i];
			metaMatchAve+=metaMatch[i];
			iterAve+=iter[i];
		}
		
		matchAve/= match.length;
		metaMatchAve/= match.length;
		iterAve/= iter.length;
		
		for(int i=0;i<match.length ;i++){
			matchStd+= (match[i] - matchAve)* (match[i] - matchAve);
			metaMatchStd+= (metaMatch[i] - metaMatchAve)* (metaMatch[i] - metaMatchAve);
			iterStd+= (iter[i] - iterAve)* (iter[i] - iterAve);
        }
        
		matchStd=Math.sqrt(matchStd/(match.length-1));
		metaMatchStd=Math.sqrt(metaMatchStd/(match.length-1));
		iterStd=Math.sqrt(iterStd/(iter.length-1));
		
	
		
		
		System.out.println();
		System.out.println(iterAve+"\t"+iterStd);
		System.out.println(matchAve/1E7+"\t"+matchStd/1E7);
		System.out.println(metaMatchAve/1E7+"\t"+metaMatchStd/1E7);
		
		
		return results;
	}

	
	public static String[] matchQuery(String file,int numExp, double iter ){
		String[] results =new String[9];
		
		try{
			
			
			InputStream is = new FileInputStream(file);
			
			int counter = -1;
			double[] match  = new double[numExp];
			double[] metaMatch  = new double[numExp];
			
			boolean flag =false;
			double lastAcc = 0, delete = 0; ;
			String lastLine = "";
			
			
			String line;        // 用来保存每行读取的内容
	        BufferedReader reader = new BufferedReader(new InputStreamReader(is));
	        
	        
	        line = reader.readLine();       // 读取第一行
	        while (line != null) {          // 如果 line 为空说明读完了
	        	//System.out.println(line);
	        	if(line.contains("Experiment:")){
//	        		System.out.print(lastLine);
	        		counter++;
	        		flag = true;
	        		lastAcc = 0;
	        	}else if(line.contains("training accuracy")){
	        		flag = false;
				}else if(flag&&line.contains("iteration")){
	        		int split1 = line.indexOf("iteration ");
	        		int split2 = line.indexOf(" :");
	        		int iteration = Integer.parseInt(line.substring(split1+10, split2));
	        		lastLine = line;
	        		if(iteration > iter){
	        			String[] p = line.split(" ");
	        			metaMatch[counter] = Double.parseDouble(p[p.length-1]);
	        			match[counter] = Double.parseDouble(p[p.length-2]);
	        			flag = false;
	        			System.out.println(counter+"\t"+lastLine);
	        		}
	        		
	        		
	        	}
	        	line = reader.readLine(); 
	        }
	        
	        is.close();
	        
	        double aveMatch =0, aveMetaMatch = 0;
	        for(int i=0;i<match.length ;i++){
	        	aveMatch+=match[i];
	        	aveMetaMatch+=metaMatch[i];
	        }
	        aveMatch/=match.length ;
	        aveMetaMatch/=metaMatch.length ;
	        
	        double stdMatch = 0, stdMetaMatch = 0;;
	        for(int i=0;i<match.length ;i++){
	        	stdMatch+= (match[i] - aveMatch)* (match[i] - aveMatch);	
	        	stdMetaMatch+= (metaMatch[i] - aveMetaMatch)* (metaMatch[i] - aveMetaMatch);
	        }
	        stdMatch=Math.sqrt(stdMatch/(match.length-1));
	        stdMetaMatch=Math.sqrt(stdMetaMatch/(metaMatch.length-1));
	        
	        
	        DecimalFormat f = new DecimalFormat("0.000");
	        System.out.print(f.format(aveMatch/1E7));
	        System.out.print("\t");
	        System.out.print(f.format(stdMatch/1E7));
	        System.out.print("\n");
	        
	        
	        System.out.print(f.format(aveMetaMatch/1E7));
	        System.out.print("\t");
	        System.out.print(f.format(stdMetaMatch/1E7));
	        System.out.print("\n");
	        
	        
		}catch(Exception e){
			e.printStackTrace();
		}
		
		
		return results;
	}
	
	public static int[] IntToBinary(int interger, int length){
		int[] binary =new int[length];
		int value = interger;
		int i=0;
		while(value>0){
			binary[i++]=value%2;
			value = value/2;
		}
		return binary;
	}
}
