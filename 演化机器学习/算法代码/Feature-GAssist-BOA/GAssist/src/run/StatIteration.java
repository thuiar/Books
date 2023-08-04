package run;

import boa.util.MyUtil;

public class StatIteration {

	public static void main(String[] arg){
		
		String dir = "D://Document/Paper/my paper/BOA-Feature/data/new";
		//String dir = "D://Document/Paper/my paper/multiclass/data/new";
		
		String data = "/mush";
		String method= "/feature";
		String type= "/nominal";
		String index= "_01";
		
		
		//MyUtil.iterationStat(dir+data+method+type+index+"_acc.txt",30);
		MyUtil.matchQuery(dir+data+method+type+index+"_acc.txt",30, 160);
		
		
	}
	
}


