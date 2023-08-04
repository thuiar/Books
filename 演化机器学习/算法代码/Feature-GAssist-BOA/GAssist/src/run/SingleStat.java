package run;

import boa.util.MyUtil;

public class SingleStat {
	
		public static void main(String[] arg){
			String dir = "D://Document/Paper/my paper/BOA-Feature/data/new";
			//String dir = "D://Document/Paper/my paper/multiclass/data/new";
			
			String data = "/wdbc";
			String method= "/reduced";
			String type= "/9999";
			String index = "_4info";
			String[] result= new String[9];
			String[] r=null;

			r = MyUtil.stepStat("D://workspace/workplace for java/multiclass-GAssist-BOA/GAssist/feature/allbp/best",30);
			
			//r = MyUtil.stepStat(dir+data+method+type+index+"_acc.txt",30);
			for(int i=0;i<r.length ;i++){
				result[i] = r[i]; 			
			}
			
		
		}
}
