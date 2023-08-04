package run;

import boa.util.MyUtil;

public class Stat {
	public static void main(String[] arg){
	
		//String dir = "D://Document/Paper/my paper/BOA-Feature/data/new";
		String dir = "D://Document/Paper/my paper/multiclass/data/new";
		
		String data = "/iris";
		String method= "/GA";
		String type= "/feq";
		String[] result= new String[9];
		String[] r=null;

//		r = MyUtil.stepStat(dir+data+method+type+"_1_acc.txt",1);
//		for(int i=0;i<r.length ;i++){
//			result[i] = r[i]; 			
//		}
		
		
		r = MyUtil.stepStat(dir+data+method+type+"_0_acc.txt",30);
		for(int i=0;i<r.length ;i++){
			result[i] = r[i]; 			
		}
		r = MyUtil.stepStat(dir+data+method+type+"_1_acc.txt",30);
		for(int i=0;i<r.length ;i++){
			result[i] = result[i]+"\t"+r[i]; 			
		}
		r = MyUtil.stepStat(dir+data+method+type+"_2_acc.txt",30);
		for(int i=0;i<r.length ;i++){
			result[i] = result[i]+"\t"+r[i]; 			
		}
		r = MyUtil.stepStat(dir+data+method+type+"_3_acc.txt",30);
		for(int i=0;i<r.length ;i++){
			result[i] = result[i]+"\t"+r[i]; 			
		}
		r = MyUtil.stepStat(dir+data+method+type+"_4_acc.txt",30);
		for(int i=0;i<r.length ;i++){
			result[i] = result[i]+"\t"+r[i]; 			
		}
		
		for(int i=0;i<r.length ;i++){
			System.out.println(result[i]); 			
		}
		
	}
	
	
}


