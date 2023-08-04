package run;

import boa.util.MyUtil;

public class TrainTest {
	public static void main(String[] arg){
		
		for(int i=0;i<5;i++){
			MyUtil.splitFile("musk2.arff",i);
			
		}

	}
}
