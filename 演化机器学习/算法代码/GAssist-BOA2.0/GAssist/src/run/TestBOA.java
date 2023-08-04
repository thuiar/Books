package run;

import boa.BOA;

public class TestBOA {
		
	public static void main(String arg[]){
		
		long start = System.currentTimeMillis();
		BOA boa =new BOA();
		boa.boa();
		long end = System.currentTimeMillis();
		System.out.println("Time consumed:"+(double)(end-start)/1000);
		
	}
}
