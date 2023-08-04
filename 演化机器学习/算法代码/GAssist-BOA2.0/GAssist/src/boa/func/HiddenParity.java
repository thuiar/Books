package boa.func;

public class HiddenParity {
	public static int valid(int[] code, int validLength){
		int sum=0;
		for(int i=0;i<validLength;i++){
			sum+=code[i];
		}
		if(sum%2==0) return 1;
		else return 0;
		
	}
}
