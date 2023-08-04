package boa.func;

public class CountOne {
	public static int valid(int[] code){
		int sum = 0;
		for(int i=0;i<code.length;i++){
			sum+=code[i];
		}
		if(sum > code.length/2)		return 1;
		return 0;
	}
}
