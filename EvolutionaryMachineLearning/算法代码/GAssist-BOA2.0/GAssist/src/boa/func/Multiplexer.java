package boa.func;

public class Multiplexer {
	public static int valid(int[] code, int lengthOfAddr){
		int value = 0;
		for(int i=0;i<lengthOfAddr;i++){
			value = (value*2)+code[i];
		}
		if(code[lengthOfAddr+value]!=1)		return 0;
		return 1;
	}
}
