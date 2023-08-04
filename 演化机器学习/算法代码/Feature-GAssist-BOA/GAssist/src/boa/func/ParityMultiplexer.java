package boa.func;

public class ParityMultiplexer {
	public static int valid(int[] code, int lengthOfParity, int lengthOfAddr){
		int length = (int)Math.pow(2, lengthOfAddr) + lengthOfAddr;
		int[] MPCode = new int[length];
		for(int i=0;i<length;i++){
			
			int[] tempCode=new int[lengthOfParity];
			System.arraycopy(code, i*lengthOfParity, tempCode, 0, lengthOfParity);		
			MPCode[i]=HiddenParity.valid(tempCode, lengthOfParity); 
		}
		return Multiplexer.valid(MPCode, lengthOfAddr);
	}
}
