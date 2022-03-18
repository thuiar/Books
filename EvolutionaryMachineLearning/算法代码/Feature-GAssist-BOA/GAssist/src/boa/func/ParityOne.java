package boa.func;

public class ParityOne {
	public static int valid(int[] code, int lengthOfParity, int lengthOfOne){
		int[] oneCode = new int[lengthOfOne];
		for(int i=0;i<lengthOfOne;i++){
			int[] tempCode=new int[lengthOfParity];
			System.arraycopy(code, i*lengthOfParity, tempCode, 0, lengthOfParity);		
			oneCode[i]=HiddenParity.valid(tempCode, lengthOfParity); 
		}
		return CountOne.valid(oneCode);
	}
}
