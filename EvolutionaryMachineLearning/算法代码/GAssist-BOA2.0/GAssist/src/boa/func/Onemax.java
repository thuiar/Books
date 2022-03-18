package boa.func;


public class Onemax extends FunctionFrame {
	
	public double compute(char[] code){
		double fitness = 0;
		for(int i=0;i<code.length ;i++)
			fitness +=code[i]-'0';
		return fitness;
	}
	
	public boolean isOptimal(char[] code){
		for(int i=0;i<code.length ;i++)
			if(code[i]=='0') return false;
		return true;
				
	}
	
	public double optimal(int size){
		return (double)size;
	}

}
