package boa.func;



public abstract class FunctionFrame {
	
	public abstract double optimal(int size);
	public abstract double compute (char[] code);
	public abstract boolean isOptimal(char[] code);

}
