package boa.func;

public class Deceptive  extends FunctionFrame {
	private int length;  
	
	public Deceptive(int l){
		length = l;
	}
	
	public double compute(char[] code){
		double fitness = 0 ,unit = 0;
		for(int i=0;i<code.length ;i+=length){
			unit = 0;
			for(int j=0;j<length;j++){
				unit +=code[i+j]-'0';
			}
			if(unit == length){
				fitness+=1;
			}else{
				fitness+=1-0.1*(1+unit);
			}
		}
		return fitness;
	}
	
	public boolean isOptimal(char[] code){
		for(int i=0;i<code.length ;i++)
			if(code[i]=='0') return false;
		return true;
				
	}
	
	public double optimal(int size){
		return (double)(size/length);
	}
}

