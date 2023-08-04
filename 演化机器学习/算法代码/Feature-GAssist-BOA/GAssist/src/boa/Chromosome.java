package boa;

import java.util.List;

import GAssist.Globals_GABIL;
import GAssist.InstanceWrapper;
import GAssist.Parameters;
import boa.func.FunctionFrame;

public class Chromosome  implements Comparable {
	private char[] code;
	private double fitness;
	
	
	//Random  Chromosome
	public Chromosome(int length){
		code =new char[length];
		 for (int j=0; j<length; j++){
		    if (Math.random()<0.5){
		    	code[j]='0';
		    }else{
		    	code[j]='1';
		    }
		 }	
	}

	//Construction for the single rule in the GAssist
	public Chromosome(int []crm, int start, int length, double fit){
		code =new char[length-1];
		for (int j=0; j<length-1; j++){
		    if (crm[start+j]==0){
		    	code[j]='0';
		    }else{
		    	code[j]='1';
		    }
		 }	
		fitness = fit;
	}
	
	public Chromosome(int []crm){
		code =new char[crm.length];
		for (int j=0; j<crm.length; j++){
		    if (crm[j]==0){
		    	code[j]='0';
		    }else{
		    	code[j]='1';
		    }
		}
	}
	
	public Chromosome(InstanceWrapper is){
		code = new char[Globals_GABIL.ruleSize];
		for(int i=0;i<code.length;i++){
			code[i]='0';
		}
		for(int i=0;i<Parameters.numAttributes;i++){
			code[Globals_GABIL.offset[i]+is.getNominalValue(i)]='1';
		}
		code[Globals_GABIL.ruleSize-1] = (char)('0' + is.classOfInstance());
		
	}
	
	
	public Chromosome copy(){
		Chromosome chro =new Chromosome(this.code.length);
		chro.code = this.code;
		chro.fitness =this.fitness;
		return chro;
	}
	
	public void evaluate(FunctionFrame function){
		fitness = function.compute(code);
	}
	
	
	
	
	public double getFitness(FunctionFrame function){
		fitness = function.compute(code);
		return fitness;
	}
	
	
	public int compareTo(Object o){
		Chromosome other = (Chromosome) o;
		if(fitness>other.fitness) return 1;
		else if(fitness<other.fitness) return -1;
		else return 0;
	}
	
	public char[] getCode(){
		return code;
	}
	
	public double getFitness(){
		return fitness;
	}
	
	public int getLength(){
		return code.length;
	}
	
	public boolean equal(Object o){
		Chromosome c =(Chromosome) o;
		if(c.code.length !=  this.code.length)
			return false;
		for(int i=0;i<this.code.length;i++){
			if(this.code[i]!=c.code[i]){
				return false;
			}
		}
		return true;
	}
	
	
	public boolean check(List<int[]> group){
		for(int[] g:group){
			boolean flag =false;
			for(int i=0;i<g.length;i++){
				if(code[i]=='1'){
					flag = true;
				}
			}
			if(!flag)
				return false;
				
		}
		return true;
	}
	
	public double[] getPNCode(){
		double[] pn = new double[code.length];
		for(int i=0; i<pn.length ;i++){
			if(code[i]=='0'){
				pn[i] = -1;
			}else{
				pn[i] = 1;
			}
		}
		return pn;
	}
}