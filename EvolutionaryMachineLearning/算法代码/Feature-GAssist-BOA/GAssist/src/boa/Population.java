package boa;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import boa.func.FunctionFrame;


public class Population {
	private int size;
	private int length;
	private List<Chromosome> candidates;
	private boolean isSorted;
	
	
	public Population(){
		candidates =new ArrayList<Chromosome>();
		this.isSorted  =  false;	
	}
	
	//randomly initialize Chromosome
	public Population(int aSize, int aLength){
		this.size  = aSize;
		this.length = aLength;
		candidates =new ArrayList<Chromosome>();
		for(int i=0;i<size;i++){
			Chromosome chro =new Chromosome(length);
			candidates.add(chro);
		}
		this.isSorted  =  false;	
	}
	
	//set the size of the Population only
	public Population(int aSize){
		this.size  = aSize;
		candidates =new ArrayList<Chromosome>();
		this.isSorted  =  false;	
	}
	
	
	
	public void addIndividual(Chromosome chr){
		candidates.add(chr);
		size++;
		this.isSorted  =  false;	
	}
	
	
	//从小到大排列
	public void sort(){
		if(!isSorted){
			Collections.sort(candidates);
		}
	}
	
	/*
	public Population tournamentSelection(){
		
	}
	*/
	
	public Population truncationSelection(){
		this.sort();
		Population offspring =new Population((int)(BOAParameter.PARENT_PERCENT*size));
		
		for (int i=offspring.size-1, j = candidates.size()-1; i>=0; i--, j--){
			offspring.candidates.add(0,candidates.get(j));
		}
		offspring.isSorted =true;
		return offspring;
	}
	
	public Population wheelSelection(){
		long t1=System.currentTimeMillis();
		
		this.sort();
		Population offspring =new Population((int)(BOAParameter.PARENT_PERCENT*size));
		double[] seed =new double[offspring.size];
		double fitSum =0;
		for (int i=0; i<candidates.size(); i++){
			fitSum+=candidates.get(i).getFitness(); 
		}
		for (int i=0; i<offspring.size; i++){
			seed[i]=Math.random()*fitSum; 
		}
		Arrays.sort(seed);
		
		int index=0; 
        double sum=candidates.get(index).getFitness();
        for (int i=0; i<offspring.size; i++){
        	while(seed[i]>sum){
        		index++;
        		sum+=candidates.get(index).getFitness();
        	}
        	offspring.candidates.add(0,candidates.get(index));
        }
        
        long t2=System.currentTimeMillis();
        System.out.println(t2-t1);
		return offspring;
	}
	
	private int selectRW(){
		this.sort();
		double fitSum =0;
		double tempSum[] =new double[candidates.size()];
		for (int i=0; i<candidates.size(); i++){
			fitSum+=candidates.get(i).getFitness();
			tempSum[i] = fitSum; 
		}
		double choiceP=Math.random()*fitSum;
		double wheel=candidates.get(0).getFitness();
				
		int low=0,high=candidates.size()-1;
		
//		while(tempSum[low]<tempSum[high]){
//			if(choiceP<tempSum[(low+high)/2]){
//				high =  (low+high)/2-1;
//			}else if(choiceP>tempSum[(low+high)/2]){
//				low =  (low+high)/2+1;	
//			}else
//				return low;
//		}
		return high;
	}
	
	
	public void replace(Population offspring){
		this.sort();
		for (int i=0; i<offspring.size;i++){
			candidates.set(i, offspring.candidates.get(i));
		}
		this.isSorted  =  false;	
	}
	
	public void append(Population pop){
		for (int i=0; i<pop.size;i++){
			candidates.add(pop.candidates.get(i));
		}
		
	}
	
	public List<Chromosome> getChromosomes(){
		return candidates;
	}
	
	public int getSize(){
		size = candidates.size();
		return size;
	}
	
	public int getLength(){
		return candidates.get(0).getLength();
	}
	
	public void evaluate(FunctionFrame function){
		for(Chromosome chro:candidates){
			chro.evaluate(function);
		}	
	}
	
	//
	public double getMaxFitness(){
		this.sort();
		return candidates.get(size-1).getFitness();
	}
	
	public double getMinFitness(){
		this.sort();
		return  candidates.get(0).getFitness();
	}
	
	public double getAveFitness(){
		double sum=0;
		for(Chromosome chro:candidates){
			sum+=chro.getFitness();
		}
		return sum/size;
	}
	
	public void check(List<int[]> group){
		for(Chromosome chro:candidates){
			if(!chro.check(group)){
				candidates.remove(chro);
			}
		}
		
	}
	
	public double[][] getPNCodes(){
		double[][] pn = new double[candidates.size()][length];
		for(int i=0;i<candidates.size();i++){
			pn[i] = candidates.get(i).getPNCode();
		}
		return pn;
	}
	
	
}
