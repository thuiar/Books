package boa;

import java.util.List;

import com.mathworks.toolbox.javabuilder.MWClassID;
import com.mathworks.toolbox.javabuilder.MWNumericArray;

import MDLSearch.MDLSearchclass;
import boa.func.FunctionFrame;
import boa.func.Trap;

public class BOA {
	private Population pop;
	private Bayesian bayesian;
	private List<int[]> group;
	
	
	public boolean boa(){
			
		FunctionFrame func = new Trap(5) ;
		//random
		//FunctionFrame func = new Onemax() ;
		
		pop =new Population(BOAParameter.POP_SIZE, BOAParameter.PROBLEM_SIZE);
		pop.evaluate(func);  
		int counter =0;
		boolean endFlag=false;
		while(counter<BOAParameter.MAX_GEN&&!endFlag){
			
			System.out.print("Iteration:"+counter+"  \t");
			
			
			Population parent = pop.truncationSelection();
			 
			constructTheNetwork(parent);
			Population offspring = generateOffsprings();	    
			
//			Population offspring = generateMDLOffspring(parent);
//			Population offspring = generateL1Offspring(parent);
			
//			Population offspring = generateOffspring(parent);	    
			
			//offspring =new Population(BOAParameter.POP_SIZE, BOAParameter.PROBLEM_SIZE);
			offspring.evaluate(func);
//			System.out.print("\t Pop ave Fit:" + pop.getAveFitness());
//			System.out.print("\t Parent ave Fit:" + parent.getAveFitness());
//			System.out.print("\t Offspring ave Fit:" + offspring.getAveFitness());
		    pop.replace(offspring);
		    
			
		    System.out.print("Max Fit"+pop.getMaxFitness()+"\t");
		    System.out.print("Min Fit"+pop.getMinFitness()+"\t");
		    System.out.print("Ave Fit"+pop.getAveFitness()+"\n");
		    
		    if( pop.getMaxFitness()-pop.getAveFitness()<1E-5){
		    	endFlag = true;
		    }   
		    counter++;
		}
		
		if(endFlag&&pop.getMaxFitness()==func.optimal(BOAParameter.PROBLEM_SIZE)){
			return true;
		}
		
		return false;
	}
	
	public void setGroup(List<int[]> g){
		group = g;
	}
	
	public void constructTheNetwork(Population parent){
		 bayesian=new Bayesian(parent);
		 if(group!=null){
			 bayesian.forbid(group);
			 
		 }
		 bayesian.constructTheNetwork();
	}
	
	public void constructTheNetwork_MDL(Population parent) {
		double[][] pn = parent.getPNCodes();
		double[][] cand = new double[pn.length][pn[0].length];
		for(int i=0;i<cand.length;i++){
			for(int j=0;j<pn[0].length;j++){
				cand[i][j] = 1;
			}
		}
		
		
		MWNumericArray data =  new MWNumericArray(pn,MWClassID.DOUBLE);
		MWNumericArray scope =  new MWNumericArray(cand,MWClassID.DOUBLE);
		try{
			
			MDLSearchclass MDL = new MDLSearchclass();
			Object[] result  = MDL.MDLSearch(1, data , scope);
			MWNumericArray a = (MWNumericArray) result[0]; 
            double[][] array= (double[][])a.toArray();
           // System.out.println("generate over");
            bayesian=new Bayesian(array, parent);
		}catch(Exception ex){
			ex.printStackTrace();
		}
	}
	
	
	public Population generateOffsprings(){
		return generateOffsprings((int)(BOAParameter.POP_SIZE*BOAParameter.OFFSPRING_PERCENT));
	}
	
	public Population generateOffsprings(int popsize){
		Population p = bayesian.generateNewInstances(popsize);
		return p;	
	}
	
	
}

