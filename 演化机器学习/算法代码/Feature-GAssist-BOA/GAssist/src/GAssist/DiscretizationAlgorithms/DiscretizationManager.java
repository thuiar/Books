package GAssist.DiscretizationAlgorithms;

import java.util.StringTokenizer;
import java.util.Vector;

import GAssist.LogManager;
import GAssist.Parameters;
import GAssist.PopulationWrapper;

public class DiscretizationManager {
	static Vector discretizers;

	public static void init() {
		// It's ugly, I know
		discretizers = new Vector();
		addDiscretizer(Parameters.discretizer1);
		addDiscretizer(Parameters.discretizer2);
		addDiscretizer(Parameters.discretizer3);
		addDiscretizer(Parameters.discretizer4);
		addDiscretizer(Parameters.discretizer5);
		addDiscretizer(Parameters.discretizer6);
		addDiscretizer(Parameters.discretizer7);
		addDiscretizer(Parameters.discretizer8);
		addDiscretizer(Parameters.discretizer9);
		addDiscretizer(Parameters.discretizer10);
	}

	public static void addDiscretizer(String name) {
		StringTokenizer st = new StringTokenizer(name,"_");
		String discretizerName=st.nextToken();
		Discretizer disc=null;

		if(discretizerName.equalsIgnoreCase("Greedy")){
			disc=new GreedyDiscretizer();
		}
		else if(discretizerName.equalsIgnoreCase("UniformWidth")){
			if(!st.hasMoreElements()) {
				LogManager.printErr("Error in discretizer "+name+". It should have a parameter");
				System.exit(1);
			} 
			int numIntervals=Integer.parseInt(st.nextToken());
			disc=new UniformWidthDiscretizer(numIntervals);
		} else if(discretizerName.equalsIgnoreCase("UniformFrequency")){
			if(!st.hasMoreElements()) {
				LogManager.printErr("Error in discretizer "+name+". It should have a parameter");
				System.exit(1);
			} 
			int numIntervals=Integer.parseInt(st.nextToken());
			disc=new UniformFrequencyDiscretizer(numIntervals);
		} else if(discretizerName.equalsIgnoreCase("ChiMerge")){
			if(!st.hasMoreElements()) {
				LogManager.printErr("Error in discretizer "+name+". It should have a parameter");
				System.exit(1);
			} 
			double confidence=Double.parseDouble(st.nextToken());
			disc=new ChiMergeDiscretizer(confidence);
		} else if(discretizerName.equalsIgnoreCase("ID3")){
			disc=new Id3Discretizer();
		} else if(discretizerName.equalsIgnoreCase("Fayyad")){
			disc=new FayyadDiscretizer();
		} else if(discretizerName.equalsIgnoreCase("Random")){
			disc=new RandomDiscretizer();
		} else if(discretizerName.equalsIgnoreCase("USD")){
			disc=new USDDiscretizer();
		} else if(discretizerName.equalsIgnoreCase("Disabled")){
		} else {
			LogManager.printErr("Unknown discretizer "+name);
			System.exit(1);
		}

		if(disc != null) {
			disc.buildCutPoints(PopulationWrapper.is);
			double[][] cps = disc.cutPoints;
//			for(int i=0;i<cps.length;i++){
//				MyUtil.printlDoubleArray(cps[i]);
//			}
//			if(cps[0]!=null)
//				disc.mergeIntervals(PopulationWrapper.is);
//			discretizers.addElement(disc);
			
			
			for(int i=0;i<cps.length;i++){
				if(cps[i]!=null){
					disc.mergeIntervals(i,PopulationWrapper.is);
				}
			}
			discretizers.addElement(disc);
			
		}
	}

	public static int getNumDiscretizers() {
		return discretizers.size();
	}

	public static Discretizer getDiscretizer(int num) {
		return (Discretizer)discretizers.elementAt(num);
	}
} 
