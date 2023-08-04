package relief;

import GAssist.Parameters;
import GAssist.Dataset.Instance;
import GAssist.Dataset.InstanceSet;

public class Relief {
	
	private InstanceSet is;
	private final static double iteration = 1000;
	private final static int HIT = 0;
	private final static int MISS = 1;
	
	
	public Relief(InstanceSet i){
		this.is = i;
	}
	
	public double computeWeight(int index){
		double w=0;
		for(int i=0;i<iteration;i++){
			int rand = (int)(Math.random()*is.numInstances());
			Instance r = is.getInstance(rand);
			/* find the closest neighour */
			Instance ns[] = neighbours(r);
			w=w - r.distance(ns[HIT], index)/iteration + r.distance(ns[MISS], index)/iteration;
		}
		return w; 
	}
	
	public void score(){
		double weights[] = new double[Parameters.numAttributes];
		for(int i=0;i<weights.length ;i++){
			weights[i] = computeWeight(i);
			System.out.println(i+":\t"+weights[i]);
		}
		
	}
	
	
	
	public Instance[] neighbours(Instance instance){
		Instance[] ns=new Instance[2];
		double dis0 = Double.MAX_VALUE;
		double dis1 = Double.MAX_VALUE;
		for(int i=0;i<is.numInstances();i++){
			Instance currentIstance = is.getInstance(i);
			double distance = instance.distance(currentIstance);
			if((currentIstance.getInstanceClass()==instance.getInstanceClass())){
				if(distance!=0&&distance<dis0){
					ns[HIT] = currentIstance;
					dis0 = distance ;
				}
			}else{
				if(distance<dis1){
					ns[MISS] = currentIstance;
					dis1 = distance ;
				}
			}
		}
		//System.out.println(dis0+"\t"+dis1);
		return ns;
	}
}
