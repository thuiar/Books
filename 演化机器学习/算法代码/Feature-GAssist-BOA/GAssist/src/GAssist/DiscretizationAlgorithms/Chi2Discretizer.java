package GAssist.DiscretizationAlgorithms;

import java.util.Vector;

import GAssist.Dataset.Instance;
import GAssist.Dataset.InstanceSet;

public class Chi2Discretizer {

	protected Vector discretizeAttribute(int attribute,InstanceSet is) {
		Instance[] instances =is.getInstances(); 
		double[] realValues = new double[instances.length];
		int []points= new int[instances.length];
		int numPoints=0;
		for(int i=0;i<instances.length;i++){
			if(!instances[i].areAttributesMissing()[attribute]) {
				points[numPoints++]=i;
				realValues[i]=instances[i].getRealAttribute(attribute);
			}
		}
		//sortValues(attribute,points,0,numPoints-1);
		
		
		
		Vector cp=new Vector();
		return cp;
	}
	
	
	
}
