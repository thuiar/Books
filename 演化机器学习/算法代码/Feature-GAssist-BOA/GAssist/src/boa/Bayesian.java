package boa;

import java.util.ArrayList;
import java.util.List;

import GAssist.Globals_GABIL;
import GAssist.Parameters;
import boa.bn.Net;
import boa.bn.Node;
import boa.util.Parameter;

public class Bayesian {
	private Net net;
	private Population pop;
	double gain[][];
	boolean[] full;
	
	
//	public Bayesian(int nodeNum, Population p){
//		pop = p;
//		net = new Net(nodeNum);
//		gain =new double[nodeNum][nodeNum];
//		
//		//initialize the gain
//		for(int i=0;i<net.getSize();i++) {
//			for(int j=0;j<net.getSize();j++){
//				if(i!=j){
//					gain[i][j]=Parameter.CAN_CONNECTED;
//				}else{
//					gain[i][j]=Parameter.CANNOT_CONNECTED;
//				}
//			}				
//		}
//		full = new boolean[net.getSize()];
//	}
	
	public Bayesian(double[][] matrix, Population p) {
		pop = p;
		net = new Net(matrix.length);
		net.addEdges(matrix);
	}
	
	public Bayesian(Population p){
		int nodeNum = p.getLength();
		pop = p;
		net = new Net(nodeNum);
		gain =new double[nodeNum][nodeNum];
		
		//initialize the gain
		for(int i=0;i<net.getSize();i++) {
			for(int j=0;j<net.getSize();j++){
				if(i!=j){
					gain[i][j]=Parameter.CAN_CONNECTED;
				}else{
					gain[i][j]=Parameter.CANNOT_CONNECTED;
				}
			}				
		}
		full = new boolean[net.getSize()];
	}
	
	//arrays每个集合中的任意两个下标指定的节点之间不能有连接，所以将gain设置为CANNOT_CONNECTED;
	public void forbid(List<int[]> arrays) {
		for (int[] a : arrays) {

			for (int i = 0; i < a.length; i++)
				for (int j = i + 1; j < a.length; j++) {
					gain[a[i]][a[j]] = Parameter.CANNOT_CONNECTED;
					gain[a[j]][a[i]] = Parameter.CANNOT_CONNECTED;
				}
		}

	}
		
	
	
	
	//constructTheNetwork(parents,G,boaParams);
	//generateNewInstances(parents,offspring,G,boaParams);
	
	
	public void constructTheNetwork() {
		net.removeAllEdges();
		// compute the gain through every node

		for (Node node : net.getNodeList()) {
			computeCandidaPa(node);
		}
		// add links
		double maxGain;
		int maxFrom, maxTo;
		int maxNumberOfEdges = net.getSize() * BOAParameter.MAX_NODE_INCOMING;
		boolean finito = false;

		for (int numAdded = 0; (numAdded < maxNumberOfEdges) && (!finito); numAdded++) {
			maxGain = -1;
			maxFrom = -1;
			maxTo = -1;

			for (int i = 0; i < net.getSize(); i++) {
				for (int j = 0; j < net.getSize(); j++) {
					if (i != j) {
						// what about an edge from k to l?
						if (gain[i][j] > maxGain) {
							maxFrom = i;
							maxTo = j;
							maxGain = gain[i][j];
						}
					}
				}
			}

			if (maxGain > 0) {
				Node nodeFrom = net.getNodeList().get(maxFrom);
				Node nodeTo = net.getNodeList().get(maxTo);
				
				// add the new edge
				net.addEdge(nodeFrom, nodeTo, gain);
				//System.out.print(maxFrom + "-->" + maxTo + " ");
				
				// the edge can't be added anymore
				gain[maxFrom][maxTo] = gain[maxTo][maxFrom] = Parameter.CANNOT_CONNECTED;
				// edges that would create cycles with the new edge have to be
				// disabled
				net.checkCycles(nodeFrom, nodeTo, gain);

				// is the ending node of the new edge full yet?
				full[maxTo] = (nodeTo.getParent().size() >= BOAParameter.MAX_NODE_INCOMING);
				// recompute the gains that are needed

				if (full[maxTo]) {
					for (int i = 0; i < net.getSize(); i++) {
						gain[i][maxTo] = Parameter.CANNOT_CONNECTED;
					}
				} else {
					computeCandidaPa(nodeTo);
				}

				// for(Node node : net.getNodeList()){
				// computeCandidaPa(node);
				// }

			} else {
				finito = true;
			}

		}
		// for(Node node:net.getNodeList()){
		// System.out.print(node.getIndex()+"<-:");
		// for(Node parent: node.getParent()){
		//				
		// System.out.print(parent.getIndex()+"\t");
		// }
		// System.out.println();
		// }

	}
	
	private void computeCandidaPa(Node node){
		for(Node anotherNode : net.getNodeList()){
			if(!anotherNode.equals(node)&&net.canAddEdge(anotherNode, node)
					&&gain[anotherNode.getIndex()][node.getIndex()]!=Parameter.CANNOT_CONNECTED){
				gain[anotherNode.getIndex()][node.getIndex()]=
					node.computeLogGain(pop, anotherNode);
			}else{
				gain[anotherNode.getIndex()][node.getIndex()]=Parameter.CANNOT_CONNECTED;
			}
		}
		
	}
	
	
	public Population generateNewInstances(int popsize){
		Population offspring =new Population(popsize,net.getSize()); 
		// topology sort first
		
		List<Node> top = net.topology();
		//net.printEdge();
		// sampling according to the topology
		if(top.size()!=net.getSize()){
			System.out.println("Net Construction Error "+ top.size()+ " "+net.getSize());
		}
		for(Node node: top){
			node.samping(pop, offspring);
		}
		return offspring;
	}
	
	public List<ArrayList<Integer>> printPC(){
		List<ArrayList<Integer>> array=new ArrayList<ArrayList<Integer>>();
		for(int i=0;i<Parameters.numAttributes;i++){
			array.add(new ArrayList<Integer>());
		}
		for(Node node:net.getNodeList()){
			//System.out.print(node.getIndex()+":");
			for(Node pc:node.getPC()){
			//	System.out.print(pc.getIndex()+",");
			}
			//System.out.println("----------------------------------");
			
			int attr0 = Globals_GABIL.indexOfAttribute(node.getIndex());
			System.out.print(attr0+":");
			for(Node pc:node.getPC()){
				int attr = Globals_GABIL.indexOfAttribute(pc.getIndex());
				System.out.print(attr+",");
				ArrayList<Integer> list = array.get(attr0);
				if(!list.contains(attr)){
					list.add(attr);
				}
			}
			System.out.println();
		}
		return array;
	}
}
