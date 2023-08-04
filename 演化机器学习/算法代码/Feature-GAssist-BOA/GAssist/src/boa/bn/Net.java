package boa.bn;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

import GAssist.LogManager;
import boa.util.Parameter;

public class Net {
	
	private List<Node> nodeList;
	private int size;
	private int[][] path; 
	private int[][] coincidence;
	
//	private final static int NOT_CONNECTED = 0;
//	private final static int CONNECTED = 1;
	
	
	public Net(int aSize){
		this.size  = aSize;
		nodeList=new ArrayList<Node>();
		for(int i=0;i<size;i++) {
			Node aNode =new Node(i);
			nodeList.add(aNode);
		}
		path = new int[size][size];
		coincidence = new int[size][size];
		for(int i=0;i<size;i++) {
			path[i][i] = Parameter.CONNECTED;
			coincidence[i][i] = Parameter.CONNECTED;
		}
		
	}
	
	public void removeAllEdges(){
		
		for(Node node:nodeList){
			node.removeAllEdge();
		}
		
		for(int i=0;i<size;i++) {
			for(int j=0;j<size;j++){
				if(i!=j){
					path[i][j]= Parameter.NOT_CONNECTED;
					coincidence[i][j] = Parameter.NOT_CONNECTED;
				}
			}
		}
	}
	
	public void addEdges(double[][] matrix){
		for(int i=0;i<matrix.length;i++)
			for(int j=0;j<matrix.length;j++){
				if(matrix[i][j]==1){
					nodeList.get(i).addChild(nodeList.get(j));
				}
				
			}
	}
	
	
	public void addEdge(Node parent, Node child, double[][] gain){
		int i= nodeList.indexOf(parent);
		int j= nodeList.indexOf(child);
		
		if(parent.isChild(child)){
			return;
		}
		
		parent.addChild(child);
		
		coincidence[i][j] = Parameter.CONNECTED;
		path[i][j] = Parameter.CONNECTED;
		
		//update matrix
		
//		for(int k=0; k<size; k++){
//			if (path[k][i]==Parameter.CONNECTED){
//				path[k][j]=Parameter.CONNECTED;
//			}
//		}
//		for(int k=0; k<size; k++){
//			if (path[j][k]==Parameter.CONNECTED){
//				path[i][k]=Parameter.CONNECTED;
//			}
//		}
		
		for(int k=0; k<size; k++)
			  if (path[k][i]==Parameter.CONNECTED)
			     for (int l=0; l<size; l++)
//			         if ((l!=k)&&(path[j][l]==Parameter.CONNECTED))
			         if (path[j][l]==Parameter.CONNECTED){
			            path[k][l]=Parameter.CONNECTED;
			            gain[l][k]=Parameter.CANNOT_CONNECTED;
			            
			         }
		
	}
	
	public boolean canAddEdge(Node fromNode,Node toNode){
		if(fromNode.isChild(toNode))
			return false;
		if(fromNode.getIndex()==8 && toNode.getIndex()==9){
			System.out.print("");
		}
		if(coincidence[fromNode.getIndex()][toNode.getIndex()]==Parameter.CONNECTED
				||path[toNode.getIndex()][fromNode.getIndex()]==Parameter.CONNECTED)
			return false;
		
		return true;
		
		
	}
	
	
	public void checkCycles(Node fromNode,Node toNode, double gain[][]){
		int newFrom= nodeList.indexOf(fromNode);
		int newTo= nodeList.indexOf(toNode);
		
		for(int k=0; k<size; k++){                 	
		    for(int l=0; l<size; l++){
			// does the new edge forbid creating an edge k,l by means of a path that might create a cycle with this?
			if (gain[k][l]>Parameter.CANNOT_CONNECTED&&
					path[l][newFrom]==Parameter.CONNECTED&&path[newTo][k]==Parameter.CONNECTED)
				gain[k][l]=Parameter.CANNOT_CONNECTED ;
			
			// does the new edge forbid creating an edge l,k by means of a path that might create a cycle with this?
			if (gain[l][k]>Parameter.CANNOT_CONNECTED &&
					path[k][newFrom]==Parameter.CONNECTED&&path[newTo][l]==Parameter.CONNECTED)
				gain[l][k]=Parameter.CANNOT_CONNECTED ;
		    }
		}
		
	}
	
	
	public int getSize(){
		return size;
	}
	
	public List<Node> getNodeList(){
		return nodeList;
	}
	
	public List<Node> topology(){
		for(Node node : nodeList){
			node.clear();
		}
		List<Node> topologyNodeList=new ArrayList<Node>();
		
		
		Queue<Node> queue =new LinkedList<Node>();
		for(Node node : nodeList){
			if(node.getParentSize()==0){
				queue.offer(node);
			}
		}
		if(queue.isEmpty()){
			System.out.println("Cycle Error");
		}
		while(!queue.isEmpty()){
			Node parent = queue.poll();
			topologyNodeList.add(parent);
			for(Node node : nodeList){
				if(!node.isVisit()&&node.getParent().contains(parent)){
					node.decreseParentSize();
					if(node.getParentSize()==0){
						queue.offer(node);
						node.visit();
					}	
				}
				
			}
		}
		return topologyNodeList;
	}
	
	
	public void printEdge(){
		int counter=0;
		for(Node node:nodeList){
			//LogManager.println_file("\tChildren of "+node.getIndex()+":");
			//LogManager.print_file("\t");
			for(Node child:node.getChildren()){
				//LogManager.print_file(child.getIndex()+" ");
				counter++;
			}
			//LogManager.println_file(" ");
		}
		LogManager.println_file(Integer.toString(counter));
		
	}
}
