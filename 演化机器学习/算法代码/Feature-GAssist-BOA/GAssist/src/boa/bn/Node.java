package boa.bn;

import java.util.ArrayList;
import java.util.List;

import boa.Chromosome;
import boa.Population;
import boa.util.MyUtil;

public class Node {
	private String name;
	private int index;
	private List<String> properties;
	private Net net;

	private List<Node> parents;
	private List<Node> children;
	private List<Integer> parnetIndex;

	private boolean flag;
	private int parentsNum;
	private int childrenNum;

	// private List<Column> columns;

	public Node(String aName) {
		name = aName;
		parents = new ArrayList<Node>();
		children = new ArrayList<Node>();
		parnetIndex = new ArrayList<Integer>();

	}

	public Node(int index) {
		this.index = index;
		name = Integer.toString(index);
		parents = new ArrayList<Node>();
		children = new ArrayList<Node>();
		parnetIndex = new ArrayList<Integer>();
	}

	public void clear() {
		flag = false;
		parentsNum = parents.size();
		childrenNum = children.size();
	}

	public void visit() {
		flag = true;
	}

	public boolean isVisit() {
		return flag;
	}

	public void removeAllEdge() {
		parents.clear();
		children.clear();
		parnetIndex.clear();
	}

	public void addChild(Node node) {
		children.add(node);
		node.parents.add(this);
		node.parnetIndex.add(this.index);
	}

	public boolean isChild(Node node) {
		return children.contains(node);
	}

	// K2 score
	public double computeLogGain(Population pop, Node candidatePa) {

		int numConf = 1 << parents.size();
		// int numConf2 = 1<<(parents.size()+1);
		int numConf2 = numConf * 2;

		double gain = 0;// logGain

		int[][][] count = countParent(pop, candidatePa.getIndex());

		int[] oldTotalCount = new int[numConf];
		int[] oldTotalPriorCount = new int[numConf];
		int[][] oldCount = new int[numConf][2];
		int[][] oldPriorCount = new int[numConf][2];

		int[] newTotalCount = new int[numConf2];
		int[] newTotalPriorCount = new int[numConf2];
		int[][] newCount = new int[numConf2][2];
		int[][] newPriorCount = new int[numConf2][2];

		// include the candidate parent
		for (int j = 0; j < (numConf); j++) {
			// newTotalCount[j]=0;
			// newTotalPriorCount[j]=0;

			int j2 = j << 1;

			for (int k = 0; k < 2; k++) {
				newCount[j2][k] = count[j][0][k];
				newCount[j2 + 1][k] = count[j][1][k];
				newPriorCount[j2][k] = 1;
				newPriorCount[j2 + 1][k] = 1;

				newTotalCount[j2] += newCount[j2][k];
				newTotalPriorCount[j2] += newPriorCount[j2][k];
				newTotalCount[j2 + 1] += newCount[j2 + 1][k];
				newTotalPriorCount[j2 + 1] += newPriorCount[j2 + 1][k];
			}

		}
		;
		for (int j = 0; j < numConf2; j++)// compute the K2 including the
											// candidate parent
		{
			gain -= MyUtil.logSumDifference(newTotalPriorCount[j],
					newTotalPriorCount[j] + newTotalCount[j]);
			for (int k = 0; k < 2; k++) {
				gain += MyUtil.logSumDifference(newPriorCount[j][k],
						newPriorCount[j][k] + newCount[j][k]);
			}
		}

		// exclude the candidate parent
		for (int j = 0; j < (1 << parents.size()); j++) {
			// oldTotalCount[j]=0;
			// oldTotalPriorCount[j]=0;

			int j2 = j << 1;

			for (int k = 0; k < 2; k++) {
				oldCount[j][k] = newCount[j2][k] + newCount[j2 + 1][k]; // add
																		// two
																		// candidate
				oldPriorCount[j][k] = 1;

				oldTotalCount[j] += oldCount[j][k];
				oldTotalPriorCount[j] += oldPriorCount[j][k];
			}
		}
		;
		for (int j = 0; j < numConf; j++)// compute the K2 excluding the
											// candidate parent
		{
			gain += MyUtil.logSumDifference(oldTotalPriorCount[j],
					oldTotalPriorCount[j] + oldTotalCount[j]);
			for (int k = 0; k < 2; k++) {
				gain -= MyUtil.logSumDifference(oldPriorCount[j][k],
						oldPriorCount[j][k] + oldCount[j][k]);
			}
		}

		return gain;
	}

	private int[][][] countParent(Population pop, int candidatePa) {

		int[][][] count = new int[1 << parents.size()][2][2];
		// count[][][]自左向右分别表示已知父节点编码 、待选父节点编码、 本节点编码

		for (Chromosome chro : pop.getChromosomes()) {
			// conf is the binary code of the parents
			int conf = MyUtil.indexedBinaryToInt(chro.getCode(), parnetIndex);
			count[conf][chro.getCode()[candidatePa] - '0'][chro.getCode()[index] - '0']++;
		}
		return count;
	}

	public void samping(Population pop, Population offspring) {
		int numConf = 1 << parents.size();
		int numConf2 = 1 << (parents.size() + 1);

		// computing the CPT
		double marginalAll[] = new double[numConf];
		double marginalAllButOne[] = new double[numConf];

		int[][] count = countParent(pop);
		for (int i = 0; i < numConf; i++) {
			marginalAll[i] = (double) (count[i][0] + count[i][1])/ pop.getSize();
			if(marginalAll[i]!=0){
				marginalAllButOne[i] = (double) count[i][1]/ (double) (count[i][0] + count[i][1]);
			}else{
				marginalAllButOne[i] =-1;
			}
			//System.out.print(index+":" + i+": "+ marginalAllButOne[i] + "   ");
		}
		//System.out.println();
		// Samping
		for (Chromosome chro : offspring.getChromosomes()) {

			int conf = MyUtil.indexedBinaryToInt(chro.getCode(), parnetIndex);
			if (marginalAllButOne[conf] == -1) {
				chro.getCode()[index] = '0';
			} else {
				if (Math.random()<marginalAllButOne[conf]){
					chro.getCode()[index] = '1';
				} else {
					chro.getCode()[index] = '0';
				}
			}

		}

	}

	private int[][] countParent(Population pop) {

		int[][] count = new int[1 << parents.size()][2];
		// count[][]自左向右分别表示已知父节点、 本节点编码

		for (Chromosome chro : pop.getChromosomes()) {
			// conf is the binary code of the parents
			int conf = MyUtil.indexedBinaryToInt(chro.getCode(), parnetIndex);
			count[conf][chro.getCode()[index] - '0']++;
		}
		return count;
	}

	public int getIndex() {
		return index;
	}

	public void addParenet(Node node) {
		parents.add(node);
		node.children.add(this);
		parnetIndex.add(node.index);
	}

	public boolean isParent(Node node) {
		return parents.contains(node);

	}

	public List<Node> getParent() {
		return parents;
	}

	public int getParentSize() {
		return parentsNum;
	}

	public List<Node> getChildren() {
		return children;
	}

	public void decreseParentSize() {
		parentsNum--;
	}

	public int getChildrenSize() {
		return childrenNum;
	}
	
	public List<Node> getPC(){
		List<Node> pc =new ArrayList<Node>();
		for(Node node:parents)
			pc.add(node);
		for(Node node:children)
			pc.add(node);
		return pc;	
	}
	
	
	
	public List<Node> getMB(){
		List<Node> mb =new ArrayList<Node>();
		for(Node node:parents)
			mb.add(node);
		for(Node node:children)
			mb.add(node);
		for(Node node:children)
			for(Node sporse: node.parents)
				if(this!=sporse){
					mb.add(sporse);			
				}
		return mb;
		
	}

}
