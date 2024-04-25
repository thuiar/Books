package lns;

import hs.HarmonySearch;
import comet.*;
import common.ShopData;


public class LargeNeighborhoodSearch {
       
	CometSystem sys;
	CometOptions o;
    
    public LargeNeighborhoodSearch(int perm[], int fixedMachineSel[], int N){
    	int[] machineSelInfo = new int [N / 2 ];
    	int i;
    	int[] temp = new int[ShopData.getMachineNum()];
    	int[] len = new int[ShopData.getMachineNum()];
    	for (i = 0; i < N / 2; i++){
    		machineSelInfo[i] = ShopData.getMachineIdByChoice(i + 1, perm[i]);
    		temp[machineSelInfo[i] - 1]++;
    	}
    	
    	int[][] sequenceInfo = new int[ShopData.getMachineNum()][];
    	for (i = 0; i < ShopData.getMachineNum(); i++)
    		sequenceInfo[i] = new int[temp[i]];

    	for (i = N / 2; i < N; i++){
    		int opId = perm[i];
    		int mId = ShopData.getMachineIdByChoice(opId, perm[opId - 1]);
    		sequenceInfo[mId - 1][len[mId - 1]] = opId;
    		len[mId - 1]++;
    	}

		sys = new CometSystem();
		o = new CometOptions();
		o.setFilename("LNSBlock.co");
		sys.setOptions(o);
		
    	int nMachines = ShopData.getMachineNum();
    	int nActivities = ShopData.getTotalOpNum();
    	int[][] duration = ShopData.getTimeTable();

    	int[] nOpInJob = new int[ShopData.getJobNum()];
    	for (i = 0; i < ShopData.getJobNum(); i++)
    		nOpInJob[i] = ShopData.getOpNumInJob(i + 1);
    	sys.addInput("nActivities", nActivities);
    	sys.addInput("nMachines", nMachines);

     	sys.addInput("duration", duration);
    	sys.addInput("nOpInJob", nOpInJob);
    	sys.addInput("machineSelInfo",machineSelInfo);
    	sys.addInput("sequenceInfo", sequenceInfo);
    	sys.addInput("fixedMachineSel", fixedMachineSel);
    }
    
    public void run(){
		try {
			sys.solve();
			int makeSpan = sys.getIntOutput("makeSpan");
			System.out.println(makeSpan);

		} catch (CometException e) {
			System.out.println("In Java: caught:" + e);
		}
    }

	public static void main(String[] args) {
		
		
		new ShopData("E:\\Data\\DPdata\\05a.fjs");
		int[][] perm = new int[4][];
		for (int i = 0; i < 4; i++){
			HarmonySearch hs = new HarmonySearch();
			System.out.println(hs.run());
		    perm[i] = hs.getBestPerm();
		}
		int[] fixedMachineSel = new int[ShopData.getTotalOpNum()];
		for (int j = 0; j <  ShopData.getTotalOpNum(); j++){
			if (perm[0][j] == perm[1][j] && perm[1][j] == perm[2][j] && perm[2][j] == perm[3][j]){
				fixedMachineSel[j] = ShopData.getMachineIdByChoice(j + 1, perm[0][j]);
			}
			else
				fixedMachineSel[j] = -1;
		}
		new LargeNeighborhoodSearch(perm[0], fixedMachineSel, ShopData.getTotalOpNum() * 2).run();
		
	}

    

}