/*
 * Timers.java
 *
 */

package GAssist;

/**
 * Class that manages timers: flags and parameters that are triggered at 
 * certain iterations
 */
public class Timers {

	public static boolean runTimers(int iteration,Classifier []population) {
		Globals_ADI.nextIteration();
		boolean res1=Globals_MDL.newIteration(iteration,population);
		boolean res2=timerBloat(iteration);

		if(res1 || res2) return true;
		return false;
	}

	public static void runOutputTimers(int iteration
			,Classifier []population) {
		Globals_DefaultC.checkNichingStatus(iteration,population);
	}

	
	static boolean timerBloat(int _iteration) {
		Parameters.doRuleDeletion=(_iteration>=Parameters.iterationRuleDeletion);
		//Parameters.doRuleDeletion=(_iteration%100>=Parameters.iterationRuleDeletion);
		
		Parameters.doHierarchicalSelection=(_iteration>=Parameters.iterationHierarchicalSelection);

		if(_iteration==Parameters.numIterations-1) {
			Parameters.ruleDeletionMinRules=1;
			return true;
		}

		if(_iteration==Parameters.iterationRuleDeletion)
			return true;
		
		return false;
	}
}
