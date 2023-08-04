/*
 * ParserParameters.java
 *
 */

package GAssist;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.StringTokenizer;

public class ParserParameters {
	static BufferedReader br;
	
	/** Creates a new instance of ParserParameters */
	public static void doParse(String fileName) {
		try {
			br=new BufferedReader(new FileReader(fileName));
		} catch(Exception e) {
			e.printStackTrace();
			System.exit(1);
		}

		parseParameters();	
	}
	
	static void parseParameters() {
		String str=getLine();
		while(str!= null) {
			StringTokenizer st = new StringTokenizer(str," = ");
			String name = st.nextToken();
			name=name.trim();
			name.replaceAll(" ","");

			processParameters(st,name);
			str=getLine();
		}
	}

	static void processParameters(StringTokenizer st,String paramName) {
		if(!st.hasMoreTokens()) {
			System.err.println("Parse error processing parameter "+paramName);
			System.exit(1);
		}
		String paramValue=st.nextToken();
		paramValue=paramValue.trim();

		if(isReal(paramName)) 
			insertRealParameter(paramName,paramValue);
		else if(isInteger(paramName))
			insertIntegerParameter(paramName,paramValue);
		else if(isBoolean(paramName))
			insertBooleanParameter(paramName,paramValue);
		else if(isString(paramName))
			insertStringParameter(paramName,paramValue);
		else {
			System.err.println("Unknown parameter "+paramName);
			System.exit(1);
		}
	}

	static boolean isReal(String paramName) {
		if(paramName.equalsIgnoreCase("hierarchicalSelectionThreshold")) return true;
		if(paramName.equalsIgnoreCase("probCrossover")) return true;
		if(paramName.equalsIgnoreCase("probMutationInd")) return true;
		if(paramName.equalsIgnoreCase("probOne")) return true;
		if(paramName.equalsIgnoreCase("probSplit")) return true;
		if(paramName.equalsIgnoreCase("probMerge")) return true;
		if(paramName.equalsIgnoreCase("probReinitializeBegin")) return true;
		if(paramName.equalsIgnoreCase("probReinitializeEnd")) return true;
		if(paramName.equalsIgnoreCase("initialTheoryLengthRatio")) return true;
		if(paramName.equalsIgnoreCase("weightRelaxFactor")) return true;
		if(paramName.equalsIgnoreCase("attributesThrehold")) return true;
		if(paramName.equalsIgnoreCase("attributeThrehold")) return true;
		return false;
	}

	static boolean isInteger(String paramName) {
		if(paramName.equalsIgnoreCase("iterationRuleDeletion")) return true;
		if(paramName.equalsIgnoreCase("iterationHierarchicalSelection")) return true;
		if(paramName.equalsIgnoreCase("ruleDeletionMinRules")) return true;
		if(paramName.equalsIgnoreCase("sizePenaltyMinRules")) return true;
		if(paramName.equalsIgnoreCase("numIterations")) return true;
		if(paramName.equalsIgnoreCase("seed")) return true;
		if(paramName.equalsIgnoreCase("initialNumberOfRules")) return true;
		if(paramName.equalsIgnoreCase("popSize")) return true;
		if(paramName.equalsIgnoreCase("tournamentSize")) return true;
		if(paramName.equalsIgnoreCase("numStrata")) return true;
		if(paramName.equalsIgnoreCase("maxIntervals")) return true;
		if(paramName.equalsIgnoreCase("iterationMDL")) return true;
		if(paramName.equalsIgnoreCase("MAX_NODE_INCOMING")) return true;
		if(paramName.equalsIgnoreCase("BOA_INTERVAL")) return true;
		if(paramName.equalsIgnoreCase("REDUCE_INTERVAL")) return true;
		if(paramName.equalsIgnoreCase("PRINT_INTERVAL")) return true;
		return false;
	}

	static boolean isBoolean(String paramName) {
		if(paramName.equalsIgnoreCase("adiKR")) return true;
		if(paramName.equalsIgnoreCase("useMDL")) return true;
		return false;
	}

	static boolean isString(String paramName) {
		if(paramName.equalsIgnoreCase("discretizer1")) return true;
		if(paramName.equalsIgnoreCase("discretizer2")) return true;
		if(paramName.equalsIgnoreCase("discretizer3")) return true;
		if(paramName.equalsIgnoreCase("discretizer4")) return true;
		if(paramName.equalsIgnoreCase("discretizer5")) return true;
		if(paramName.equalsIgnoreCase("discretizer6")) return true;
		if(paramName.equalsIgnoreCase("discretizer7")) return true;
		if(paramName.equalsIgnoreCase("discretizer8")) return true;
		if(paramName.equalsIgnoreCase("discretizer9")) return true;
		if(paramName.equalsIgnoreCase("discretizer10")) return true;
		if(paramName.equalsIgnoreCase("defaultClass")) return true;
		if(paramName.equalsIgnoreCase("initMethod")) return true;
		if(paramName.equalsIgnoreCase("CHI_SQUARE")) return true;

		return false;
	}


	static void insertRealParameter(String paramName,String paramValue) {
		LogManager.println("Setting parameter "+paramName+" to value "+paramValue);
		double num=Double.parseDouble(paramValue);
		try {
			Parameters param=new Parameters();
			java.lang.reflect.Field f= Parameters.class.getField(paramName);
			f.setDouble(param,num);
		} catch(Exception e) {
			System.err.println("Cannot set param "+paramName);
			System.exit(1);
		}
	}

	static void insertIntegerParameter(String paramName,String paramValue) {
		LogManager.println("Setting parameter "+paramName+" to value "+paramValue);
		int num=Integer.parseInt(paramValue);
		try {
			Parameters param=new Parameters();
			java.lang.reflect.Field f= Parameters.class.getField(paramName);
			f.setInt(param,num);
		} catch(Exception e) {
			System.err.println("Cannot set param "+paramName);
			System.exit(1);
		}
	}

	static void insertBooleanParameter(String paramName,String paramValue) {
		LogManager.println("Setting parameter "+paramName+" to value "+paramValue);
		boolean val=false;
		if(paramValue.equalsIgnoreCase("true")) val=true;

		try {
			Parameters param=new Parameters();
			java.lang.reflect.Field f= Parameters.class.getField(paramName);
			f.setBoolean(param,val);
		} catch(Exception e) {
			System.err.println("Cannot set param "+paramName);
			System.exit(1);
		}
	}


	static void insertStringParameter(String paramName,String paramValue) {
		LogManager.println("Setting parameter "+paramName+" to value "+paramValue);
		try {
			Parameters param=new Parameters();
			java.lang.reflect.Field f= Parameters.class.getField(paramName);
			f.set(param,new String(paramValue));
		} catch(Exception e) {
			System.err.println("Cannot set param "+paramName);
			System.exit(1);
		}
	}



	static String getLine() {
		String st=null;
		do {
			try {
				st=br.readLine();
			} catch(Exception e) {
				e.printStackTrace();
				System.exit(1);
			}
		} while(st!=null && st.equalsIgnoreCase(""));
		return st;
	}
}
