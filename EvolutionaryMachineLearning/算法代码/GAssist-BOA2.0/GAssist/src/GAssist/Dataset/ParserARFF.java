/*
 * ParserARFF.java
 *
 */

package GAssist.Dataset;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.StringTokenizer;

import GAssist.LogManager;
import GAssist.Parameters;

public class ParserARFF {
	BufferedReader br;
	boolean isTrain;
	int attributeCount;
	String header;
	String relation;
	
	/** Creates a new instance of ParserARFF */
	public ParserARFF(String fileName,boolean _isTrain) {
		try {
			br=new BufferedReader(new FileReader(fileName));
		} catch(Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
		isTrain=_isTrain;
		attributeCount=0;
	}
	
	/**
	 *  Parses the header of the ARFF file
	 */
	public void parseHeader() {
		String st;
		header="";
		while(true) {
			st=getLine();
			if(st == null) {
				LogManager.println("Parse error in parseHeader(). Null found were not expected");
				System.exit(1);
			}
			header+=st+"\n";
			if(st.equalsIgnoreCase("@data")) break;
			parseHeaderLine(st);
		}

		if(isTrain) {
			Parameters.numAttributes=Attributes.getNumAttributes()-1;
			LogManager.println("Num attributes "+Parameters.numAttributes);
			Attribute att=Attributes.getAttribute(
				Parameters.numAttributes);
			if(att.type!=Attribute.NOMINAL) {
				LogManager.println("Class attribute should be nominal");
				System.exit(1);
			}
			Parameters.numClasses=att.getNumNominalValues();
			Attributes.endOfHeader();
		}
	}

	public String getInstance() {
		return getLine();
	}

	public void parseHeaderLine(String line) {
		StringTokenizer st = new StringTokenizer(line);
		String token=st.nextToken();

		if(token.equalsIgnoreCase("@relation")) {
			relation = new String(st.nextToken());
			if(isTrain) LogManager.println("Relation name "+relation);
		} else if(token.equalsIgnoreCase("@attribute")) {
			parseAttributeHeader(st);
		} else if(token.equalsIgnoreCase("@inputs")) {
		} else if(token.equalsIgnoreCase("@outputs")) {
		} else {
			LogManager.println("Parse error in parseHeader(). Unknown header line |"+token+"|");
			System.exit(1);
		}
	}

	public void parseAttributeHeader(StringTokenizer st) {
		Attribute at=new Attribute();

		String name=st.nextToken();
		if(name.charAt(0)=='\'') {
			int len=name.length();
			name=name.substring(1,len-1);
		}
		at.setName(name);
		if(isTrain) LogManager.println("Attribute name "+name);

		String type=st.nextToken();
		if(type.equalsIgnoreCase("real") 
				|| type.equalsIgnoreCase("integer")
				|| type.equalsIgnoreCase("numeric")) {
			at.setType(Attribute.REAL);
			if(st.hasMoreTokens()) {
				String bounds=st.nextToken("\n");
				StringTokenizer st2=new StringTokenizer(bounds,"[,] ");
				double min=Double.parseDouble(st2.nextToken());
				double max=Double.parseDouble(st2.nextToken());
				at.setBounds(min,max);
			}
		} else if(type.startsWith("{")) {
			at.setType(Attribute.NOMINAL);
			StringTokenizer st2 = new StringTokenizer(type,"{,}");
			while ( st2.hasMoreTokens() )
				at.addNominalValue(st2.nextToken());
		} else {
			LogManager.println("Unknown attribute type "+type);
			System.exit(1);
		}

		if(isTrain) Attributes.addAttribute(at);
		else {
			if(!Attributes.getAttribute(attributeCount).equals(at)){
				LogManager.println("Definition of attribute "+attributeCount+" in test file does not match the one in the training file");
				System.exit(1);
			}
		}
	
		attributeCount++;
	}

	public String getHeader() {
		return header;
	}

	public String getRelation() {
		return relation;
	}
			
	public String getLine() {
		String st=null;
		do {
			try {
				st=br.readLine();
			} catch(Exception e) {
				e.printStackTrace();
				System.exit(1);
			}
		} while(st!=null && (st.startsWith("%") || st.equals("")));
		return st;
	}
}
