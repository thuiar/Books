package lns;

import comet.CometException;
import comet.CometOptions;
import comet.CometSystem;
import common.ShopData;

public class LNS {
	CometSystem sys;
	CometOptions o;
    
    public LNS(String rName, String sName, String oName){
		sys = new CometSystem();
		o = new CometOptions();
		o.setFilename("POSTLNS.co");
		sys.setOptions(o);
		sys.addInput("sName", sName);
		sys.addInput("rName", rName);
		sys.addInput("oName", oName);
    }
    
    public void run(){
		try {
			sys.solve();

		} catch (CometException e) {
			System.out.println("In Java: caught:" + e);
		}
    }
    public static void main(String[] args){
    	LNS lns = new LNS("E:\\Text\\01a.fjs", "E:\\01a_yy.txt", "E:\\out_01a_yy.txt");
    	lns.run();
    }
    
}
