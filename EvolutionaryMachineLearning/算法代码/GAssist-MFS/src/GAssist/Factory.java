/** 
 * Factory.java
 *
 */
package GAssist;
import GAssist.Dataset.*;

/**
 *
 */
public class Factory {
	static boolean realKR;

	public static void initialize() {
		boolean hasDefaultClass;
		if(Attributes.hasRealAttributes()) {
			if(Parameters.adiKR) {
				Globals_ADI.initialize();
				hasDefaultClass=Globals_ADI.hasDefaultClass();
			} else {
				Globals_UBR.initialize();
				hasDefaultClass=Globals_UBR.hasDefaultClass();
			}
			realKR=true;
		} else {
			realKR=false;
			Parameters.adiKR=false;
			Globals_GABIL.initialize();
			hasDefaultClass=Globals_GABIL.hasDefaultClass();
		}
		Globals_DefaultC.init(hasDefaultClass);
	}

	public static Classifier newClassifier() {
		if(realKR) {
			if(Parameters.adiKR) return new ClassifierADI();
			return new ClassifierUBR();
		}
		return new ClassifierGABIL();
	}
}
