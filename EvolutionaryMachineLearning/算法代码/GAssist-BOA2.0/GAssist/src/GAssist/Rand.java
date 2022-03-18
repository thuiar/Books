/*
 * Rand.java
 *
 * Created on 29 de marzo de 2004, 23:31
 */

/**
 *
 */

package GAssist;

import java.util.Random;

public class Rand {

	private static MTwister random;

    /** Generates a new instance of Random */
	public static void initRand() {
		if(Parameters.seed==-1) {
			Random tmpRnd=new Random();
			Parameters.seed=tmpRnd.nextLong();
		}
		random = new MTwister(Parameters.seed);
		LogManager.println("Random seed is "+Parameters.seed);
	}
    /**
     *  Returns a random real between [0,1)
     */ public static double getReal() {
		return random.genrand_real2();
	}

    /**
     *  Returns a random long between [uLow,uHigh]
     */
	public static int getInteger(int uLow, int uHigh) {
		return (uLow +
			(int) (random.genrand_real2() *
			       (uHigh + 1 - uLow)));
	}

}
