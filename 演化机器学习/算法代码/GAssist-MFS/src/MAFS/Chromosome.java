package MAFS;

import GAssist.Rand;

/***********************************************************************

 This file is part of KEEL-software, the Data Mining tool for regression, 
 classification, clustering, pattern mining and so on.

 Copyright (C) 2004-2010

 F. Herrera (herrera@decsai.ugr.es)
 L. Sánchez (luciano@uniovi.es)
 J. Alcal?Fdez (jalcala@decsai.ugr.es)
 S. García (sglopez@ujaen.es)
 A. Fernández (alberto.fernandez@ujaen.es)
 J. Luengo (julianlm@decsai.ugr.es)

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see http://www.gnu.org/licenses/

 **********************************************************************/

/**
 * 
 * File: Chromosome.java
 * 
 * A chromosome implementation for FS algorithms
 * 
 * @author Written by Joaquín Derrac (University of Granada) 13/11/2008
 * @version 1.0
 * @since JDK1.5
 * 
 */

public class Chromosome implements Comparable<Object> {

	// Cromosome data structure
	private int genes[];

	private double fitnessValue;
	private boolean valid;

	/**
	 * Builder.
	 * 
	 * @param size
	 *            Initial size
	 */
	public Chromosome(int size) {

		double u;

		genes = new int[size];

		for (int i = 0; i < size; i++) {
			u = Rand.getReal();
			if (u < 0.5) {
				genes[i] = 0;
			} else {
				genes[i] = 1;
			}
		}

		valid = false;
	}

	/**
	 * Builder.
	 * 
	 * @param info
	 *            Body of the chromosome
	 */
	public Chromosome(int info[]) {

		genes = new int[info.length];

		for (int i = 0; i < info.length; i++) {
			genes[i] = info[i];
		}

		valid = false;
	}

	/**
	 * Builder.
	 * 
	 * @param info
	 *            Body of the chromosome
	 * @param fitness
	 *            Fitness of the chromosome
	 */
	public Chromosome(int info[], double fitness) {

		genes = new int[info.length];

		for (int i = 0; i < info.length; i++) {
			genes[i] = info[i];
		}

		fitnessValue = fitness;
		valid = true;
	}

	/**
	 * Get the body of a chromosome
	 * 
	 * @return Body of a chromosome
	 */
	public int[] getGenes() {
		return genes;
	}

	/**
	 * Get the number of genes selected
	 * 
	 * @return Number of genes selected
	 */
	public int getNGenes() {

		int count = 0;

		for (int i = 0; i < genes.length; i++) {
			if (genes[i] == 1) {
				count++;
			}
		}

		return count;
	}

	/**
	 * Get the fitness value
	 * 
	 * @return Fitness value
	 */
	public double getFitness() {
		return fitnessValue;
	}

	/**
	 * Tests if the chromosome is valid
	 * 
	 * @return True if the chromosome is valid. False if not
	 */
	public boolean getValid() {
		return valid;
	}

	/**
	 * Gets reduction rate
	 * 
	 * @return Reduction rate
	 */
	private double getReductionRate() {

		double rate = 0.0;
		int count = 0;

		for (int i = 0; i < genes.length; i++) {
			count += genes[i];
		}

		rate = 1.0 - ((double) count / (double) genes.length);

		return rate;

	}

	/**
	 * PMX cross operator
	 * 
	 * @param parent
	 *            Parent chromosome
	 * 
	 * @return Offspring
	 */
	public int[] crossPMX(int[] parent) {

		int point1, point2;
		int down, up;
		int[] offspring;

		point1 = Rand.getInteger(0, parent.length - 1);
		point2 = Rand.getInteger(0, parent.length - 1);

		if (point1 > point2) {
			up = point1;
			down = point2;
		} else {
			up = point2;
			down = point1;
		}

		// crossing first offspring (self)

		for (int i = down; i < up; i++) {
			genes[i] = parent[i];
		}

		// crossing second offspring (outter)

		offspring = new int[parent.length];

		for (int i = 0; i < down; i++) {
			offspring[i] = parent[i];
		}

		for (int i = down; i < up; i++) {
			offspring[i] = genes[i];
		}

		for (int i = up; i < parent.length; i++) {
			offspring[i] = parent[i];
		}

		valid = false;

		return offspring;
	}

	/**
	 * Mutation Operator
	 */
	public void mutation() {

		int i;
		boolean change = false;

		for (i = 0; i < genes.length; i++) {
			if (Rand.getReal() < FSParameters.probMutation) {
				genes[i] = (genes[i] + 1) % 2;
				change = true;
			}
		}

		if (change) {
			valid = false;
		}
	}

	/**
	 * Compare to method
	 * 
	 * @param o1
	 *            Chromosome to compare
	 * 
	 * @return Relative order
	 */
	public int compareTo(Object o1) {
		if (this.fitnessValue > ((Chromosome) o1).fitnessValue)
			return -1;
		else if (this.fitnessValue < ((Chromosome) o1).fitnessValue)
			return 1;
		else
			return 0;
	}

	/**
	 * To string method
	 * 
	 * @return String representation of the chromosome
	 */
	public String toString() {

		int i;

		String temp = "[";

		for (i = 0; i < genes.length; i++) {
			temp += genes[i];
		}

		temp += ", " + String.valueOf(fitnessValue) + "]";

		return temp;
	}

	public void calFitness(double acc) {
		// TODO Auto-generated method stub

		double red;

		red = getReductionRate();
		fitnessValue = (FSParameters.beta * acc)
				+ ((1.0 - FSParameters.beta) * red);
		valid = true;
	}

}// end-class
