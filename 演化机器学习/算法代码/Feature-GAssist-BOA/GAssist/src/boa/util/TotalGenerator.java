package boa.util;

import java.io.FileOutputStream;
import java.io.OutputStream;
import java.io.PrintStream;

import boa.func.CountOne;
import boa.func.Multiplexer;
import boa.func.ParityMultiplexer;

public class TotalGenerator {
	
	
	public static void writeParityMultiplexer(String file, int pbits, int mAddressbits) {
		try {
			OutputStream os = new FileOutputStream(file);
			PrintStream ps = new PrintStream(os);

			int length = (int) Math.pow(2, mAddressbits) + mAddressbits;
			length *= pbits;
			
			
			ps.println("@relation Training");
			ps.println();
			
			//attributes
			for (int i = 0; i <  length; i++) {
				ps.print("@attribute ");
				ps.print("Bit"+i+" ");
				ps.println("{0,1}");
			}
			ps.println("@attribute result {0,1}");
			ps.println();
			
			int lines = (int) Math.pow(2, length);

			// output data
			ps.println("@data");
			for (int integer = 0; integer < lines; integer++) {
				int code[] = MyUtil.IntToBinary(integer,length);
				for (int j = 0; j < length; j++) {
					ps.print(code[j]);
					ps.print(",");
				}
				int result = ParityMultiplexer.valid(code, pbits ,mAddressbits);
				ps.println(result);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void writeMultiplexer(String file, int bits, double noise) {
		
		try {
			OutputStream os = new FileOutputStream(file);
			PrintStream ps = new PrintStream(os);

			int length = (int) Math.pow(2, bits) + bits;
			
			ps.println("@relation Training");
			ps.println();
			
			//attributes
			for (int i = 0; i <  length; i++) {
				ps.print("@attribute ");
				ps.print("Bit"+i+" ");
				ps.println("{0,1}");
			}
			ps.println("@attribute result {0,1}");
			ps.println();
			
			// output data
			int lines = (int) Math.pow(2, length);
			
			ps.println("@data");
			for (int integer = 0; integer < lines; integer++) {
				int code[] = MyUtil.IntToBinary(integer,length);
				for (int j = 0; j < length; j++) {
					ps.print(code[j]);
					ps.print(",");
				}
				int result =  Multiplexer.valid(code, bits);
				double s = Math.random(); 
				if(s>noise){
					ps.println(result);
				}else{
					ps.println(1-result);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void writeCountOne(String file, int bits) {
		
		int lines = (int) Math.pow(2, bits);
		
		try {
			OutputStream os = new FileOutputStream(file);
			PrintStream ps = new PrintStream(os);

			ps.println("@relation Training");
			ps.println();
			
			//attributes
			for (int i = 0; i <  bits; i++) {
				ps.print("@attribute ");
				ps.print("Bit"+i+" ");
				ps.println("{0,1}");
			}
			ps.println("@attribute result {0,1}");
			ps.println();
			

			// output data
			ps.println("@data");
			for (int i = 0; i < lines; i++) {
				int code[] = new int[bits];
				for (int j = 0; j < bits; j++) {
					if (Math.random() > 0.5)
						code[j] = 0;
					else
						code[j] = 1;
					ps.print(code[j]);
					ps.print(",");
				}
				int result = CountOne.valid(code);
				ps.println(result);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}

	public static void writeRedundancyCountOne(String file, int bits, int redundancy) {		
		try {
			OutputStream os = new FileOutputStream(file);
			PrintStream ps = new PrintStream(os);
	
			int length = bits;
			
			ps.println("@relation Training");
			ps.println();
			
			//attributes
			for (int i = 0; i <  length+redundancy; i++) {
				ps.print("@attribute ");
				ps.print("Bit"+i+" ");
				ps.println("{0,1}");
			}
			ps.println("@attribute result {0,1}");
			ps.println();
			
			// output data
			int lines = (int) Math.pow(2, length);
			
			ps.println("@data");
			for (int integer = 0; integer < lines; integer++) {
				int code[] = MyUtil.IntToBinary(integer,length);
				for (int j = 0; j < length; j++) {
					ps.print(code[j]);
					ps.print(",");
				}
				for (int j = 0; j < redundancy; j++) {
					if(Math.random()>0.5){
						ps.print("1");
					}else{
						ps.print("0");
					}
					ps.print(",");
				}
				
				int result =  CountOne.valid(code);
				ps.println(result);
				
			}
			
		}catch(Exception ex){
			ex.printStackTrace();
		}
		
	}
	
	public static void writeRedundancyMultiplexer(String file, int bits, int redundancy) {
		
		try {
			OutputStream os = new FileOutputStream(file);
			PrintStream ps = new PrintStream(os);

			int length = (int) Math.pow(2, bits) + bits;
			
			
			
			ps.println("@relation Training");
			ps.println();
			
			//attributes
			for (int i = 0; i <  length+redundancy; i++) {
				ps.print("@attribute ");
				ps.print("Bit"+i+" ");
				ps.println("{0,1}");
			}
			ps.println("@attribute result {0,1}");
			ps.println();
			
			// output data
			int lines = (int) Math.pow(2, length);
			
			ps.println("@data");
			for (int integer = 0; integer < lines; integer++) {
				int code[] = MyUtil.IntToBinary(integer,length);
				for (int j = 0; j < length; j++) {
					ps.print(code[j]);
					ps.print(",");
				}
				for (int j = 0; j < redundancy; j++) {
					if(Math.random()>0.5){
						ps.print("1");
					}else{
						ps.print("0");
					}
					ps.print(",");
				}
				
				int result =  Multiplexer.valid(code, bits);
				ps.println(result);
				
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
