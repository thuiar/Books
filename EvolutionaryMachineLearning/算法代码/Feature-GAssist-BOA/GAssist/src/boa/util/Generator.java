package boa.util;

import java.io.FileOutputStream;
import java.io.OutputStream;
import java.io.PrintStream;

import boa.func.CountOne;
import boa.func.HiddenParity;
import boa.func.Multiplexer;
import boa.func.ParityMultiplexer;
import boa.func.ParityOne;

public class Generator {
	public static void writeMultiplexer(String file, int bits, int lines) {
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
			ps.println("@data");
			for (int i = 0; i < lines; i++) {
				int code[] = new int[length];
				for (int j = 0; j < length; j++) {
					if (Math.random() > 0.5)
						code[j] = 0;
					else
						code[j] = 1;
					ps.print(code[j]);
					ps.print(",");
				}
				int result = Multiplexer.valid(code, bits);
				ps.println(result);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void writeParityMultiplexer(String file, int pbits, int mAddressbits, int lines) {
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
			

			// output data
			ps.println("@data");
			for (int i = 0; i < lines; i++) {
				int code[] = new int[length];
				for (int j = 0; j < length; j++) {
					if (Math.random() > 0.5)
						code[j] = 0;
					else
						code[j] = 1;
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
	
	public static void writeParity(String file, int bits, int validBits, int lines) {
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
				int result = HiddenParity.valid(code, validBits);
				ps.println(result);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void writeCountOne(String file, int bits, int lines) {
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

	public static void writeParityOne(String file, int pbits, int oneBits, int lines) {
		try {
			OutputStream os = new FileOutputStream(file);
			PrintStream ps = new PrintStream(os);

			int length = pbits * oneBits;
			
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
			ps.println("@data");
			for (int i = 0; i < lines; i++) {
				int code[] = new int[length];
				for (int j = 0; j < length; j++) {
					if (Math.random() > 0.5)
						code[j] = 0;
					else
						code[j] = 1;
					ps.print(code[j]);
					ps.print(",");
				}
				int result = ParityOne.valid(code, pbits ,oneBits);
				ps.println(result);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}

}
