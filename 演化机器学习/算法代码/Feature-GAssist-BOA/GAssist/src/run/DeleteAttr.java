package run;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

public class DeleteAttr {
	
	public static void main(String[] arg){
		
		String idx = "0";
		int[] index = new int[]{38,62,30,65,135,15,12,17,98,106,19,82,103,83,148,100,112,72,57,152,123,22,120,49,75,114,9,61,93,8,32,4,125,52,23,69,118,59,58,136,149,121,122,27,33,130,64,138,134,1,113,154,55,87,26,117,39,40,86,97,85,50,41,101,145,128,6,91,48,84,73,150,46,116,127,108,54,146,90,115,81,47,31,67,142,159,37,60,2,139,20,21,53,56,28,18,34,129,140,51,107,133,43,13,16,10,24,157,5,141,88,158,76,147,94};
		                               				 
		
		String data = "musk2";
		String type="info";
		//String type="relief";
		String src1 = "D://Document/Paper/ML/data/binary/real/"+data+"/"+idx+"-training-"+data+".arff";
		String des1 = "D://Document/Paper/ML/data/binary/real/"+data+"/"+type+"_"+idx+"-training-"+data+".arff";
		
		String src2 = "D://Document/Paper/ML/data/binary/real/"+data+"/"+idx+"-test-"+data+".arff";
		String des2 = "D://Document/Paper/ML/data/binary/real/"+data+"/"+type+"_"+idx+"-test-"+data+".arff";
		
		
		
		List<Integer> attr = new ArrayList<Integer>();
		for(int i=0;i<index.length;i++){
			attr.add(index[i]-1);
		}
		
		System.out.println(index.length);
		
		DeleteAttr.splitFile(src1, des1, attr);
		DeleteAttr.splitFile(src2, des2, attr);
	}
	

	public static void splitFile(String src, String des, List<Integer> attr) {

		try {
			InputStream is = new FileInputStream(src);
			OutputStream os = new FileOutputStream(des);

			PrintStream ps = new PrintStream(os);

			StringBuffer buffer = new StringBuffer();
			String line; // 用来保存每行读取的内容
			BufferedReader reader = new BufferedReader(
					new InputStreamReader(is));

			boolean dataFlag = false;
			int counter = 1;

			int attrIndex = 0;

			line = reader.readLine(); // 读取第一行
			while (line != null) { // 如果 line 为空说明读完了
				//System.out.println(line);

				if (line.contains("@relation")) {
					ps.println(line);
				}else if (line.contains("@attribute")) {
					if (attr.contains(attrIndex)) {

					} else {
						ps.println(line);
					}
					attrIndex++;
				} else if (line.matches("@data")) {
					dataFlag = true;
					ps.println(line);
				} else if (dataFlag) {
					String[] values = line.split(",");
					StringBuffer buf = new StringBuffer();
					for (int i = 0; i < values.length - 1; i++) {
						if (!attr.contains(i)) {
							buf.append(values[i]);
							buf.append(",");
						}
					}
					buf.append(values[values.length - 1]);
					ps.println(buf.toString());
				}

				line = reader.readLine(); // 读取下一行

			}
			is.close();
			os.close();

		} catch (Exception e) {
			e.printStackTrace();
		}

	}

}
