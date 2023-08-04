package boa.util;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;

public class Transform {

	public static void transform(String srcFile, String disFile, double[][] split) {
		try {

			InputStream is = new FileInputStream(srcFile);
			OutputStream os = new FileOutputStream(disFile);

			BufferedReader reader = new BufferedReader(
					new InputStreamReader(is));
			PrintStream ps = new PrintStream(os);

			String line = reader.readLine(); // 读取第一行
			StringBuffer buffer = new StringBuffer();

			boolean dataFlag = false;
			int index =0;
			
			while (line != null) { // 如果 line 为空说明读完了
				if (dataFlag) {//离散化
					String[] data = line.split(",");
					for (int i = 0; i < data.length-1; i++) {
						
						if(data[i].contains("?")){
							ps.print(data[i] + ",");
						}else{
							if(split[i]!=null){
								double src = Double.parseDouble(data[i]);
								int dis = MyUtil.discretizer(src, split[i]);
								ps.print(dis + ",");
							}else{
								ps.print(data[i] + ",");
							}
							
						}
						
					}
					//output the class label
					ps.print(data[data.length-1]);
					ps.println();
				} else { //对原有属性定义进行修改
					if (line.trim().matches("@data")) {
						dataFlag = true;
					}else if(line.contains("@attribute")){
						System.out.println(line);
						if(line.contains("real")){
							line = line.replace("real", MyUtil.getArray(split[index].length));
						}else if(line.contains("numeric")){
							line = line.replace("numeric", MyUtil.getArray(split[index].length));
						}
						index++;
					}
					ps.println(line);
				}
				
				line = reader.readLine();
			}

			is.close();
			os.close();

		} catch (Exception ex) {
			ex.printStackTrace();
		}

	}
}
