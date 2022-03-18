/*
 * LogManager.java
 *
 * Class that controls all the log message. For simplicity purpouses all log
 * will be directed to stdout and stderr
 */

package GAssist;

/**
 * Class that controls all the log message. For simplicity, all log will be directed to stdout and stderr.<br>
 * 统一控制所有Log的输出，如有需要可以重定向到指定文件。
 */
public class LogManager {
   	static FileManagement logFile;
   
	public static void initLogManager() {
		/*logFile= new FileManagement();
		try {
			logFile.initWrite(Parameters.logOutputFile);
		} catch(Exception e) {
			System.err.println("Failed initializing log file");
			System.exit(1);
		}*/
	}
	
	public static void initLogManager(String filename) {
		logFile= new FileManagement();
		try {
			logFile.initWrite(filename);
		} catch(Exception e) {
			System.err.println("Failed initializing log file");
			System.exit(1);
		}
	}
	
	public static void println(String line) {
		try {
			//logFile.writeLine(line+"\n");
			System.out.println(line);
		} catch(Exception e) {
			System.err.println("Failed writing to log");
			System.exit(1);
		}
	}

	public static void printErr(String line) {
		try {
			System.err.println(line);
			//logFile.writeLine(line+"\n");
		} catch(Exception e) {
			System.err.println("Failed writing to log");
			System.exit(1);
		}
	}


	public static void print(String line) {
		try {
			System.out.print(line);
			//logFile.writeLine(line);
		} catch(Exception e) {
			System.err.println("Failed writing to log");
			System.exit(1);
		}
	}

	
	public static void closeLog() {
		try {
			//logFile.closeWrite();
		} catch(Exception e) {
			System.err.println("Failed closing log file");
			System.exit(1);
		}
	}
	
	//Added by WY, 07/01/2011
	public static void println_file(String line) {
		try {
			logFile.writeLine(line+"\n");
		} catch(Exception e) {
			System.err.println("Failed writing to log");
			System.exit(1);
		}
	}
	
	//Added by WY, 07/01/2011
	public static void print_file(String line) {
		try {
			logFile.writeLine(line);
		} catch(Exception e) {
			System.err.println("Failed writing to log");
			System.exit(1);
		}
	}
}
