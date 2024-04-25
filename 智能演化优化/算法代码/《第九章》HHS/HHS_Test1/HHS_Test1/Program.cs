using System;
using System.Collections;
using System.IO;
using System.Text.RegularExpressions;

namespace HHS_Test1
{
    class Program
    {
        class op
        {
            int job;
            int stage;
            int time;
        }
        static int machines;
        static int totaljobs;
        static int[][][] jobtables;
        static void Main(string[] args)
        {
            try
            {
                StreamReader sr = new StreamReader("test.txt");
                try
                {
                    machines = int.Parse(sr.ReadLine());
                }
                catch (FormatException fe)
                {
                    Console.WriteLine(fe);
                }
                try
                {
                    totaljobs = int.Parse(sr.ReadLine());
                }
                catch (FormatException fe)
                {
                    Console.WriteLine(fe);
                }
                Console.WriteLine("machines={0:D}, totaljobs={1:D}", machines, totaljobs);
                jobtables = new int[totaljobs][][];
                int job = 0;
                string str = sr.ReadLine();
                while (str != null)
                {
                    Console.WriteLine(str);
                    string[] temp = str.Split('\t');
                    ArrayList ele = new ArrayList(temp);
                    if (temp[0].Equals("") == false)
                    {
                        int stages = 1;
                        ele.RemoveRange(0, 2);
                        while ((str = sr.ReadLine()) != null && (str.Split('\t')[0].Equals("") == true))
                        {
                            Console.WriteLine(str);
                            string[] temp1 = str.Split('\t');
                            for (int i = 2; i < temp1.Length; i++)
                            {
                                ele.Add(temp1[i]);
                            }
                            stages++;
                        }
                        jobtables[job] = new int[stages][];
                        for (int k = 0; k < stages; k++)
                        {
                            jobtables[job][k] = new int[machines];
                        }
                        string[] temp2 = (string[])ele.ToArray(typeof(string));
                        int stage = 0;
                        Regex regex = new Regex("^[0-9]*[1-9][0-9]*$");
                        for (int i = 0; i < temp2.Length; i += machines)
                        {
                            for (int j = 0; j < machines; j++)
                            {
                                if (regex.IsMatch(temp2[i + j]))
                                {
                                    jobtables[job][stage][j] = int.Parse(temp2[i + j].Trim());
                                }
                                else
                                {
                                    jobtables[job][stage][j] = -1;
                                }
                            }
                            stage++;
                        }
                        job++;
                    }
                }
            }
            catch (FileNotFoundException fnf)
            {
                Console.WriteLine(fnf.Message);
            }
            for (int i = 0; i < totaljobs; i++)
            {
                for (int j = 0; j <= jobtables[i].GetUpperBound(0); j++)
                {
                    for (int k = 0; k < machines; k++)
                    {
                        Console.Write("{0:D}\t", jobtables[i][j][k]);
                    }
                    Console.WriteLine("");
                }
            }
        }
    }
}
