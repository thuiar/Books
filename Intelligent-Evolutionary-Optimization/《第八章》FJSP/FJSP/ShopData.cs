using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Collections;

namespace FJSP
{
    struct ProcessInfo:IComparable 
    {
        public int machineID;
        public double processTime;
        public ProcessInfo(int id, double time)
        {
            machineID = id;
            processTime = time;
        }
        public int CompareTo(object obj)
        {
            int res = 0;
            try
            {
                ProcessInfo sObj = (ProcessInfo)obj;
                if (this.processTime > sObj.processTime)
                {
                    res = 1;
                }
                else if (this.processTime < sObj.processTime)
                {
                    res = -1;
                }
            }
            catch (Exception ex)
            {
                throw new Exception("Compare Exception!", ex.InnerException);
            }
            return res; 
        }

    }
    struct Operation
    {
        public int jobID;
        public int opID;
        public Operation(int r, int k)
        {
            jobID = r;
            opID = k;
        }
    }
    class ShopData
    {
        protected int machineNum;
        protected int jobNum;
        protected List<ProcessInfo>[][] infoTable;
        protected int[] PJ;
        protected int[] SJ;
        protected Hashtable op2Id;
        protected List<Operation> id2Op;

        public ShopData(string filePath)
        {
            FileStream fs = new FileStream(filePath, FileMode.Open);
            StreamReader pr = new StreamReader(fs);
            string line = pr.ReadLine();

            string[] st = line.Trim().Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            jobNum = int.Parse(st[0].ToString());
            machineNum = int.Parse(st[1].ToString());
            infoTable = new List<ProcessInfo>[jobNum][];
            op2Id = new Hashtable();
            id2Op = new List<Operation>();
            int n = 0, i, j, k;
            for (i = 0; i < jobNum; i++)
            {
                line = pr.ReadLine();
                int opNum = int.Parse(line.Trim());
                infoTable[i] = new List<ProcessInfo>[opNum];
                for (j = 0; j < opNum; j++)
                {
                    line = pr.ReadLine();
                    st = line.Trim().Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                    //for (int x = 0; x < st.Length; x++)
                    //    Console.WriteLine(st[x]);
                    //Console.Read();
                    //Console.WriteLine();
                    int mNum = int.Parse(st[0].ToString());

                    infoTable[i][j] = new List<ProcessInfo>(mNum);
                    for (k = 1; k < 2 * mNum; k += 2)
                    {
                        int mID = int.Parse(st[k].ToString());
                        double pTime = double.Parse(st[k + 1].ToString());
                        infoTable[i][j].Add(new ProcessInfo(mID, pTime));
                    }
                    id2Op.Add(new Operation(i + 1, j + 1));
                    op2Id.Add(new Operation(i + 1, j + 1), ++n);
                    infoTable[i][j].Sort();
                }

            }
            PJ = new int[n + 1];
            SJ = new int[n + 1];
            n = 0;
            for (i = 0; i < jobNum; i++)
            {
                for (j = 0; j < infoTable[i].Length; j++)
                {
                    ++n;
                    if (j == 0)
                        PJ[n] = -1;
                    else if (j == infoTable[i].Length - 1)
                    {
                        SJ[n - 1] = n;
                        PJ[n] = n - 1;
                        SJ[n] = -1;
                    }
                    else
                    {
                        SJ[n - 1] = n;
                        PJ[n] = n - 1;
                    }
                }
            }


            //for (i = 0; i < jobNum; i++)
            //{
            //    for (j = 0; j < infoTable[i].Length; j++)
            //    {
            //        Console.WriteLine((i + 1) + " " + (j + 1));
            //        for (k = 0; k < infoTable[i][j].Count; k++)
            //        {
            //            Console.Write("M" + infoTable[i][j].ElementAt(k).machineID + ":" + (k + 1) + " ");
            //        }
            //        Console.WriteLine();
            //    }
            //}
            //Console.Read();
        }

        public int getMachineNmum()
        {
            return machineNum;
        }
        public int getJobNum()
        {
            return jobNum;
        }
        public int getOpNum(int r)
        {
            return infoTable[r - 1].Length;
        }
        public List<ProcessInfo> getProcessInfo(int r, int k)
        {
            return infoTable[r - 1][k - 1];
        }
        public List<ProcessInfo> getProcessInfo(int id)
        {
            Operation op = getOp(id);
            return getProcessInfo(op);
        }
        public List<ProcessInfo> getProcessInfo(Operation op)
        {
            return infoTable[op.jobID - 1][op.opID - 1];
        }

        public int getOptionalMachineNum(int r, int k)
        {
            return infoTable[r - 1][k - 1].Count;
        }
        public int getTotalOpNum()
        {
            int total = 0;
            for (int i = 1; i <= jobNum; i++)
                total += getOpNum(i);
            return total;
        }
        public Operation getOp(int id)
        {
            return id2Op.ElementAt(id - 1);
        }
        public int getID(Operation op)
        {
            return int.Parse(op2Id[op].ToString());
        }

        public int sucJOp(int id)
        {
            return SJ[id];
        }

        public int preJOp(int id)
        {
            return PJ[id];
        }

        public void print()
        {
            Console.Write(jobNum + " ");
            Console.WriteLine(machineNum);
            for (int i = 0; i < jobNum; i++)
            {
                Console.WriteLine(infoTable[i].Length);
                for (int j = 0; j < infoTable[i].Length; j++)
                {
                    Console.Write(infoTable[i][j].Count + " ");
                    for (int k = 0; k < infoTable[i][j].Count; k++)
                        Console.Write(infoTable[i][j].ElementAt(k).machineID + " " + infoTable[i][j].ElementAt(k).processTime + " ");
                    Console.WriteLine();
                }
            }

        }
    }
}
