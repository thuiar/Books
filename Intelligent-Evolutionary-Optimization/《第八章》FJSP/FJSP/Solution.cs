using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace FJSP
{
    class Solution
    {
        ShopData shopData;
        double[] processTime;
        List<List<int>> block;
        double makeSpan;
        int[] PM;
        int[] SM;
        int[] machineID;
        int[] firstOp;
        int[] jobNumOfMachine;
        

        public Solution(ShopData shopData, Particle p)
        {
            this.shopData = shopData;
            opSequence(p);
            computePathInfo();
        }
        public Solution() { }

        private void opSequence(Particle p)
        {
            processTime = new double[p.dim + 1];
            PM = new int[p.dim + 1];
            SM = new int[p.dim + 1];
            machineID = new int[p.dim + 1];
            firstOp = new int[shopData.getMachineNmum() + 1];
            jobNumOfMachine = new int[shopData.getMachineNmum() + 1];
            int[] lastOp = new int[shopData.getMachineNmum() + 1];
            int i;
            for (i = 1; i <= shopData.getMachineNmum(); i++)
            {
                firstOp[i] = -1;
                jobNumOfMachine[i] = 0;
            }
            for (i = 0; i < p.dim; i++)
            {
                int mId = shopData.getProcessInfo(i + 1).ElementAt(p.X[i] - 1).machineID;
                processTime[i + 1] = shopData.getProcessInfo(i + 1).ElementAt(p.X[i] - 1).processTime;
                machineID[i + 1] = mId;
                if (firstOp[mId] == -1)
                {
                    firstOp[mId] = lastOp[mId] = i + 1;
                    PM[i + 1] = -1;
                    jobNumOfMachine[mId]++;
                }
                else
                {
                    SM[lastOp[mId]] = i + 1;
                    PM[i + 1] = lastOp[mId];
                    lastOp[mId] = i + 1;
                    jobNumOfMachine[mId]++;
                }
            }
            for (i = 1; i <= shopData.getMachineNmum(); i++)
                SM[lastOp[i]] = -1;
        }
        private bool computePathInfo()
        {
            int[] topicalSort = getTopicalSort();
            if (topicalSort == null)
            {
                this.makeSpan = double.MaxValue;
                return false;
            }

            int n = topicalSort.Length;
            double[] sE = new double[n];
            double[] sL = new double[n];

            double makeSpan = 0;
            int w, i;
            

            for (i = 1; i < n; i++)
            {
                w = topicalSort[i];
                sE[w] = 0;
                if (shopData.preJOp(w) != -1)
                    sE[w] = sE[shopData.preJOp(w)] + processTime[shopData.preJOp(w)];
                if (PM[w] != -1 && sE[PM[w]] + processTime[PM[w]] > sE[w])
                    sE[w] = sE[PM[w]] + processTime[PM[w]];
                if (sE[w] + processTime[w] > makeSpan)
                    makeSpan = sE[w] + processTime[w];
            }
            this.makeSpan = makeSpan;

            for (i = n - 1; i >= 1; i--)
            {
                w = topicalSort[i];
                sL[w] = makeSpan - processTime[w];
                if (shopData.sucJOp(w) != -1)
                    sL[w] = sL[shopData.sucJOp(w)] - processTime[w];
                if (SM[w] != -1 && sL[SM[w]] - processTime[w] < sL[w])
                    sL[w] = sL[SM[w]] - processTime[w];
            }

            List<int> criticalOpList = new List<int>();
            for (i = 1; i < n; i++)
            {
                w = topicalSort[i];
                if (sL[w] == sE[w])
                    criticalOpList.Add(w);
                
            }

            List<int> bk;
            block = new List<List<int>>();
            int u = criticalOpList.ElementAt(0);
            while (u != -1)
            {
                bk = new List<int>();
                do {
                    bk.Add(u);
                    u = SM[u];
                } while (criticalOpList.Contains(u));
                block.Add(bk);
                u = shopData.sucJOp(bk.Last());
            }
            return true;
        }

        private int[] getTopicalSort()
        {
            int n = PM.Length;
            int i;
            int top = 0, v, k = 0;
            int[] inDegree = new int[n];
            int[] topicalSort = new int[n];
            for (i = 1; i < n; i++)
            {
                if (shopData.preJOp(i) != -1 && PM[i] != -1)
                    inDegree[i] = 2;
                else if (shopData.preJOp(i) != -1 || PM[i] != -1)
                    inDegree[i] = 1;
                else
                    inDegree[i] = 0;
            }
            for (i = 1; i < n; i++)
            {
                if (inDegree[i] == 0)
                {
                    inDegree[i] = top;
                    top = i;
                }
            }
            for (i = 1; i < n; i++)
            {
                if (top == 0)
                    return null;
                else
                {
                    v = top;
                    top = inDegree[top];
                    topicalSort[++k] = v;

                    if (shopData.sucJOp(v) != -1 && --inDegree[shopData.sucJOp(v)] == 0)
                    {
                        inDegree[shopData.sucJOp(v)] = top;
                        top = shopData.sucJOp(v);
                    }
                    if (SM[v] != -1 && --inDegree[SM[v]] == 0)
                    {
                        inDegree[SM[v]] = top;
                        top = SM[v];
                    }
                }
            }
            return topicalSort;
        }

        static int GetRandomSeed()
        {
            byte[] bytes = new byte[4];
            System.Security.Cryptography.RNGCryptoServiceProvider rng = new System.Security.Cryptography.RNGCryptoServiceProvider();
            rng.GetBytes(bytes);
            return BitConverter.ToInt32(bytes, 0);
        } 


        public Solution genNeighbor_I()
        {

            Solution newSolution = new Solution();
            newSolution.shopData = this.shopData;
            newSolution.processTime = this.processTime;
            newSolution.machineID = this.machineID;
            newSolution.PM = new int[this.PM.Length];
            newSolution.SM = new int[this.SM.Length];
            newSolution.firstOp = new int[this.firstOp.Length];
            newSolution.jobNumOfMachine = this.jobNumOfMachine;
            for (int i = 1; i < this.PM.Length; i++)
            {
                newSolution.PM[i] = this.PM[i];
                newSolution.SM[i] = this.SM[i];
            }
            for (int i = 1; i < this.firstOp.Length; i++)
                newSolution.firstOp[i] = this.firstOp[i];
            

            int id1, id2, r, k;
            Random random = new Random(GetRandomSeed());
            while (true)
            {
                r = random.Next(1, shopData.getMachineNmum() + 1);
                if (jobNumOfMachine[r] >= 2)
                {
                    k = random.Next(1, jobNumOfMachine[r]);
                    //Console.WriteLine("r: " + r + "k: " + k);
                    id1 = firstOp[r];
                    for (int i = 1; i < k;i++)
                        id1 = SM[id1];
                    break;

                }
                
            }
            id2 = SM[id1];

            //Console.WriteLine("ID1: " + id1 + " " + "ID2: " + id2);

            int a = PM[id1];
            int b = SM[id1];
            int c = PM[id2];
            int d = SM[id2];

            newSolution.PM[id1] = id2;
            newSolution.SM[id1] = d;
            newSolution.PM[id2] = a;
            newSolution.SM[id2] = id1;

            if (a != -1)
                newSolution.SM[a] = id2;
            if (d != -1)
                newSolution.PM[d] = id1;

            if (firstOp[machineID[id1]] == id1)
                newSolution.firstOp[machineID[id1]] = id2;

            if (newSolution.computePathInfo())
                return newSolution;
            else
                return this;
        }
        public Solution genNeighbor_III()
        {
            if (this.block.Count == 1)
                return null;
            else
            {
                int i;
                for (i = 0; i < this.block.Count; i++)
                {
                    if (this.block.ElementAt(i).Count != 1)
                        break;
                }
                if (i == this.block.Count)
                    return null;
            }

            Solution newSolution = new Solution();
            newSolution.shopData = this.shopData;
            newSolution.processTime = this.processTime;
            newSolution.machineID = this.machineID;
            newSolution.PM = new int[this.PM.Length];
            newSolution.SM = new int[this.SM.Length];
            newSolution.firstOp = new int[this.firstOp.Length];

            for (int i = 1; i < this.PM.Length; i++)
            {
                newSolution.PM[i] = this.PM[i];
                newSolution.SM[i] = this.SM[i];
            }
            for (int i = 1; i < this.firstOp.Length; i++)
                newSolution.firstOp[i] = this.firstOp[i];

            int id1, id2, r, k;
            Random random = new Random(GetRandomSeed());
            while (true)
            {
                r = random.Next(0, this.block.Count);
                if (block.ElementAt(r).Count >= 2)
                    break;
            }
            if (r == 0)
            {
                id1 = block.ElementAt(r).ElementAt(block.ElementAt(r).Count - 2);
                id2 = block.ElementAt(r).Last();
            }
            else if (r == block.Count - 1)
            {
                id1 = block.ElementAt(r).ElementAt(0);
                id2 = block.ElementAt(r).ElementAt(1);
            }
            else
            {
                if (random.Next(0, 2) == 0)
                {
                    id1 = block.ElementAt(r).ElementAt(0);
                    id2 = block.ElementAt(r).ElementAt(1);
                }
                else
                {
                    id1 = block.ElementAt(r).ElementAt(block.ElementAt(r).Count - 2);
                    id2 = block.ElementAt(r).Last();
                }
            }

            int a = PM[id1];
            int b = SM[id1];
            int c = PM[id2];
            int d = SM[id2];

            newSolution.PM[id1] = id2;
            newSolution.SM[id1] = d;
            newSolution.PM[id2] = a;
            newSolution.SM[id2] = id1;

            if (a != -1)
                newSolution.SM[a] = id2;
            if (d != -1)
                newSolution.PM[d] = id1;

            if (firstOp[machineID[id1]] == id1)
                newSolution.firstOp[machineID[id1]] = id2;

            if (newSolution.computePathInfo())
                return newSolution;
            else
                return this;
        }

        public Solution genNeighbor_II()
        {
            if (this.block.Count == 1)
                return null;
            else
            {
                int i;
                for (i = 0; i < this.block.Count; i++)
                {
                    if (this.block.ElementAt(i).Count != 1)
                        break;
                }
                if (i == this.block.Count)
                    return null;
            }

            Solution newSolution = new Solution();
            newSolution.shopData = this.shopData;
            newSolution.processTime = this.processTime;
            newSolution.machineID = this.machineID;
            newSolution.PM = new int[this.PM.Length];
            newSolution.SM = new int[this.SM.Length];
            newSolution.firstOp = new int[this.firstOp.Length];

            for (int i = 1; i < this.PM.Length; i++)
            {
                newSolution.PM[i] = this.PM[i];
                newSolution.SM[i] = this.SM[i];
            }
            for (int i = 1; i < this.firstOp.Length; i++)
                newSolution.firstOp[i] = this.firstOp[i];

            int id1, id2, r, k;
            Random random = new Random(GetRandomSeed());
            while (true)
            {
                r = random.Next(0, this.block.Count);
                if (block.ElementAt(r).Count >= 2)
                {
                    k = random.Next(0, block.ElementAt(r).Count - 1);
                    break;
                }
            }
            id1 = block.ElementAt(r).ElementAt(k);
            id2 = block.ElementAt(r).ElementAt(k + 1);
            //Console.WriteLine(shopData.getOp(id1).jobID + " " + shopData.getOp(id1).opID);
            //Console.WriteLine(shopData.getOp(id2).jobID + " " + shopData.getOp(id2).opID);
            int a = PM[id1];
            int b = SM[id1];
            int c = PM[id2];
            int d = SM[id2];

            newSolution.PM[id1] = id2;
            newSolution.SM[id1] = d;
            newSolution.PM[id2] = a;
            newSolution.SM[id2] = id1;

            if (a != -1)
                newSolution.SM[a] = id2;
            if (d != -1)
                newSolution.PM[d] = id1;

            if (firstOp[machineID[id1]] == id1)
                newSolution.firstOp[machineID[id1]] = id2;

            if (newSolution.computePathInfo())
            {
                //newSolution.print();
                //Console.Read();
                return newSolution;
            }
            else
            {
                //this.print();
                //Console.Read();
                return this;
            }
        }
        public double getMakeSpan()
        {
            return makeSpan;
        }

        public void print()
        {
            int i;
            for (i = 1; i <= shopData.getMachineNmum(); i++)
            {
                Console.WriteLine("Machine " + i + ": ");
                int u = firstOp[i];
                //Console.WriteLine("u: " + u);
                //Console.Read();
                while (u != -1)
                {
                    Console.Write("(" + shopData.getOp(u).jobID + "," + shopData.getOp(u).opID  + ") ");
                    u = SM[u];
                }
                Console.WriteLine();
            }
            int[] topicalSort = getTopicalSort();
            int n = topicalSort.Length;
            double[] sE = new double[n];
            double[] sL = new double[n];

            double makeSpan = 0;
            int w;


            for (i = 1; i < n; i++)
            {
                w = topicalSort[i];
                sE[w] = 0;
                if (shopData.preJOp(w) != -1)
                    sE[w] = sE[shopData.preJOp(w)] + processTime[shopData.preJOp(w)];
                if (PM[w] != -1 && sE[PM[w]] + processTime[PM[w]] > sE[w])
                    sE[w] = sE[PM[w]] + processTime[PM[w]];
                if (sE[w] + processTime[w] > makeSpan)
                    makeSpan = sE[w] + processTime[w];
            }
            Console.WriteLine(makeSpan);
            for (i = 1; i < n; i++)
            {
                Console.WriteLine(shopData.getOp(i).jobID + "-" + shopData.getOp(i).opID + " " + sE[i] + " " + (sE[i] + processTime[i]));
            }

        }
    }
}
