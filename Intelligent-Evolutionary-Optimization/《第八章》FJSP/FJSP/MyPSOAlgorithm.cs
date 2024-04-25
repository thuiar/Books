using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace FJSP
{
    class MyPSOAlgorithm:PSOAlgorithm
    {
        ShopData shopData;
        public Solution bestSolution;
        public MyPSOAlgorithm(int d, int n) : base(d, n) { }
        public MyPSOAlgorithm(ShopData sd, int num):base(sd.getTotalOpNum(),num)
        {
            shopData = sd;
            int dim = sd.getTotalOpNum();
            int[] xup = new int[dim];
            int[] xdown = new int[dim];
            int[] vmax = new int[dim];
            int i, j, k = 0, n = 0;
            for (i = 0; i < sd.getJobNum(); i++)
            {
                for (j = k; j < k + sd.getOpNum(i + 1); j++)
                {
                    xdown[n] = 1;
                    xup[n] = sd.getOptionalMachineNum(i + 1, j - k + 1);
                    //xup[n] = 1;
                    vmax[n] = xup[n];
                    //vmax[n] = 2;
                    n++;
                }
            }
            //xdown[1] = xup[1] = 2;
            //vmax[1] = 2;
            //xdown[3] = xup[3] = 2;
            //vmax[3] = 2;
            //xdown[18] = xup[18] = 2;
            //vmax[18] = 2;
            //xdown[36] = xup[36] = 3;
            //vmax[36] = 3;
            //xdown[39] = xup[39] = 2;
            //vmax[39] = 2;
            //xdown[42] = xup[42] = 2;
            //vmax[42] = 2;
            //xdown[49] = xup[49] = 2;
            //vmax[49] = 2;
            //xdown[52] = xup[52] = 2;
            //vmax[52] = 2;
            //xdown[53] = xup[53] = 3;
            //vmax[53] = 2;
            //xdown[55] = xup[55] = 2;
            //vmax[55] = 2;


            //xdown[1] = xup[1] = 2;
            //vmax[1] = 2;
            //xdown[7] = xup[7] = 2;
            //vmax[7] = 2;
            //xdown[18] = xup[18] = 2;
            //vmax[18] = 2;
            //xdown[33] = xup[33] = 2;
            //vmax[33] = 2;
            //xdown[36] = xup[36] = 2;
            //vmax[36] = 2;
            //xdown[49] = xup[49] = 2;
            //vmax[49] = 2;
            //xdown[54] = xup[54] = 2;
            //vmax[54] = 2;
            //xdown[55] = xup[55] = 2;
            //vmax[55] = 2;
            setXdown(xdown);
            setXup(xup);
            setVmax(vmax);
        }
        public Solution getInitialSolution(Particle p)
        {
           return new Solution(shopData, p);
        }

        public override double getFit(Particle p)
        {
            Solution curSolution = getInitialSolution(p), newSolution;
            int L = shopData.getTotalOpNum() - shopData.getMachineNmum();
            double Tk=10, T0=10, Te=0.01, B = 0.95;
            double best = double.MaxValue;
      
            while (Tk > Te)
            {
                for (int i = 1; i <= 10; i++)
                {
                    newSolution = curSolution.genNeighbor_II();

                    double delta;
                    if (newSolution != null)
                        delta = newSolution.getMakeSpan() - curSolution.getMakeSpan();
                    else
                        return best;
                    if (delta <= 0 || Math.Exp(-delta / Tk) > new Random().NextDouble())
                    {
                        curSolution = newSolution;
                        if (curSolution.getMakeSpan() < best)
                        {
                            if (bestSolution == null)
                                bestSolution = curSolution;
                            else
                            {
                                if (curSolution.getMakeSpan() < bestSolution.getMakeSpan())
                                    bestSolution = curSolution;
                            }
                            best = curSolution.getMakeSpan();
                        }
                    }
                }
                Tk *= B;
            }
            //Console.WriteLine("best: " + best);
            return best;
        }

    }
}
