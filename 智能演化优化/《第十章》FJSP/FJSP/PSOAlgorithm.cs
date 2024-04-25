using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace FJSP
{ 
    class PSOAlgorithm
    {
        Particle[] particle;   //微粒群数组
        int pNum;   //微粒个数
        protected int gBestIndex;  //最好微粒索引
        double wMax;   //最大权重
        double wMin;   //最小权重
        double c1;   //加速度系数1
        double c2;   //加速度系数2
        int[] xUp;   //微粒坐标上界数组
        int[] xDown; //微粒坐标下界数组
        int[] vMax;  //微粒最大速度数组

        public PSOAlgorithm() { }
        public PSOAlgorithm(int dim, int num)
        {
            particle = new Particle[num];
            for (int i = 0; i < num; i++)
                particle[i] = new Particle(dim);
            pNum = num;
            gBestIndex = 0;
            xUp = new int[dim];
            xDown = new int[dim];
            vMax = new int[dim];
            wMax = 1.2;
            wMin = 0.4;
            c1 = 2;
            c2 = 2; 
        }
        void Initialize()
        {
            if (particle == null) 
                return;
            gBestIndex = 0;
            Random ra = new Random();
            for (int i = 0; i < pNum; i++)
            {
                for (int j = 0; j < particle[i].dim; j++)
                {
                    //Console.WriteLine("d: " + xDown[j] + "u: " + xUp[j]);
                    particle[i].X[j] = ra.Next(xDown[j], xUp[j] + 1);
                    //Console.WriteLine("d:" + xDown[j] + " u:" + xUp[j]);
                    //Console.WriteLine(j + " " + particle[i].X[j]);
                    particle[i].xBest[j] = particle[i].X[j];
                    particle[i].V[j] = ra.Next(-vMax[j], vMax[j] + 1);
                    //particle[i].V[j] = ra.Next(-4, 5);
                }
                particle[i].fit = getFit(particle[i]);
                particle[i].fitBest = particle[i].fit;
                if (particle[i].fit > particle[gBestIndex].fit) 
                    gBestIndex = i; 
            }
        }

        public void particleFly(int iter, int iterm)
        {
            if (particle == null) return;
            double w = wMax - (wMax - wMin) * iter / iterm;
            int i, j;
            Random ra = new Random();
            for (i = 0; i < pNum; i++)
            {
                for (j = 0; j < particle[i].dim; j++)
                {
                    particle[i].V[j] = (int)Math.Round(w * particle[i].V[j]
                                        + c1 * ra.NextDouble() * (particle[i].xBest[j] - particle[i].X[j])
                                        + c2 * ra.NextDouble() * (particle[gBestIndex].xBest[j] - particle[i].X[j]));

                }
                for (j = 0; j < particle[i].dim; j++)
                {
                    if (particle[i].V[j] > vMax[j]) 
                        particle[i].V[j] = vMax[j];
                    if (particle[i].V[j] < -vMax[j])
                        particle[i].V[j] = -vMax[j]; 
                }
                for (j = 0; j < particle[i].dim; j++)
                {
                    particle[i].X[j] += particle[i].V[j];
                    if (particle[i].X[j] > xUp[j])
                        //particle[i].X[j] = xUp[j];
                        particle[i].X[j] = xDown[j] + ra.Next(0, 2);
                    if (particle[i].X[j] < xDown[j])
                        particle[i].X[j] = xDown[j]; 
                    //+new Random().Next(0, 2);

                } 
            }
            calFit();
            for (i = 0; i < pNum; i++)
            {
                //Console.WriteLine("^ " +particle[i].fit);
                if (particle[i].fit <= particle[i].fitBest)
                {
                    particle[i].fitBest = particle[i].fit;
                    for (j = 0; j < particle[i].dim; j++)
                        particle[i].xBest[j] = particle[i].X[j];
                }
            }

            gBestIndex = 0;
            for (i = 0; i < pNum; i++)
            {
                if (particle[i].fitBest <= particle[gBestIndex].fitBest && i != gBestIndex)
                    gBestIndex = i;
            }
            //Console.WriteLine("gBestIndex: " + gBestIndex);
        }

        public void run(int n)
        {
            Initialize();
            for (int i = 0; i < n; i++)
            {
                particleFly(i + 1, n);
                //Console.Write(gBestIndex);
                for (int j = 0; j < particle[gBestIndex].xBest.Length; j++)
                    Console.Write(particle[gBestIndex].xBest[j] + " ");
                Console.WriteLine();
                Console.WriteLine(particle[gBestIndex].fitBest);
                //Console.Read();

            }
        }
        public virtual double getFit(Particle p)
        {
            return 0;
        }

        void calFit()
        {
            if (particle == null)
                return;
            for (int i = 0; i < pNum; i++)
                particle[i].fit = getFit(particle[i]);
        }
        public void setXup(int[] xup)
        {
            xUp = xup;
        }
        public int[] getXup()
        {
            return xUp;
        }
        public void setXdown(int[] xdown)
        {
            xDown = xdown;
        }
        public void setVmax(int[] vmax)
        {
            vMax = vmax;
        }
        public void setWmax(double w)
        {
            wMax = w;
        }
        public void setWmin(double w)
        {
            wMin = w;
        }
        public void setC1(double c)
        {
            c1 = c;
        }
        public void setC2(double c)
        {
            c2 = c;
        }

    }
}
