using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace FJSP
{
    class Particle
    {
        public int[] X; //微粒的坐标数组
        public int[] V; //微粒的速度数组
        public int[] xBest;  //微粒的最好位置数组
        public int dim;  //微粒的维数
        public double fit;
        public double fitBest;  //微粒的最好位置适合度

        public Particle(int d)
        {
            dim = d;
            X = new int[dim];
            V = new int[dim];
            xBest = new int[dim]; 
        }
    }
}
