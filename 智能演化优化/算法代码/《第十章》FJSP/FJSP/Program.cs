using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Collections;
namespace FJSP
{
    class Program
    {
        struct Ts
        {
            int x;
            int y;
            public Ts(int a, int b)
            {
                x = a;
                y = b;
            }
        }
        //public static int[] f()
        //{
        //    int[] a = new int[3];
        //    a[0] = a[1] = a[2] = 3;
        //    return a;
        //}
        static void Main(string[] args)
        {
            ShopData sd = new ShopData("D://newpro_1.txt");
            MyPSOAlgorithm myAlgorithm = new MyPSOAlgorithm(sd, 200);
            myAlgorithm.run(50);
            myAlgorithm.bestSolution.print();
            //int[] a = f();
            //Console.WriteLine(a[2]);
       //     //ShopData sd = new ShopData();
       //   //  Console.WriteLine(sd.getOpNum(1));
       ////     sd.print();
       //  //   Console.Read();
       //     int[] x = new int[3];
       //     x[0] = 1;
       //     x[1] = 2;
       //     x[2] = 3;
       //     PSOAlgorithm pa = new PSOAlgorithm(3,2);
       //     pa.setXup(x);
       //     int[] y = pa.getXup();
       //     Console.WriteLine(y.Length);
       //     Console.WriteLine(y[1]);
       //     Random ran = new Random();
       //     int i = ran.Next(1, 10);
       //     Console.WriteLine("&" + i);
       //     int a =(int)(Math.Round(3.1));
       //     Console.WriteLine(a);

       //     Hashtable hs = new Hashtable();
       //     hs.Add(new Ts(1, 2), 1);
       //     hs.Add(new Ts(2, 3), 11);
       //     Ts ts = new Ts(2, 3);
       //     Console.WriteLine(hs[1]);
       //     Solution s = new Solution();
       //     Console.WriteLine( s.getMakeSpan());

            //Random rand = new Random();
            //for (int i = 0; i < 100; i++)
            //    Console.WriteLine(rand.Next(1, 11));
            Console.Read();
        }
    }
}
