using System;
using System.Collections.Generic;

namespace SimplexMethod
{
    class MainClass
    {
        

        public static void Main(string[] args)
        {
            //bool Max = true;
            //Vector objCoeffs = new Vector(new double[] { 2, 7, -4, 3 }),
            //limVector = new Vector(new double[]{ 15, 16, 25, 29, 18});
            //Matrix limMatrix = new Matrix(new double[,] {{3, 3, -2, 0}, {0, -1, 2, -4}, {5, 0, 2, 0}, 
            //{3, -2, 0, 5}, {1, 3, 4, 0}});
            //bool[] signArr = { true, true, true, false };
            //short[] relationArr = { 0, 1, 1, -1, -1 };

            //bool Max = false;
            //Vector objCoeffs = new Vector(new double[] { 2, 3 }),
            //limVector = new Vector(new double[] { 16, 10});
            //Matrix limMatrix = new Matrix(new double[,] { { 1, 2 }, { 2, 1 } });
            //bool[] signArr = { true, true };
            //short[] relationArr = { -1, 1 };

            //bool Max = true;
            //bool[] signArr = { true, true };
            //short[] relationArr = { -1, -1, -1 };
            //Matrix limMatrix = new Matrix(
            //new double[,]{{-1, 1}, {2, 1}, {1, -1}});
            //Vector objCoeffs = new Vector(new double[] {1, -2}),
            //limVector = new Vector(new double[]{0, 3, 1});

            //bool Max = true;
            //bool[] signArr = { true, true };
            //short[] relationArr = { -1, -1, 1 };
            //Matrix limMatrix = new Matrix(
            //new double[,] { { 1, 1 }, { 3, -2 }, { 5, 3 } });
            //Vector objCoeffs = new Vector(new double[] { 7, 1 }),
            //limVector = new Vector(new double[] { 14, 15, 21 });

            bool Max = true;
            bool[] signArr = { true, true };
            short[] relationArr = { -1, -1, -1 };
            Matrix limMatrix = new Matrix(
            new double[,] { { 10, 8 }, { 5, 10 }, { 6, 12 } });
            Vector objCoeffs = new Vector(new double[] { 14, 18 }),
            limVector = new Vector(new double[] { 168, 180, 144 });


            LinearProgrammingProblem problem =
                new LinearProgrammingProblem(limMatrix, limVector, objCoeffs,
                                             false, Max, relationArr, signArr);
            Console.WriteLine("Linear programming problem:\n{0}", problem);
            problem.Solve();
            Console.WriteLine("Solution: {0}\n", problem.Solution);
            Console.ReadKey();
        }
    }
}
