using System;

namespace SimplexMethod
{
    class MainClass
    {
        public static void Main(string[] args)
        {
            bool isMax = true;
            bool[] signArray = { true, true, true, false};
            short[] relationArray = { 0, 1, 1, -1, -1 };
            Matrix limitationMatrix = new Matrix(
                new double[,]{{3, 3, -2, 0}, {0, -1, 2, -4}, {5, 0, 2, 0}, {3, -2, 0, 5}, {1, 3, 4, 0}});
            Vector objectiveFunctionCoefficients = new Vector(new double[] {2, 7, -4, 3}),
            limitationVector = new Vector(new double[]{15, 16, 25, 29, 18});
            LinearProgrammingProblem currentProblem = 
                new LinearProgrammingProblem(objectiveFunctionCoefficients, limitationMatrix, 
                                             limitationVector, relationArray, 
                                             signArray, isMax);
            Console.WriteLine(currentProblem);
            Console.WriteLine(currentProblem.EqualCanonicalProblem);
            Console.ReadKey();
        }
    }
}
