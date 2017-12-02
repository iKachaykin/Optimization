using System;

namespace SimplexMethod
{
    class MainClass
    {
        public static void Main(string[] args)
        {
            Matrix A = (Matrix.RandomMatrix(4, 4) * 20).Floor - Matrix.OnesMatrix(4, 4) * 10;
            Console.WriteLine(Convert.ToString(A) + "\n\n");
            Console.WriteLine(Convert.ToString(A.Reversed) + "\n\n");
            Console.WriteLine(Convert.ToString(A * A.Reversed) + "\n\n");
            Console.ReadKey();
        }
    }
}
