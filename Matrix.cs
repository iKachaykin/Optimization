using System;
namespace SimplexMethod
{
    public class Matrix : ICloneable
    {
        protected const double Epsilon = 1E-8;
        protected double[,] matrix;
        public int FirstDimension { get; private set; }
        public int SecondDimension { get; private set; }
        public virtual int Dimension { get => FirstDimension * SecondDimension; }

        public Matrix(int FirstDimension = 2, int SecondDimension = 2, double defaultVal = 0.0)
        {
            if (FirstDimension <= 0 || SecondDimension <= 0)
                throw new InvalidOperationException();
            this.FirstDimension = FirstDimension;
            this.SecondDimension = SecondDimension;
            matrix = new double[FirstDimension, SecondDimension];
            for (int i = 0; i < FirstDimension; i++)
                for (int j = 0; j < SecondDimension; j++)
                    matrix[i, j] = defaultVal;
        }

        public Matrix(Matrix other)
        {
            FirstDimension = other.FirstDimension;
            SecondDimension = other.SecondDimension;
            matrix = (double[,])other.matrix.Clone();
        }

        public Matrix(double[,] matrix)
        {
            FirstDimension = matrix.GetLength(0);
            SecondDimension = matrix.GetLength(1);
            this.matrix = (double[,])matrix.Clone();
        }

        public Matrix(params Vector[] vectors)
        {
            if (!Vector.DimensionEquality(vectors))
                throw new InvalidOperationException();
            FirstDimension = vectors[0].Dimension;
            SecondDimension = vectors.Length;
            matrix = new double[FirstDimension, SecondDimension];
            for (int i = 0; i < FirstDimension; i++)
                for (int j = 0; j < SecondDimension; j++)
                    matrix[i, j] = vectors[j][i];
            
        }

        public object Clone() => new Matrix(this);

        public double this[int i, int j]
        {
            get
            {
                if (i < 0 || i >= FirstDimension || j < 0 || j >= SecondDimension)
                    throw new IndexOutOfRangeException();
                return matrix[i, j];
            }
            set
            {
                if (i < 0 || i >= FirstDimension || j < 0 || j >= SecondDimension)
                    throw new IndexOutOfRangeException();
                matrix[i, j] = value;
            }
        }

        public static Matrix operator +(Matrix matrix) => new Matrix(matrix);

        public static Matrix operator -(Matrix matrix)
        {
            Matrix res = new Matrix(matrix);
            for (int i = 0; i < res.FirstDimension; i++)
                for (int j = 0; j < res.SecondDimension; j++)
                    res.matrix[i, j] = -res.matrix[i, j];
            return res;
        }

        public static Matrix operator +(Matrix left, Matrix right)
        {
            if (left.FirstDimension != right.FirstDimension ||
                left.SecondDimension != right.SecondDimension)
                throw new InvalidOperationException();
            Matrix res = new Matrix(left);
            for (int i = 0; i < res.FirstDimension; i++)
                for (int j = 0; j < res.SecondDimension; j++)
                    res.matrix[i, j] += right.matrix[i, j];
            return res;
        }

        public static Matrix operator -(Matrix left, Matrix right)
        {
            if (left.FirstDimension != right.FirstDimension ||
                left.SecondDimension != right.SecondDimension)
                throw new InvalidOperationException();
            Matrix res = new Matrix(left);
            for (int i = 0; i < res.FirstDimension; i++)
                for (int j = 0; j < res.SecondDimension; j++)
                    res.matrix[i, j] -= right.matrix[i, j];
            return res;
        }

        public static Matrix operator *(double c, Matrix A)
        {
            Matrix res = new Matrix(A);
            for (int i = 0; i < res.FirstDimension; i++)
                for (int j = 0; j < res.SecondDimension; j++)
                    res.matrix[i, j] *= c;
            return res;
        }

        public static Matrix operator *(Matrix A, double c)
        {
            Matrix res = new Matrix(A);
            for (int i = 0; i < res.FirstDimension; i++)
                for (int j = 0; j < res.SecondDimension; j++)
                    res.matrix[i, j] *= c;
            return res;
        }

        public static Vector operator *(Matrix A, Vector v)
        {
            if (A.SecondDimension != v.Dimension)
                throw new InvalidOperationException();
            Vector res = new Vector(A.FirstDimension);
            for (int i = 0; i < A.FirstDimension; i++)
                for (int j = 0; j < A.SecondDimension; j++)
                    res[i] += A.matrix[i, j] * v[j];
            return res;
        }

        public static Vector operator *(Vector v, Matrix A)
        {
            if (A.FirstDimension != v.Dimension)
                throw new InvalidOperationException();
            Vector res = new Vector(A.SecondDimension);
            for (int j = 0; j < A.SecondDimension; j++)
                for (int i = 0; i < A.FirstDimension; i++)
                    res[j] = A.matrix[i, j] * v[i];
            return res;
        }

        public static Matrix operator *(Matrix left, Matrix right)
        {
            if (left.SecondDimension != right.FirstDimension)
                throw new InvalidOperationException();
            Matrix res = new Matrix(left.FirstDimension, right.SecondDimension);
            for (int i = 0; i < res.FirstDimension; i++)
                for (int j = 0; j < res.SecondDimension; j++)
                    for (int k = 0; k < left.SecondDimension; k++)
                        res.matrix[i, j] += left.matrix[i, k] * right.matrix[k, j];
            return res;
        }

        public static Matrix operator /(Matrix A, double c)
        {
            Matrix res = new Matrix(A);
            for (int i = 0; i < res.FirstDimension; i++)
                for (int j = 0; j < res.SecondDimension; j++)
                    res.matrix[i, j] /= c;
            return res;
        }

        public static bool operator ==(Matrix left, Matrix right)
        {
            if (left.FirstDimension != right.FirstDimension ||
                left.SecondDimension != right.SecondDimension)
                return false;
            for (int i = 0; i < left.FirstDimension; i++)
                for (int j = 0; j < right.SecondDimension; j++)
                    if (Math.Abs(left.matrix[i, j] - right.matrix[i, j]) > Epsilon)
                        return false;
            return true;
        }

        public static bool operator !=(Matrix left, Matrix right)
        {
            if (left.FirstDimension != right.FirstDimension ||
                left.SecondDimension != right.SecondDimension)
                return true;
            for (int i = 0; i < left.FirstDimension; i++)
                for (int j = 0; j < right.SecondDimension; j++)
                    if (Math.Abs(left.matrix[i, j] - right.matrix[i, j]) > Epsilon)
                        return true;
            return false;
        }

        public static explicit operator Vector(Matrix A)
        {
            if (A.FirstDimension == 1)
            {
                Vector res = new Vector(A.SecondDimension);
                for (int i = 0; i < res.Dimension; i++)
                    res[i] = A.matrix[0, i];
                return res;
            }
            else if (A.SecondDimension == 1)
            {
                Vector res = new Vector(A.FirstDimension);
                for (int i = 0; i < res.Dimension; i++)
                    res[i] = A.matrix[i, 0];
                return res;
            }
            else throw new InvalidOperationException();
        }

        public static explicit operator double(Matrix A) =>
        A.Dimension == 1 ? A.matrix[0, 0] : throw new InvalidOperationException();

        public static explicit operator Matrix(double c) => new Matrix(1, 1, c);

        public override bool Equals(object obj)
        {
            if (obj == null || GetType() != obj.GetType())
                return false;
            Matrix A = (Matrix)obj;
            return this == A;
        }

        public override int GetHashCode() => Convert.ToInt32(EuclidNorm);

        public override string ToString()
        {
            double delt = 0.0;
            String res = "";
            for (int i = 0; i < FirstDimension; i++)
            {
                if (i == 0)
                    res += FirstDimension == 1 ? "(" : "/";
                else if (i == FirstDimension - 1)
                    res += "\\";
                else
                    res += "|";
                for (int j = 0; j < SecondDimension; j++)
                {
                    delt = Math.Abs(matrix[i, j] - Math.Round(matrix[i, j]));
                    res += delt < Epsilon ?
                        Convert.ToString(Convert.ToInt32(Math.Round(matrix[i, j]))) + "\t\t\t":
                               Convert.ToString(matrix[i, j]) + "\t";
                }
                if (i == 0)
                    res += FirstDimension == 1 ? ")\n" : "\\\n";
                else if (i == FirstDimension - 1)
                    res += "/\n";
                else
                    res += "|\n";
            }
            return res;
        }

        public Matrix T
        {
            get
            {
                Matrix res = new Matrix(SecondDimension, FirstDimension);
                for (int i = 0; i < FirstDimension; i++)
                    for (int j = 0; j < SecondDimension; j++)
                        res.matrix[j, i] = matrix[i, j];
                return res;
            }
        }

        public double EuclidNorm
        {
            get
            {
                double res = 0.0;
                for (int i = 0; i < FirstDimension; i++)
                    for (int j = 0; j < SecondDimension; j++)
                        res += matrix[i, j] * matrix[i, j];
                return Math.Sqrt(res);
            }
        }

        public Vector[] Vectors
        {
            get
            {
                Vector[] res = new Vector[SecondDimension];
                for (int j = 0; j < SecondDimension; j++)
                {
                    res[j] = new Vector(FirstDimension);
                    for (int i = 0; i < FirstDimension; i++)
                        res[j][i] = matrix[i, j];
                }
                return res;
            }
        }

        public Matrix Floor
        {
            get
            {
                Matrix res = new Matrix(this);
                for (int i = 0; i < res.FirstDimension; i++)
                    for (int j = 0; j < res.SecondDimension; j++)
                        res.matrix[i, j] = Math.Floor(res.matrix[i, j]);
                return res;
            }
        }

        public Matrix Ceil
        {
            get
            {
                Matrix res = new Matrix(this);
                for (int i = 0; i < res.FirstDimension; i++)
                    for (int j = 0; j < res.SecondDimension; j++)
                        res.matrix[i, j] = Math.Ceiling(res.matrix[i, j]);
                return res;
            }
        }

        public double Determinant
        {
            get
            {
                if (FirstDimension != SecondDimension)
                    throw new InvalidOperationException();
                int dimension = FirstDimension;
                double det = 1, tmp = 0;
                int transp = 0;
                Matrix thisCopy = new Matrix(this);
                for (int mainDiagIndex = 0, absMaxInRowIndex; mainDiagIndex < dimension; mainDiagIndex++)
                {
                    if (Math.Abs(thisCopy.matrix[mainDiagIndex, mainDiagIndex]) < Epsilon)
                    {
                        absMaxInRowIndex = mainDiagIndex;
                        for (int j = mainDiagIndex + 1; j < dimension; j++)
                        {
                            if (Math.Abs(thisCopy.matrix[mainDiagIndex, j]) > Math.Abs(thisCopy.matrix[mainDiagIndex, absMaxInRowIndex]))
                                absMaxInRowIndex = j;
                        }
                        if (Math.Abs(thisCopy.matrix[mainDiagIndex, absMaxInRowIndex]) < Epsilon)
                            return 0;
                        transp++;
                        for (int i = 0; i < dimension; i++)
                        {
                            tmp = thisCopy.matrix[i, absMaxInRowIndex];
                            thisCopy.matrix[i, absMaxInRowIndex] = thisCopy.matrix[i, mainDiagIndex];
                            thisCopy.matrix[i, mainDiagIndex] = tmp;
                        }
                    }
                    for (int i = mainDiagIndex + 1; i < dimension; i++)
                    {
                        tmp = thisCopy.matrix[i, mainDiagIndex];
                        for (int j = mainDiagIndex; j < dimension; j++)
                            thisCopy.matrix[i, j] -= thisCopy.matrix[mainDiagIndex, j] / thisCopy.matrix[mainDiagIndex, mainDiagIndex] * tmp;
                    }
                }
                for (int mainDiagIndex = 0; mainDiagIndex < dimension; mainDiagIndex++)
                    det *= thisCopy.matrix[mainDiagIndex, mainDiagIndex];
                det *= Math.Pow(-1, transp);
                return det;
            }
        }

        public Matrix Reversed
        {
            get
            {
                if (FirstDimension != SecondDimension)
                    throw new InvalidOperationException();
                int dimension = FirstDimension;
                double tmp = 0;
                Matrix res = new Matrix(dimension, dimension), thisCopy = new Matrix(this);
                for (int i = 0; i < dimension; i++)
                    res.matrix[i, i] = 1;
                for (int mainDiagIndex = 0, absMaxInColIndex; mainDiagIndex < dimension; mainDiagIndex++)
                {
                    if (Math.Abs(thisCopy.matrix[mainDiagIndex, mainDiagIndex]) < Epsilon)
                    {
                        absMaxInColIndex = mainDiagIndex;
                        for (int i = mainDiagIndex + 1; i < dimension; i++)
                        {
                            if (Math.Abs(thisCopy.matrix[i, mainDiagIndex]) > Math.Abs(thisCopy.matrix[absMaxInColIndex, mainDiagIndex]))
                                absMaxInColIndex = i;
                        }
                        if (Math.Abs(thisCopy.matrix[absMaxInColIndex, mainDiagIndex]) < Epsilon)
                            throw new InvalidOperationException();
                        for (int j = 0; j < dimension; j++)
                        {
                            tmp = thisCopy.matrix[absMaxInColIndex, j];
                            thisCopy.matrix[absMaxInColIndex, j] = thisCopy.matrix[mainDiagIndex, j];
                            thisCopy.matrix[mainDiagIndex, j] = tmp;
                            tmp = res.matrix[absMaxInColIndex, j];
                            res.matrix[absMaxInColIndex, j] = res.matrix[mainDiagIndex, j];
                            res.matrix[mainDiagIndex, j] = tmp;
                        }
                    }
                    for (int i = mainDiagIndex + 1; i < dimension; i++)
                    {
                        tmp = thisCopy.matrix[i, mainDiagIndex];
                        for (int j = mainDiagIndex; j < dimension; j++)
                            thisCopy.matrix[i, j] -= thisCopy.matrix[mainDiagIndex, j] / thisCopy.matrix[mainDiagIndex, mainDiagIndex] * tmp;
                        for (int j = 0; j < dimension; j++)
                            res.matrix[i, j] -= res.matrix[mainDiagIndex, j] / thisCopy.matrix[mainDiagIndex, mainDiagIndex] * tmp;
                    }
                }
                for (int mainDiagIndex = dimension - 1; mainDiagIndex >= 0; mainDiagIndex--)
                {
                    for (int i = mainDiagIndex - 1; i >= 0; i--)
                    {
                        tmp = thisCopy.matrix[i, mainDiagIndex];
                        for (int j = 0; j < dimension; j++)
                        {
                            thisCopy.matrix[i, j] -= thisCopy.matrix[mainDiagIndex, j] / thisCopy.matrix[mainDiagIndex, mainDiagIndex] * tmp;
                            res.matrix[i, j] -= res.matrix[mainDiagIndex, j] / thisCopy.matrix[mainDiagIndex, mainDiagIndex] * tmp;
                        }
                    }
                }
                for (int i = 0; i < dimension; i++)
                {
                    tmp = thisCopy.matrix[i, i];
                    for (int j = 0; j < dimension; j++)
                    {
                        thisCopy.matrix[i, j] /= tmp;
                        res.matrix[i, j] /= tmp;
                    }
                }
                return res;
                
            }
        }

        public static Matrix NullMatrix(int FirstDimension = 2, int SecondDimension = 2) =>
        FirstDimension > 0 && SecondDimension > 0 ? new Matrix(FirstDimension, SecondDimension, 0.0) :
        throw new InvalidOperationException();

        public static Matrix OnesMatrix(int FirstDimension = 2, int SecondDimension = 2) =>
        FirstDimension > 0 && SecondDimension > 0 ? new Matrix(FirstDimension, SecondDimension, 1.0) :
        throw new InvalidOperationException();

        public static Matrix RandomMatrix(int FirstDimension = 2, int SecondDimension = 2)
        {
            Random random = new Random();
            Matrix res = new Matrix(FirstDimension, SecondDimension);
            for (int i = 0; i < FirstDimension; i++)
                for (int j = 0; j < SecondDimension; j++)
                    res.matrix[i, j] = random.NextDouble();
            return res;
        }

        public static bool DimensionEquality(params Matrix[] mArray)
        {
            int firstDim = mArray[0].FirstDimension, secondDim = mArray[0].SecondDimension;
            for (int i = 1; i < mArray.Length; i++)
                if (mArray[i].FirstDimension != firstDim || mArray[i].SecondDimension != secondDim)
                    return false;
            return true;
        }
    }
}
