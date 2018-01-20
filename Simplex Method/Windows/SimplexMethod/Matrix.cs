using System;

namespace SimplexMethod
{
    public class Matrix : ICloneable
    {
        private const double Epsilon = 1E-8;
        private double[,] matrix;

        public int Dimension { get { return FirstDimension * SecondDimension; } }
        public int FirstDimension { get; private set; }
        public int SecondDimension { get; private set; }

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

        public object Clone() { return new Matrix(this); }

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

        public static Matrix operator +(Matrix matrix) { return new Matrix(matrix); }

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
            if ((object)left == null && (object)right == null)
                return true;
            if ((object)left == null || (object)right == null)
                return false;
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
            if ((object)left == null && (object)right == null)
                return false;
            if ((object)left == null || (object)right == null)
                return true;
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

        public static explicit operator double(Matrix A)
        {
            if (A.Dimension != 1)
                throw new InvalidOperationException();
            return A.matrix[0, 0];
        }

        public static explicit operator Matrix(double c) { return new Matrix(1, 1, c); }

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
                        Convert.ToString(Convert.ToInt32(Math.Round(matrix[i, j]))) + "\t\t\t" :
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
                        absMaxInRowIndex = thisCopy.FindAbsMaxInRowPosition(mainDiagIndex, mainDiagIndex, dimension);
                        if (Math.Abs(thisCopy.matrix[mainDiagIndex, absMaxInRowIndex]) < Epsilon)
                            return 0;
                        transp++;
                        thisCopy.SwapColumns(absMaxInRowIndex, mainDiagIndex);
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
                Matrix res = E(dimension), thisCopy = new Matrix(this);
                for (int mainDiagIndex = 0, absMaxInColIndex; mainDiagIndex < dimension; mainDiagIndex++)
                {
                    if (Math.Abs(thisCopy.matrix[mainDiagIndex, mainDiagIndex]) < Epsilon)
                    {
                        absMaxInColIndex = thisCopy.FindAbsMaxInColumnPosition(mainDiagIndex, mainDiagIndex, dimension);
                        if (Math.Abs(thisCopy.matrix[absMaxInColIndex, mainDiagIndex]) < Epsilon)
                            throw new InvalidOperationException();
                        thisCopy.SwapRows(absMaxInColIndex, mainDiagIndex);
                        res.SwapRows(absMaxInColIndex, mainDiagIndex);
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

        public int Rank
        {
            get
            {
                int res = 0;
                double tmp = 0;
                Matrix thisCopy = new Matrix(this);
                for (int mainDiagIndex = 0, absMaxInRowIndex, lastRowIndex = FirstDimension - 1;
                     mainDiagIndex < Math.Min(lastRowIndex + 1, SecondDimension); mainDiagIndex++)
                {
                    if (Math.Abs(thisCopy[mainDiagIndex, mainDiagIndex]) < Epsilon)
                    {
                        absMaxInRowIndex = thisCopy.FindAbsMaxInRowPosition(mainDiagIndex, mainDiagIndex, SecondDimension);
                        if (Math.Abs(thisCopy.matrix[mainDiagIndex, absMaxInRowIndex]) < Epsilon)
                        {
                            thisCopy.SwapRows(mainDiagIndex, lastRowIndex);
                            lastRowIndex--;
                            mainDiagIndex--;
                            continue;
                        }
                        else
                            thisCopy.SwapColumns(mainDiagIndex, absMaxInRowIndex);
                    }
                    for (int i = mainDiagIndex + 1; i < FirstDimension; i++)
                    {
                        tmp = thisCopy.matrix[i, mainDiagIndex];
                        for (int j = mainDiagIndex; j < SecondDimension; j++)
                            thisCopy.matrix[i, j] -= thisCopy.matrix[mainDiagIndex, j] / thisCopy.matrix[mainDiagIndex, mainDiagIndex] * tmp;
                    }
                    res = mainDiagIndex + 1;
                }
                return res;
            }
        }

        public Vector GetRow(int index)
        {
            if (index < 0 || index >= FirstDimension)
                throw new IndexOutOfRangeException();
            Vector res = new Vector(SecondDimension);
            for (int j = 0; j < SecondDimension; j++)
                res[j] = matrix[index, j];
            return res;
        }

        public Vector GetColumn(int index)
        {
            if (index < 0 || index >= SecondDimension)
                throw new IndexOutOfRangeException();
            Vector res = new Vector(FirstDimension);
            for (int i = 0; i < FirstDimension; i++)
                res[i] = matrix[i, index];
            return res;
        }

        public Matrix GetRows(int start, int end = -1)
        {
            if (end == -1)
                end = FirstDimension;
            if (start < 0 || start >= FirstDimension || end <= 0 ||
                end > FirstDimension || start >= end)
                throw new IndexOutOfRangeException();
            Matrix res = new Matrix(end - start, SecondDimension);
            for (int iFirst = start, iSecond = 0; iFirst < end; iFirst++, iSecond++)
                for (int j = 0; j < SecondDimension; j++)
                    res[iSecond, j] = matrix[iFirst, j];
            return res;
        }

        public Matrix GetColumns(int start, int end = -1)
        {
            if (end == -1)
                end = SecondDimension;
            if (start < 0 || start >= SecondDimension || end <= 0 ||
                end > SecondDimension || start >= end)
                throw new IndexOutOfRangeException();
            Matrix res = new Matrix(FirstDimension, end - start);
            for (int jFirst = start, jSecond = 0; jFirst < end; jFirst++, jSecond++)
                for (int i = 0; i < FirstDimension; i++)
                    res[i, jSecond] = matrix[i, jFirst];
            return res;
        }

        public static Matrix NullMatrix(int FirstDimension = 2, int SecondDimension = 2)
        {
            if (FirstDimension <= 0 || SecondDimension <= 0)
                throw new InvalidOperationException();
            return new Matrix(FirstDimension, SecondDimension, 0.0);
        }

        public static Matrix OnesMatrix(int FirstDimension = 2, int SecondDimension = 2)
        {
            if (FirstDimension <= 0 || SecondDimension <= 0)
                throw new InvalidOperationException();
            return new Matrix(FirstDimension, SecondDimension, 1.0);
        }

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

        public static Matrix E(int dimension)
        {
            Matrix res = new Matrix(dimension, dimension, 0);
            for (int i = 0; i < dimension; i++)
                res.matrix[i, i] = 1;
            return res;
        }

        public static Matrix UniteVectors(params Vector[] vectors) { return new Matrix(vectors); }

        private static void Swap(ref double a, ref double b) 
        {
            double tmp = a;
            a = b;
            b = tmp;
        }

        private void SwapColumns(int firstColumnIndex, int secondColumnIndex)
        {
            if (firstColumnIndex < 0 || firstColumnIndex >= matrix.GetLength(1) ||
                secondColumnIndex < 0 || secondColumnIndex >= matrix.GetLength(1))
                throw new IndexOutOfRangeException();
            for (int i = 0; i < matrix.GetLength(0); i++)
                Swap(ref matrix[i, firstColumnIndex], ref matrix[i, secondColumnIndex]);
        }

        private void SwapRows(int firstRowIndex, int secondRowIndex)
        {
            if (firstRowIndex < 0 || firstRowIndex >= matrix.GetLength(0) ||
                secondRowIndex < 0 || secondRowIndex >= matrix.GetLength(0))
                throw new IndexOutOfRangeException();
            for (int j = 0; j < matrix.GetLength(1); j++)
                Swap(ref matrix[firstRowIndex, j], ref matrix[secondRowIndex, j]);
        }

        private int FindAbsMaxInColumnPosition(int columnIndex, int start, int end)
        {
            if (columnIndex < 0 || columnIndex >= matrix.GetLength(1) ||
                start < 0 || start >= matrix.GetLength(0) || start >= end ||
                end <= 0 || end > matrix.GetLength(0))
                throw new IndexOutOfRangeException();
            int res = start;
            for (int i = start + 1; i < end; i++)
                if (Math.Abs(matrix[i, columnIndex]) > Math.Abs(matrix[res, columnIndex]))
                    res = i;
            return res;
        }

        private int FindAbsMaxInRowPosition(int rowIndex, int start, int end)
        {
            if (rowIndex < 0 || rowIndex >= matrix.GetLength(0) ||
               start < 0 || start >= matrix.GetLength(1) || start >= end ||
               end <= 0 || end > matrix.GetLength(1))
                throw new IndexOutOfRangeException();
            int res = start;
            for (int j = start + 1; j < end; j++)
                if (Math.Abs(matrix[rowIndex, j]) > Math.Abs(matrix[rowIndex, res]))
                    res = j;
            return res;
        }
    }
}
