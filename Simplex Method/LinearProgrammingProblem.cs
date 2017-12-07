using System;
using System.Collections.Generic;

namespace SimplexMethod
{
    public class LinearProgrammingProblem
    {
        private const double Epsilon = 1E-8;

        private bool maxObjectiveValue, algorithmPrint;
        private bool[] signArray;
        private char variableChar;
        private short[] relationArray;
        private int validCharNum = 10;
        private string validCharString = "pqrstvwxyz";
        private Vector limitationVector, objectiveFunctionCoefficients, defaultSolution;
        private Matrix limitationMatrix;

        public bool Solvability { get; private set; }
        public Vector Solution { get; private set; }

        public LinearProgrammingProblem
        (Vector objectiveFunctionCoefficients, Matrix limitationMatrix,
         Vector limitationVector, bool maxObjectiveValue = true, bool algorithmPrint = false, 
         char variableChar = 'x', Vector defaultSolution = null)
        {
            if (!IsInputDataValid(objectiveFunctionCoefficients, limitationMatrix, limitationVector) || 
                defaultSolution != null && !IsAllowableSolution(defaultSolution))
                throw new ArgumentException();
            this.variableChar = variableChar;
            this.maxObjectiveValue = maxObjectiveValue;
            this.algorithmPrint = algorithmPrint;
            this.objectiveFunctionCoefficients = new Vector(objectiveFunctionCoefficients);
            this.limitationMatrix = new Matrix(limitationMatrix);
            this.limitationVector = new Vector(limitationVector);
            this.defaultSolution = defaultSolution;
            relationArray = new short[limitationMatrix.FirstDimension];
            signArray = new bool[limitationMatrix.SecondDimension];
            for (int i = 0; i < relationArray.Length; i++)
                relationArray[i] = 0;
            for (int i = 0; i < signArray.Length; i++)
                signArray[i] = true;
            Solvability = Solve();
        }

        public LinearProgrammingProblem
        (Vector objectiveFunctionCoefficients, Matrix limitationMatrix, 
         Vector limitationVector, short[] relationArray, 
         bool maxObjectiveValue = true, bool algorithmPrint = false,
         char variableChar = 'x', Vector defaultSolution = null)
        {
            if (!IsInputDataValid(objectiveFunctionCoefficients, limitationMatrix, limitationVector, relationArray) || 
                defaultSolution != null && !IsAllowableSolution(defaultSolution))
                throw new ArgumentException();
            this.variableChar = variableChar;
            this.maxObjectiveValue = maxObjectiveValue;
            this.algorithmPrint = algorithmPrint;
            this.objectiveFunctionCoefficients = new Vector(objectiveFunctionCoefficients);
            this.limitationMatrix = new Matrix(limitationMatrix);
            this.limitationVector = new Vector(limitationVector);
            this.relationArray = (short[])relationArray.Clone();
            this.defaultSolution = defaultSolution;
            signArray = new bool[limitationMatrix.SecondDimension];
            for (int i = 0; i < signArray.Length; i++)
                signArray[i] = true;
            Solvability = Solve();
        }

        public LinearProgrammingProblem
        (Vector objectiveFunctionCoefficients, Matrix limitationMatrix, 
         Vector limitationVector, bool[] signArray, 
         bool maxObjectiveValue = true, bool algorithmPrint = false,
         char variableChar = 'x', Vector defaultSolution = null)
        {
            if (!IsInputDataValid(objectiveFunctionCoefficients, limitationMatrix, limitationVector, null, signArray) ||
                defaultSolution != null && !IsAllowableSolution(defaultSolution))
                throw new ArgumentException();
            this.variableChar = variableChar;
            this.maxObjectiveValue = maxObjectiveValue;
            this.algorithmPrint = algorithmPrint;
            this.objectiveFunctionCoefficients = new Vector(objectiveFunctionCoefficients);
            this.limitationMatrix = new Matrix(limitationMatrix);
            this.limitationVector = new Vector(limitationVector);
            this.signArray = (bool[])signArray.Clone();
            this.defaultSolution = defaultSolution;
            relationArray = new short[limitationMatrix.FirstDimension];
            for (int i = 0; i < relationArray.Length; i++)
                relationArray[i] = 0;
            Solvability = Solve();
        }

        public LinearProgrammingProblem
        (Vector objectiveFunctionCoefficients, Matrix limitationMatrix, 
         Vector limitationVector, short[] relationArray, bool[] signArray, 
         bool maxObjectiveValue = true, bool algorithmPrint = false,
         char variableChar = 'x', Vector defaultSolution = null)
        {
            if (!IsInputDataValid(objectiveFunctionCoefficients, limitationMatrix, limitationVector, relationArray, signArray) ||
                defaultSolution != null && !IsAllowableSolution(defaultSolution))
                throw new ArgumentException();
            this.variableChar = variableChar;
            this.maxObjectiveValue = maxObjectiveValue;
            this.algorithmPrint = algorithmPrint;
            this.objectiveFunctionCoefficients = new Vector(objectiveFunctionCoefficients);
            this.limitationMatrix = new Matrix(limitationMatrix);
            this.limitationVector = new Vector(limitationVector);
            this.signArray = (bool[])signArray.Clone();
            this.relationArray = (short[])relationArray.Clone();
            this.defaultSolution = defaultSolution;
            Solvability = Solve();
        }

        public override string ToString()
        {
            string res = "";
            for (int j = 0; j < objectiveFunctionCoefficients.Dimension; j++)
            {
                res += Convert.ToString(objectiveFunctionCoefficients[j]) + " * " + variableChar + Convert.ToString(j + 1);
                if (j != objectiveFunctionCoefficients.Dimension - 1)
                    res += " + ";
            }
            res += " --> ";
            if (maxObjectiveValue)
                res += "max\n";
            else
                res += "min\n";
            for (int i = 0; i < limitationMatrix.FirstDimension; i++)
            {
                for (int j = 0; j < limitationMatrix.SecondDimension; j++)
                {
                    res += Convert.ToString(limitationMatrix[i, j]) + " * " + variableChar + Convert.ToString(j + 1);
                    if (j != limitationMatrix.SecondDimension - 1)
                        res += " + ";
                }
                if (relationArray[i] == -1)
                    res += " <= ";
                else if (relationArray[i] == 0)
                    res += " = ";
                else
                    res += " >= ";
                res += Convert.ToString(limitationVector[i]) + "\n";
            }
            for (int j = 0; j < signArray.Length; j++)
                if (signArray[j])
                    res += variableChar + Convert.ToString(j + 1) + " >= 0\n";
            return res;
        }

        public bool Canonical
        {
            get
            {
                foreach (short relation in relationArray)
                    if (relation != 0)
                        return false;
                foreach (bool sign in signArray)
                    if (!sign)
                        return false;
                return maxObjectiveValue && limitationMatrix.Rank == limitationMatrix.FirstDimension;
            }
        }

        public LinearProgrammingProblem EqualCanonicalProblem
        {
            get
            {
                Random random = new Random();
                char variableChar = this.variableChar;
                while (variableChar == this.variableChar)
                    variableChar = validCharString[random.Next(validCharNum)];
                bool maxObjectiveValue = true;
                int iTmp = 0;
                Vector limitationVector = new Vector(this.limitationVector);
                Matrix limitationMatrix = new Matrix(this.limitationMatrix);
                short[] relationArray = new short[this.relationArray.Length];
                Vector[] tmpVectorArray;
                List<int> indexesList = new List<int>();
                for (int j = 0; j < this.signArray.Length; j++)
                    if (!this.signArray[j])
                        indexesList.Add(j);
                foreach (int index in indexesList)
                {
                    tmpVectorArray = (Vector[])limitationMatrix.Vectors.Clone();
                    Array.Resize(ref tmpVectorArray, tmpVectorArray.Length + 1);
                    tmpVectorArray[index + iTmp + 1] = -tmpVectorArray[index + iTmp];
                    Array.Copy(limitationMatrix.Vectors, index + iTmp + 1, tmpVectorArray, 
                               index + iTmp + 2, limitationMatrix.SecondDimension - index - iTmp - 1);
                    limitationMatrix = Matrix.UniteVectors(tmpVectorArray);
                    iTmp++;
                }
                for (int i = 0; i < this.relationArray.Length; i++)
                {
                    relationArray[i] = 0;
                    if (this.relationArray[i] != 0)
                    {
                        tmpVectorArray = (Vector[])limitationMatrix.Vectors.Clone();
                        Array.Resize(ref tmpVectorArray, tmpVectorArray.Length + 1);
                        tmpVectorArray[tmpVectorArray.Length - 1] = 
                            -this.relationArray[i] * Vector.UnitVector(limitationMatrix.FirstDimension, i);
                        limitationMatrix = Matrix.UniteVectors(tmpVectorArray);
                    }
                }
                bool[] signArray = new bool[limitationMatrix.SecondDimension];
                for (int i = 0; i < signArray.Length; i++)
                    signArray[i] = true;
                Vector objectiveFunctionCoefficients = new Vector(limitationMatrix.SecondDimension);
                for (int i = 0, newIndex = 0; i < this.objectiveFunctionCoefficients.Dimension; i++, newIndex++)
                {
                    objectiveFunctionCoefficients[newIndex] = this.objectiveFunctionCoefficients[i];
                    if (indexesList.Contains(i))
                    {
                        newIndex++;
                        objectiveFunctionCoefficients[newIndex] = -this.objectiveFunctionCoefficients[i];
                    }
                }
                for (int i = this.objectiveFunctionCoefficients.Dimension + indexesList.Count; i < objectiveFunctionCoefficients.Dimension; i++)
                    objectiveFunctionCoefficients[i] = 0;
                if (!this.maxObjectiveValue)
                    objectiveFunctionCoefficients = -objectiveFunctionCoefficients;
                return new LinearProgrammingProblem(objectiveFunctionCoefficients, 
                                                    limitationMatrix, limitationVector,
                                                    relationArray, signArray,
                                                    maxObjectiveValue, algorithmPrint, variableChar);
            }
        }

        public bool IsAllowableSolution(Vector solution)
        {
            if (solution.Dimension != limitationMatrix.SecondDimension)
                throw new ArgumentException();
            Vector[] vectors = limitationMatrix.T.Vectors;
            double composition = 1;
            for (int i = 0; i < vectors.Length; i++)
            {
                composition = vectors[i] * solution;
                if (relationArray[i] == -1 && composition > limitationVector[i])
                    return false;
                else if (relationArray[i] == 0 && Math.Abs(composition - limitationVector[i]) > Epsilon)
                    return false;
                else if (relationArray[i] == 1 && composition < limitationVector[i])
                    return false;
            }
            for (int j = 0; j < solution.Dimension; j++)
                if (signArray[j] && solution[j] < 0)
                    return false;
            return true;

        }

        private static bool IsInputDataValid
        (Vector objectiveFunctionCoefficients, Matrix limitationMatrix,
         Vector limitationVector, short[] relationArray = null, 
         bool[] signArray = null)
        {
            bool res = objectiveFunctionCoefficients.Dimension == limitationMatrix.SecondDimension &&
                                                    limitationVector.Dimension == limitationMatrix.FirstDimension;
            if (relationArray != null)
            {
                res = res && relationArray.Length == limitationMatrix.FirstDimension;
                for (int i = 0; i < relationArray.Length; i++)
                    res = res && (relationArray[i] == -1 || relationArray[i] == 0 || relationArray[i] == 1);
            }
            if (signArray != null)
                res = res && signArray.Length == limitationMatrix.SecondDimension;
            return res;
        }

        private bool Solve()
        {
            return true;
        }

        private bool SimplexAlgorithmWithDefaultSolution()
        {
            return true;
        }

        private bool ArtificialBasisMethod()
        {
            return true;
        }
    }
}
