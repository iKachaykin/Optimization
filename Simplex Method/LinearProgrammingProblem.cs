using System;
using System.Collections.Generic;

namespace SimplexMethod
{
    public class LinearProgrammingProblem
    {
        private int validCharNum = 10;
        private string validCharString = "pqrstvwxyz";
        private char variableChar;
        private bool isMaxObjectiveValue;
        private Matrix limitationMatrix;
        private Vector limitationVector, objectiveFunctionCoefficients;
        private short[] relationArray;
        private bool[] signArray;

        public LinearProgrammingProblem
        (Vector objectiveFunctionCoefficients, Matrix limitationMatrix, 
         Vector limitationVector, bool isMaxObjectiveValue = true, char variableChar = 'x')
        {
            if (!IsInputDataValid(objectiveFunctionCoefficients, limitationMatrix, limitationVector))
                throw new ArgumentException();
            this.variableChar = variableChar;
            this.isMaxObjectiveValue = isMaxObjectiveValue;
            this.objectiveFunctionCoefficients = new Vector(objectiveFunctionCoefficients);
            this.limitationMatrix = new Matrix(limitationMatrix);
            this.limitationVector = new Vector(limitationVector);
            relationArray = new short[limitationMatrix.FirstDimension];
            signArray = new bool[limitationMatrix.SecondDimension];
            for (int i = 0; i < relationArray.Length; i++)
                relationArray[i] = 0;
            for (int i = 0; i < signArray.Length; i++)
                signArray[i] = true;
        }

        public LinearProgrammingProblem
        (Vector objectiveFunctionCoefficients, Matrix limitationMatrix, 
         Vector limitationVector, short[] relationArray, 
         bool isMaxObjectiveValue = true, char variableChar = 'x')
        {
            if (!IsInputDataValid(objectiveFunctionCoefficients, limitationMatrix, limitationVector, relationArray))
                throw new ArgumentException();
            this.variableChar = variableChar;
            this.isMaxObjectiveValue = isMaxObjectiveValue;
            this.objectiveFunctionCoefficients = new Vector(objectiveFunctionCoefficients);
            this.limitationMatrix = new Matrix(limitationMatrix);
            this.limitationVector = new Vector(limitationVector);
            this.relationArray = (short[])relationArray.Clone();
            signArray = new bool[limitationMatrix.SecondDimension];
            for (int i = 0; i < signArray.Length; i++)
                signArray[i] = true;
        }

        public LinearProgrammingProblem
        (Vector objectiveFunctionCoefficients, Matrix limitationMatrix, 
         Vector limitationVector, bool[] signArray, 
         bool isMaxObjectiveValue = true, char variableChar = 'x')
        {
            if (!IsInputDataValid(objectiveFunctionCoefficients, limitationMatrix, limitationVector, null, signArray))
                throw new ArgumentException();
            this.variableChar = variableChar;
            this.isMaxObjectiveValue = isMaxObjectiveValue;
            this.objectiveFunctionCoefficients = new Vector(objectiveFunctionCoefficients);
            this.limitationMatrix = new Matrix(limitationMatrix);
            this.limitationVector = new Vector(limitationVector);
            this.signArray = (bool[])signArray.Clone();
            relationArray = new short[limitationMatrix.FirstDimension];
            for (int i = 0; i < relationArray.Length; i++)
                relationArray[i] = 0;
        }

        public LinearProgrammingProblem
        (Vector objectiveFunctionCoefficients, Matrix limitationMatrix, 
         Vector limitationVector, short[] relationArray, bool[] signArray, 
         bool isMaxObjectiveValue = true, char variableChar = 'x')
        {
            if (!IsInputDataValid(objectiveFunctionCoefficients, limitationMatrix, limitationVector, relationArray, signArray))
                throw new ArgumentException();
            this.variableChar = variableChar;
            this.isMaxObjectiveValue = isMaxObjectiveValue;
            this.objectiveFunctionCoefficients = new Vector(objectiveFunctionCoefficients);
            this.limitationMatrix = new Matrix(limitationMatrix);
            this.limitationVector = new Vector(limitationVector);
            this.signArray = (bool[])signArray.Clone();
            this.relationArray = (short[])relationArray.Clone();
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
            if (isMaxObjectiveValue)
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
                return isMaxObjectiveValue && limitationMatrix.Rank == limitationMatrix.FirstDimension;
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
                bool isMaxObjectiveValue = true;
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
                if (!this.isMaxObjectiveValue)
                    objectiveFunctionCoefficients = -objectiveFunctionCoefficients;
                return new LinearProgrammingProblem(objectiveFunctionCoefficients, 
                                                    limitationMatrix, limitationVector, 
                                                    relationArray, signArray, 
                                                    isMaxObjectiveValue, variableChar);
            }
        }

        private static bool IsInputDataValid
        (Vector objectiveFunctionCoefficients, Matrix limitationMatrix,
         Vector limitationVector, short[] relationArray = null, bool[] signArray = null)
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
    }
}
