using System;
using System.Collections.Generic;

namespace SimplexMethod
{
    public class LinearProgrammingProblem : ICloneable
    {
        private const double    Epsilon = 1E-8;
        private int             validCharNum = 10;
        private string          validCharString = "pqrstvwxyz";

        private bool            algorithmPrint;
        private Vector[]        defaultBasis;
        private int[]           defaultBasisIndexes;
        private Vector          defaultBasisSolution;
        private Matrix          limitationMatrix;
        private Vector          limitationVector;
        private bool            maxObjectiveValue;
        private Vector          objectiveFunctionCoefficients;
        private short[]         relationArray;
        private bool[]          signArray;
        private Vector          solution;
        private char            variableChar;

        public short Solvability { get; private set; }

        public LinearProgrammingProblem
        (Matrix limitationMatrix, Vector limitationVector, 
         Vector objectiveFunctionCoefficients, bool algorithmPrint = false, 
         bool maxObjectiveValue = true, short[] relationArray = null, 
         bool[] signArray = null, char variableChar = 'x')
        {
            if (!InputDataValid(limitationMatrix, limitationVector, 
                                objectiveFunctionCoefficients,relationArray, signArray))
                throw new ArgumentException("The dimensions/lengths of arguments have to correspond to each other!");
            this.algorithmPrint                = algorithmPrint;
            defaultBasis                       = null;
            defaultBasisIndexes                = null;
            defaultBasisSolution               = null;
            this.limitationMatrix              = new Matrix(limitationMatrix);
            this.limitationVector              = new Vector(limitationVector);
            this.maxObjectiveValue             = maxObjectiveValue;
            this.objectiveFunctionCoefficients = new Vector(objectiveFunctionCoefficients);
            solution                           = null;
            Solvability                        = -1;
            this.variableChar                  = variableChar;
            if (relationArray == null)
            {
                this.relationArray = new short[limitationMatrix.FirstDimension];
                for (int i = 0; i < this.relationArray.Length; i++)
                    this.relationArray[i] = 0;
            }
            else
                this.relationArray = (short[])relationArray.Clone();
            if (signArray == null)
            {
                this.signArray = new bool[limitationMatrix.SecondDimension];
                for (int i = 0; i < this.signArray.Length; i++)
                    this.signArray[i] = true;
            }
            else
                this.signArray = (bool[])signArray.Clone();
        }

        public LinearProgrammingProblem(LinearProgrammingProblem other)
        {
            algorithmPrint                = other.algorithmPrint;
            defaultBasis                  = other.defaultBasis;
            defaultBasisIndexes           = (int[])other.defaultBasisIndexes.Clone();
            defaultBasisSolution          = other.defaultBasisSolution;
            limitationMatrix              = other.limitationMatrix;
            limitationVector              = other.limitationVector; 
            maxObjectiveValue             = other.maxObjectiveValue;
            objectiveFunctionCoefficients = other.objectiveFunctionCoefficients;
            relationArray                 = (short[])other.relationArray.Clone();
            signArray                     = (bool[])other.signArray.Clone();
            solution                      = other.solution;
            Solvability                   = other.Solvability;
            variableChar                  = other.variableChar;
        }

        public object Clone() => new LinearProgrammingProblem(this);

        public override string ToString()
        {
            string res = "";
            for (int j = 0; j < objectiveFunctionCoefficients.Dimension; j++)
            {
                res += Convert.ToString(objectiveFunctionCoefficients[j]) + " * " 
                              + variableChar + Convert.ToString(j + 1);
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
                    res += Convert.ToString(limitationMatrix[i, j]) + " * " 
                                  + variableChar + Convert.ToString(j + 1);
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
                return 
                    maxObjectiveValue && 
                    limitationMatrix.Rank == limitationMatrix.FirstDimension && 
                    limitationMatrix.FirstDimension != limitationMatrix.SecondDimension;
            }
        }

        public LinearProgrammingProblem EqualCanonicalProblem
        {
            get
            {
                List<int> indexesList   = new List<int>();
                int indexTmp            = 0;
                Matrix limitationMatrix = new Matrix(this.limitationMatrix);
                Vector limitationVector = new Vector(this.limitationVector);
                bool maxObjectiveValue  = true;
                Random random           = new Random();
                short[] relationArray   = new short[this.relationArray.Length];
                Vector[] tmpVectorArray;
                char variableChar       = this.variableChar;
                while (variableChar == this.variableChar)
                    variableChar = validCharString[random.Next(validCharNum)];
                for (int j = 0; j < this.signArray.Length; j++)
                    if (!this.signArray[j])
                        indexesList.Add(j);
                foreach (int index in indexesList)
                {
                    tmpVectorArray = (Vector[])limitationMatrix.Vectors.Clone();
                    Array.Resize(ref tmpVectorArray, tmpVectorArray.Length + 1);
                    tmpVectorArray[index + indexTmp + 1] = -tmpVectorArray[index + indexTmp];
                    Array.Copy(limitationMatrix.Vectors, index + indexTmp + 1, tmpVectorArray, 
                               index + indexTmp + 2, limitationMatrix.SecondDimension - index - indexTmp - 1);
                    limitationMatrix = Matrix.UniteVectors(tmpVectorArray);
                    indexTmp++;
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
                return new LinearProgrammingProblem(limitationMatrix, limitationVector, 
                                                    objectiveFunctionCoefficients, 
                                                    algorithmPrint, maxObjectiveValue,
                                                    relationArray, signArray, variableChar);
            }
        }

        public Vector Solution { get => solution; }

        public bool AllowableSolution(Vector solution)
        {
            if (solution.Dimension != limitationMatrix.SecondDimension)
                throw new ArgumentException("basis.Length must be equal to limitationMatrix.FirstDimension!", nameof(solution));
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

        public Vector AllowableBasis(int[] basis)
        {
            if (!Canonical)
                throw new InvalidOperationException("For noncanonical problems basis doesn't exist!");
            if (basis.Length != limitationMatrix.FirstDimension)
                throw new ArgumentException("basis.Length must be equal to limitationMatrix.FirstDimension!", nameof(basis));
            int[] basisCopy = (int[])basis.Clone();
            Array.Sort(basisCopy);
            for (int i = 0; i < basisCopy.Length - 1; i++)
                if (basisCopy[i] == basisCopy[i + 1] || basisCopy[i] < 0 || basisCopy[i + 1] < 0)
                    return null;
            Vector[] basisVectors = new Vector[basis.Length], 
            limitationVectors = limitationMatrix.Vectors;
            Vector basisSolution = new Vector(limitationMatrix.SecondDimension), 
            limitationVectorBasisCoefficients = new Vector(basis.Length);
            for (int i = 0; i < basisVectors.Length; i++)
                basisVectors[i] = limitationVectors[basis[i]];
            if (Matrix.UniteVectors(basisVectors).Rank != basis.Length)
                return null;
            limitationVectorBasisCoefficients =
                Matrix.UniteVectors(basisVectors).Reversed * limitationVector;
            for (int i = 0; i < basis.Length; i++)
                basisSolution[basis[i]] = limitationVectorBasisCoefficients[i];
            return AllowableSolution(basisSolution) ? basisSolution : null;
        }

        public Vector AllowableBasis(Vector[] basisVectors)
        {
            if(!Canonical)
                throw new InvalidOperationException("For noncanonical problems basis doesn't exist!");
            if (basisVectors.Length != limitationMatrix.FirstDimension)
                throw new ArgumentException("basisVectors.Length must be equal to limitationMatrix.FirstDimension!", nameof(basisVectors));
            foreach(Vector vector in basisVectors)
                if (vector.Dimension != limitationMatrix.FirstDimension)
                    throw new ArgumentException("Each basis vector must have dimension, which is equal to limitationMatrix.FirstDimension!", nameof(basisVectors));
            bool vectorPartOfLimitationMatrix = false;
            Vector[] limitationVectors = limitationMatrix.Vectors;
            List<int> indexesAllowable = new List<int>();
            int indexTmp = 0;
            int[] basisIndexes = new int[limitationMatrix.FirstDimension];
            for (int i = 0; i < limitationMatrix.SecondDimension; i++)
                indexesAllowable.Add(i);
            foreach(Vector basisVector in basisVectors)
            {
                vectorPartOfLimitationMatrix = false;
                for (int i = 0; i < limitationVectors.Length; i++)
                {
                    if(basisVector == limitationVectors[i] && indexesAllowable.Contains(i))
                    {
                        vectorPartOfLimitationMatrix = true;
                        basisIndexes[indexTmp] = i;
                        indexTmp++;
                        indexesAllowable.Remove(i);
                        break;
                    }
                }
                if (!vectorPartOfLimitationMatrix) return null;
            }
            return AllowableBasis(basisIndexes);
        }

        public Vector AllowableBasis(Matrix basisMatrix) => AllowableBasis(basisMatrix.Vectors);

        private static bool InputDataValid
        (Matrix limitationMatrix, Vector limitationVector, 
         Vector objectiveFunctionCoefficients, short[] relationArray = null, 
         bool[] signArray = null)
        {
            bool res = objectiveFunctionCoefficients.Dimension == limitationMatrix.SecondDimension 
                && limitationVector.Dimension == limitationMatrix.FirstDimension;
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

        public void SetDefaultBasis(int[] defaultBasisIndexes)
        {
            defaultBasisSolution = AllowableBasis(defaultBasisIndexes);
            if (defaultBasisSolution == null)
                throw new ArgumentException("Correspond basis is not allowed!", nameof(defaultBasisIndexes));
            this.defaultBasisIndexes = (int[])defaultBasisIndexes.Clone();
            defaultBasis = new Vector[this.defaultBasisIndexes.Length];
            Vector[] limitationVectors = (Vector[])limitationMatrix.Vectors.Clone();
            for (int i = 0; i < defaultBasis.Length; i++)
                defaultBasis[i] = limitationVectors[defaultBasisIndexes[i]];
        }

        public void SetDefaultBasis(Vector[] defaultBasisVectors)
        {
            defaultBasisSolution = AllowableBasis(defaultBasisVectors);
            if (defaultBasisSolution == null)
                throw new ArgumentException("Correspond basis is not allowed!", nameof(defaultBasisVectors));
            defaultBasis = (Vector[])defaultBasisVectors.Clone();
            defaultBasisIndexes = new int[defaultBasisVectors.Length];
            int indexTmp = 0;
            Vector[] limitationVectors = (Vector[])limitationMatrix.Vectors.Clone();
            List<int> indexesAllowable = new List<int>();
            for (int i = 0; i < limitationMatrix.SecondDimension; i++)
                indexesAllowable.Add(i);
            foreach (Vector basisVector in defaultBasis)
            {
                for (int i = 0; i < limitationVectors.Length; i++)
                {
                    if (basisVector == limitationVectors[i] && indexesAllowable.Contains(i)) 
                    {
                        defaultBasisIndexes[indexTmp] = i;
                        indexTmp++;
                        indexesAllowable.Remove(i);
                        break;
                    }
                }
            }
        }


        public void SetDefaultBasisSolution(Vector defaultBasisSolution)
        {
            if (!Canonical)
                throw new InvalidOperationException("For noncanonical problems basis doesn't exist!");
            if (!AllowableSolution(defaultBasisSolution))
                throw new ArgumentException("Inputed solution is not allowable!", nameof(defaultBasisSolution));
            int positiveComponentsCount = 0, i = 0;
            List<int> allowableIndexes = new List<int>();
            Vector[] limitationVectors = (Vector[])limitationMatrix.Vectors.Clone();
            for (int j = 0; j < defaultBasisSolution.Dimension; j++)
                if (defaultBasisSolution[j] > Epsilon)
                    positiveComponentsCount++;
            if (positiveComponentsCount > limitationMatrix.FirstDimension)
                throw new ArgumentException("Inputed solution can not be basis!", nameof(defaultBasisSolution));
            this.defaultBasisSolution = new Vector(defaultBasisSolution);
            defaultBasisIndexes = new int[limitationMatrix.FirstDimension];
            defaultBasis = new Vector[limitationMatrix.FirstDimension];
            for (int k = 0; k < defaultBasis.Length; k++)
                defaultBasis[k] = Vector.NullVector(defaultBasis.Length);
            for (int j = 0; j < defaultBasisSolution.Dimension; j++)
            {
                if (defaultBasisSolution[j] > Epsilon)
                    defaultBasisIndexes[i++] = j;
                else
                    allowableIndexes.Add(j);
            }
            for (int k = 0; k < i; k++)
                defaultBasis[k] = limitationVectors[defaultBasisIndexes[k]];
            while (i < limitationMatrix.FirstDimension)
            {
                defaultBasisIndexes[i] = allowableIndexes[0];
                defaultBasis[i] = limitationVectors[allowableIndexes[0]];
                if (Matrix.UniteVectors(defaultBasis).Rank == i + 1)
                    i++;
                allowableIndexes.RemoveAt(0);
            }

        }

        public void Solve()
        {
            if (defaultBasis != null && defaultBasisIndexes != null && defaultBasisSolution != null)
                SimplexAlgorithmWithDefaultSolution();
            else
                ArtificialBasisMethod();
        }

        private void SimplexAlgorithmWithDefaultSolution()
        {
            if (defaultBasis == null || defaultBasisIndexes == null || defaultBasisSolution == null)
                throw new ArgumentNullException(nameof(defaultBasis), "One or more arguments were null!");
            List<int> positiveCoordinatesIndexes = new List<int>();
            int[] basisIndexes = (int[])defaultBasisIndexes.Clone();
            int inputedVectorIndex = 0, outputedVectorIndex = 0, indexTmp = 0;
            Vector basisObjectiveFunctionCoefficients = new Vector(defaultBasis.Length),
            estimations = new Vector(limitationMatrix.SecondDimension), 
            simplexRelations = null;
            Vector[] tmpVectors = new Vector[limitationMatrix.SecondDimension + 1];
            Matrix basisMatrix = Matrix.UniteVectors(defaultBasis), 
            oldSimplexTable = new Matrix(limitationMatrix.FirstDimension, limitationMatrix.SecondDimension + 1),
            newSimplexTable = new Matrix(limitationMatrix.FirstDimension, limitationMatrix.SecondDimension + 1);
            for (int i = 0; i < basisObjectiveFunctionCoefficients.Dimension; i++)
                basisObjectiveFunctionCoefficients[i] = objectiveFunctionCoefficients[basisIndexes[i]];
            tmpVectors[0] = limitationVector;
            for (int i = 1; i < tmpVectors.Length; i++)
                tmpVectors[i] = limitationMatrix.Vectors[i - 1];
            oldSimplexTable = basisMatrix.Reversed * Matrix.UniteVectors(tmpVectors);
            while (true)
            {
                for (int j = 0; j < estimations.Dimension; j++)
                    estimations[j] = basisObjectiveFunctionCoefficients * oldSimplexTable.GetColumn(j + 1) - objectiveFunctionCoefficients[j];
                if (estimations.Min > -Epsilon)
                {
                    solution = Vector.NullVector(limitationMatrix.SecondDimension);
                    indexTmp = 0;
                    foreach (int index in basisIndexes)
                        solution[index] = oldSimplexTable[indexTmp++, 0];
                    Solvability = 1;
                    break;
                }
                for (int j = 0; j < estimations.Dimension; j++)
                {
                    if (estimations[j] < -Epsilon && oldSimplexTable.GetColumn(j + 1).Max < Epsilon)
                    {
                        Solvability = 0;
                        break;
                    }
                }
                if(Solvability == 0)
                {
                    solution = null;
                    break;
                }
                inputedVectorIndex = estimations.FirstMinIndex;
                positiveCoordinatesIndexes.Clear();
                for (int i = 0; i < oldSimplexTable.FirstDimension; i++)
                    if (oldSimplexTable[i, inputedVectorIndex + 1] > Epsilon)
                        positiveCoordinatesIndexes.Add(i);
                simplexRelations = new Vector(oldSimplexTable.FirstDimension, double.PositiveInfinity);
                for (int j = 0; simplexRelations.MinIndexes.Length != 1; j++)
                    foreach (int i in positiveCoordinatesIndexes)
                        simplexRelations[i] = oldSimplexTable[i, j] / oldSimplexTable[i, inputedVectorIndex + 1];
                outputedVectorIndex = simplexRelations.FirstMinIndex;
                newSimplexTable = Matrix.NullMatrix(limitationMatrix.FirstDimension, limitationMatrix.SecondDimension + 1);
                for (int j = 0; j < newSimplexTable.SecondDimension; j++)
                    newSimplexTable[outputedVectorIndex, j] = 
                        oldSimplexTable[outputedVectorIndex, j] / 
                        oldSimplexTable[outputedVectorIndex, inputedVectorIndex + 1];
                for (int i = 0; i < newSimplexTable.FirstDimension; i++)
                    for (int j = 0; j < newSimplexTable.SecondDimension; j++)
                        if (i != outputedVectorIndex && j != inputedVectorIndex + 1)
                            newSimplexTable[i, j] =
                                (oldSimplexTable[outputedVectorIndex, inputedVectorIndex + 1] * oldSimplexTable[i, j] -
                                 oldSimplexTable[outputedVectorIndex, j] * oldSimplexTable[i, inputedVectorIndex + 1]) /
                                oldSimplexTable[outputedVectorIndex, inputedVectorIndex + 1];
                oldSimplexTable = newSimplexTable;
                basisIndexes[outputedVectorIndex] = inputedVectorIndex;
                for (int i = 0; i < basisObjectiveFunctionCoefficients.Dimension; i++)
                    basisObjectiveFunctionCoefficients[i] = objectiveFunctionCoefficients[basisIndexes[i]];

            }

        }

        private void ArtificialBasisMethod()
        {
            List<int> positiveCoordinatesIndexes = new List<int>();
            LinearProgrammingProblem canonicalProblem = EqualCanonicalProblem;
            bool allEstimationsNonNegative = false;
            int inputedVectorIndex = 0, outputedVectorIndex = 0, count = 0;
            int[] basisIndexes = new int[canonicalProblem.limitationMatrix.FirstDimension];
            for (int i = 0; i < basisIndexes.Length; i++)
                basisIndexes[i] = canonicalProblem.limitationMatrix.SecondDimension + i;
            Vector auxiliaryProblemSolution = 
                new Vector(canonicalProblem.limitationMatrix.FirstDimension + canonicalProblem.limitationMatrix.SecondDimension),
            simplexRelations = new Vector(canonicalProblem.limitationMatrix.FirstDimension),
            canonicalSolution = new Vector(canonicalProblem.limitationMatrix.SecondDimension);
            Matrix auxiliaryProblemObjCoeffs =
                new Matrix(canonicalProblem.limitationMatrix.FirstDimension + canonicalProblem.limitationMatrix.SecondDimension, 2),
            estimations = new Matrix(canonicalProblem.limitationMatrix.FirstDimension + canonicalProblem.limitationMatrix.SecondDimension, 2),
            basisObjCoeffs = new Matrix(canonicalProblem.limitationMatrix.FirstDimension, 2), 
            auxiliaryOldSimplexTable = 
                Matrix.NullMatrix(canonicalProblem.limitationMatrix.FirstDimension, 
                                  canonicalProblem.limitationMatrix.SecondDimension + canonicalProblem.limitationMatrix.FirstDimension + 1),
            auxiliaryNewSimplexTable =
                Matrix.NullMatrix(canonicalProblem.limitationMatrix.FirstDimension, 
                                  canonicalProblem.limitationMatrix.SecondDimension + canonicalProblem.limitationMatrix.FirstDimension + 1);
            for (int j = 0; j < canonicalProblem.limitationMatrix.SecondDimension; j++)
                auxiliaryProblemObjCoeffs[j, 0] = canonicalProblem.objectiveFunctionCoefficients[j];
            for (int j = canonicalProblem.limitationMatrix.SecondDimension; j < auxiliaryProblemObjCoeffs.FirstDimension; j++)
                auxiliaryProblemObjCoeffs[j, 1] = -1;
            for (int i = 0; i < canonicalProblem.limitationMatrix.FirstDimension; i++)
            {
                auxiliaryOldSimplexTable[i, 0] = canonicalProblem.limitationVector[i];
                for (int j = 0; j < canonicalProblem.limitationMatrix.SecondDimension; j++)
                    auxiliaryOldSimplexTable[i, j + 1] = canonicalProblem.limitationMatrix[i, j];
                auxiliaryOldSimplexTable[i, canonicalProblem.limitationMatrix.SecondDimension + 1 + i] = 1;
            }
            while (true)
            {
                for (int i = 0; i < basisObjCoeffs.FirstDimension; i++)
                    for (int j = 0; j < basisObjCoeffs.SecondDimension; j++)
                        basisObjCoeffs[i, j] = auxiliaryProblemObjCoeffs[basisIndexes[i], j];
                estimations = auxiliaryOldSimplexTable.GetColumns(1).T * basisObjCoeffs - auxiliaryProblemObjCoeffs;
                allEstimationsNonNegative = true;
                for (int j = 0; j < estimations.FirstDimension; j++)
                {
                    if (estimations[j, 1] < -Epsilon || estimations[j, 1] < Epsilon && estimations[j, 0] < -Epsilon)
                    {
                        allEstimationsNonNegative = false;
                        break;
                    }
                }
                if (allEstimationsNonNegative)
                {
                    auxiliaryProblemSolution = Vector.NullVector(auxiliaryProblemSolution.Dimension);
                    for (int i = 0; i < basisIndexes.Length; i++)
                        auxiliaryProblemSolution[basisIndexes[i]] = auxiliaryOldSimplexTable[i, 0];
                    Solvability = 1;
                    break;
                }
                for (int j = 0; j < estimations.FirstDimension; j++)
                {
                    if((estimations[j, 1] < -Epsilon || 
                        estimations[j, 1] < Epsilon && estimations[j, 0] < -Epsilon) &&
                       auxiliaryOldSimplexTable.GetColumn(j + 1).Max < Epsilon)
                    {
                        Solvability = 0;
                        break;
                    }
                }
                if (Solvability == 0)
                {
                    auxiliaryProblemSolution = null;
                    break;
                }
                inputedVectorIndex = estimations.GetColumn(1).Min < -Epsilon ? 
                                                estimations.GetColumn(1).FirstMinIndex: -1;
                if (inputedVectorIndex == -1)
                {
                    inputedVectorIndex = 0;
                    for (int j = 0; j < estimations.FirstDimension; j++)
                        if (Math.Abs(estimations[j, 1]) < Epsilon &&
                            estimations[j, 0] - estimations[inputedVectorIndex, 0] < -Epsilon)
                            inputedVectorIndex = j;
                }
                positiveCoordinatesIndexes.Clear();
                for (int i = 0; i < auxiliaryOldSimplexTable.FirstDimension; i++)
                    if (auxiliaryOldSimplexTable[i, inputedVectorIndex + 1] > Epsilon)
                        positiveCoordinatesIndexes.Add(i);
                simplexRelations = new Vector(auxiliaryOldSimplexTable.FirstDimension, double.PositiveInfinity);
                for (int j = 0; simplexRelations.MinIndexes.Length != 1; j++)
                    foreach (int i in positiveCoordinatesIndexes)
                        simplexRelations[i] = auxiliaryOldSimplexTable[i, j] / 
                            auxiliaryOldSimplexTable[i, inputedVectorIndex + 1];
                outputedVectorIndex = simplexRelations.FirstMinIndex;
                auxiliaryNewSimplexTable = 
                    Matrix.NullMatrix(auxiliaryOldSimplexTable.FirstDimension, auxiliaryOldSimplexTable.SecondDimension);
                for (int j = 0; j < auxiliaryNewSimplexTable.SecondDimension; j++)
                    auxiliaryNewSimplexTable[outputedVectorIndex, j] =
                        auxiliaryOldSimplexTable[outputedVectorIndex, j] /
                        auxiliaryOldSimplexTable[outputedVectorIndex, inputedVectorIndex + 1];
                for (int i = 0; i < auxiliaryNewSimplexTable.FirstDimension; i++)
                    for (int j = 0; j < auxiliaryNewSimplexTable.SecondDimension; j++)
                        if (i != outputedVectorIndex && j != inputedVectorIndex + 1)
                            auxiliaryNewSimplexTable[i, j] =
                                (auxiliaryOldSimplexTable[outputedVectorIndex, inputedVectorIndex + 1] * auxiliaryOldSimplexTable[i, j] -
                                 auxiliaryOldSimplexTable[outputedVectorIndex, j] * auxiliaryOldSimplexTable[i, inputedVectorIndex + 1]) /
                                auxiliaryOldSimplexTable[outputedVectorIndex, inputedVectorIndex + 1];
                auxiliaryOldSimplexTable = auxiliaryNewSimplexTable;
                basisIndexes[outputedVectorIndex] = inputedVectorIndex;
                count++;
            }
            if(auxiliaryProblemSolution == null)
            {
                solution = null;
                Solvability = 0;
                return;
            }
            Solvability = 1;
            for (int j = canonicalProblem.limitationMatrix.SecondDimension;
                 j < auxiliaryProblemSolution.Dimension; j++)
            {
                if (Math.Abs(auxiliaryProblemSolution[j]) > Epsilon)
                {
                    Solvability = 0;
                    solution = null;
                    return;
                }
            }
            for (int j = 0; j < canonicalSolution.Dimension; j++)
                canonicalSolution[j] = auxiliaryProblemSolution[j];
            solution = new Vector(limitationMatrix.SecondDimension);
            for (int j = 0, tmp = 0; j < limitationMatrix.SecondDimension; j++)
            {
                if (signArray[j])
                    solution[j] = canonicalSolution[j + tmp];
                else
                {
                    solution[j] = canonicalSolution[j] - canonicalSolution[j + 1];
                    tmp++;
                }
            }
        }
    }
}
