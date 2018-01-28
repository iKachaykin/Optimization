using System;
using System.Collections.Generic;

namespace SimplexMethod
{
    public class LinearProgrammingProblem : ICloneable
    {
        #region Class fieldes --- Поля класса

        // Class fields, necessary to work with linear proramming problems (LPP)
        // Поля класса, необходимые для работы с задачами линейного программирования (ЗЛП)

        private const double    Epsilon = 1E-8;                     // Field used in order to avoid bugs caused by double variables errors --- Поле, используемое для избежания багов, вызванных погрешностями в представлении в памятим чисел с плавающей точкой
        private const int       validCharNum = 10;                  // Number of allowable symbols for variables --- Количество допустимых символов для обозначения переменных
        private const string    validCharString = "pqrstvwxyz";     // String with allowable symbols for variables --- Строка, содержащая допустимые символы для обозначения переменных

        private Vector[]        defaultBasis;                       // Default basis in simplex method --- Базис по-умолчанию для симплекс-метода
        private int[]           defaultBasisIndexes;                // Indexes of vectors in LimitationMatrix.Vectors, which compose default basis in simplex method --- Индексы векторов в LimitationMatrix.Vectors, что составляют базис по-умолчанию для симплекс-метода
        private Vector          defaultBasisSolution;               // Default basis solution in simplex method --- Опорное решение по-умолчанию, в симплекс-методе
        private Matrix          limitationMatrix;                   // Matrix which contains coefficients before variables in system of limitation --- Матрица, содержащая коэффициенты перед неизвестными в системе ограничений
        private Vector          limitationVector;                   // Vector which contains free coefficients in system of limitation --- Вектор, содержащий свободные члены в системе ограничений
        private bool            maxObjectiveValue;                  // Field which eqaul true, if wanted extremum is max, and false - otherwise --- Поле, равное true, если искомый экстремум - максимум, и false - в противном случае
        private Vector          objectiveFunctionCoefficients;      // Coefficients before variables in definition of objective function --- Коэффиценты перед неизвестными в определении целевой функции
        private short[]         relationArray;                      // Array which element equal -1, when type of respective relation is "<=", 0 - when "=", and "1" - when ">=" --- Массив, элемент которого равен -1, если соответствующий тип отношения "<=", 0 - если "=", и 1 - если ">="
        private short[]         signArray;                          // Array which element equal -1, when respective variable is "<= 0", 1 - when ">= 0", and "0", if info is missing --- Массив, элемент которого равен -1, если соответствующая переменная "<= 0", 1 - если ">= 0", и "0" - если о переменной ничего неизвестно
        private Vector          solution;                           // Field correspond to problem solution --- Поле относящееся к решению задачи
        private char            variableChar;                       // Field wich used as char of variables --- Поле используемое, как символ для неизвестных

        #endregion

        #region Constructors --- Конструкторы

        // Constructor which get all possible parameteres and set their values to respective class fields
        //
        public LinearProgrammingProblem
        (Matrix limitationMatrix, Vector limitationVector, 
         Vector objectiveFunctionCoefficients, bool algorithmPrint = false, 
         bool maxObjectiveValue = true, short[] relationArray = null, 
         short[] signArray = null, char variableChar = 'x')
        {
            if (!InputDataValid(limitationMatrix, limitationVector, 
                                objectiveFunctionCoefficients,relationArray, signArray))
                throw new ArgumentException("The dimensions/lengths of arguments have to correspond to each other!");
            this.AlgorithmPrint                = algorithmPrint;
            AllSimplexTables                   = new List<string[,]>();
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
                this.signArray = new short[limitationMatrix.SecondDimension];
                for (int i = 0; i < this.signArray.Length; i++)
                    this.signArray[i] = 1;
            }
            else
                this.signArray = (short[])signArray.Clone();
        }

        // Copy constructor
        // 
        public LinearProgrammingProblem(LinearProgrammingProblem other)
        {
            AlgorithmPrint                = other.AlgorithmPrint;
            AllSimplexTables              = new List<string[,]>();
            defaultBasis                  = other.defaultBasis;
            defaultBasisSolution          = other.defaultBasisSolution;
            limitationMatrix              = other.limitationMatrix;
            limitationVector              = other.limitationVector; 
            maxObjectiveValue             = other.maxObjectiveValue;
            objectiveFunctionCoefficients = other.objectiveFunctionCoefficients;
            relationArray                 = (short[])other.relationArray.Clone();
            signArray                     = (short[])other.signArray.Clone();
            solution                      = other.solution;
            Solvability                   = other.Solvability;
            variableChar                  = other.variableChar;
            foreach (string[,] table in other.AllSimplexTables)
                AllSimplexTables.Add(table);
            if (other.defaultBasisIndexes == null)
                defaultBasisIndexes = null;
            else
                defaultBasisIndexes = (int[])other.defaultBasisIndexes.Clone();

        }

        #endregion

        #region Methods --- Методы

        // Method which check if inputed basis (array of indeces of basis vectors in limitationMatrix.Vectors) is allowable
        // Return basis solution respective to inputed basis, if basis is allowable, and null - otherwise
        //
        //
        public Vector AllowableBasis(int[] basis)
        {
            if (!Canonical)
                throw new InvalidOperationException("For noncanonical problems basis doesn't exist!");
            if (basis.Length != LimitationNumber)
                throw new ArgumentException("basis.Length must be equal to limitationMatrix.FirstDimension!", nameof(basis));
            int[] basisCopy = (int[])basis.Clone();
            Array.Sort(basisCopy);
            for (int i = 0; i < basisCopy.Length - 1; i++)
                if (basisCopy[i] == basisCopy[i + 1] || basisCopy[i] < 0 || basisCopy[i + 1] < 0)
                    return null;
            Vector   basisSolution                     = new Vector(VariableNumber);
            Vector[] basisVectors                      = new Vector[basis.Length];
            Vector   limitationVectorBasisCoefficients = new Vector(basis.Length);
            Vector[] limitationVectors                 = limitationMatrix.Vectors;
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

        // Method which check if inputed basis (array of vectors) is allowable
        // Return basis solution respective to inputed basis, if basis is allowable, and null - otherwise
        //
        //
        public Vector AllowableBasis(Vector[] basisVectors)
        {
            if (!Canonical)
                throw new InvalidOperationException("For noncanonical problems basis doesn't exist!");
            if (basisVectors.Length != LimitationNumber)
                throw new ArgumentException("basisVectors.Length must be equal to limitationMatrix.FirstDimension!", nameof(basisVectors));
            foreach (Vector vector in basisVectors)
                if (vector.Dimension != LimitationNumber)
                    throw new ArgumentException("Each basis vector must have dimension, which is equal to limitationMatrix.FirstDimension!", nameof(basisVectors));
            int[]     basisIndexes                 = new int[LimitationNumber];
            List<int> indexesAllowable             = new List<int>();
            int       indexTmp                     = 0;
            Vector[]  limitationVectors            = limitationMatrix.Vectors;
            bool      vectorPartOfLimitationMatrix = false;
            for (int i = 0; i < VariableNumber; i++)
                indexesAllowable.Add(i);
            foreach (Vector basisVector in basisVectors)
            {
                vectorPartOfLimitationMatrix = false;
                for (int i = 0; i < limitationVectors.Length; i++)
                {
                    if (basisVector == limitationVectors[i] && indexesAllowable.Contains(i))
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

        // Method which check if inputed basis (matrix containing basis vectors) is allowable
        // Return basis solution respective to inputed basis, if basis is allowable, and null - otherwise
        //
        //
        public Vector AllowableBasis(Matrix basisMatrix) => AllowableBasis(basisMatrix.Vectors);

        // Method which check is inputed vector an allowable solution of LPP
        // 
        public bool AllowableSolution(Vector solution)
        {
            if (solution.Dimension != VariableNumber)
                throw new ArgumentException("solution.Dimension must be equal to limitationMatrix.FirstDimension!", nameof(solution));
            double   composition = 1;
            Vector[] vectors     = limitationMatrix.T.Vectors;
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
                if (signArray[j] == 1 && solution[j] < 0 || signArray[j] == -1 && solution[j] > 0)
                    return false;
            return true;

        }

        // Method which solve LPP, using artificial basis method and simplex algorithm
        // 
        private void ArtificialBasisMethod()
        {
            LinearProgrammingProblem canonicalProblem = EqualCanonicalProblem;
            AllSimplexTables.Clear();
            bool      allEstimationsNonNegative  = false;
            Matrix    auxiliaryNewSimplexTable   = Matrix.NullMatrix(canonicalProblem.LimitationNumber,
                                                                     canonicalProblem.VariableNumber +
                                                                     canonicalProblem.LimitationNumber + 1);
            Matrix    auxiliaryOldSimplexTable   = Matrix.NullMatrix(canonicalProblem.LimitationNumber,
                                                                     canonicalProblem.VariableNumber +
                                                                     canonicalProblem.LimitationNumber + 1);
            Matrix    auxiliaryProblemObjCoeffs  = new Matrix(canonicalProblem.LimitationNumber +
                                                              canonicalProblem.VariableNumber, 2);
            Vector    auxiliaryProblemSolution   = new Vector(canonicalProblem.LimitationNumber +
                                                              canonicalProblem.VariableNumber);
            int[]     basisIndexes               = new int[canonicalProblem.LimitationNumber];
            Matrix    basisObjCoeffs             = new Matrix(canonicalProblem.LimitationNumber, 2);
            Vector    canonicalSolution          = new Vector(canonicalProblem.VariableNumber);
            int       count                      = 0;
            Matrix    estimations                = new Matrix(canonicalProblem.LimitationNumber +
                                                              canonicalProblem.VariableNumber, 2);
            int       inputedVectorIndex         = 0;
            int       outputedVectorIndex        = 0;
            List<int> positiveCoordinatesIndexes = new List<int>();
            Vector    simplexRelations           = new Vector(canonicalProblem.LimitationNumber);
            for (int i = 0; i < canonicalProblem.LimitationNumber; i++)
                if (canonicalProblem.limitationVector[i] < -Epsilon)
                {
                    canonicalProblem.limitationVector[i] = -canonicalProblem.limitationVector[i];
                    for (int j = 0; j < canonicalProblem.VariableNumber; j++)
                        canonicalProblem.limitationMatrix[i, j] = -canonicalProblem.limitationMatrix[i, j];
                }
            for (int i = 0; i < basisIndexes.Length; i++)
                basisIndexes[i] = canonicalProblem.VariableNumber + i;
            for (int j = 0; j < canonicalProblem.VariableNumber; j++)
                auxiliaryProblemObjCoeffs[j, 0] = canonicalProblem.objectiveFunctionCoefficients[j];
            for (int j = canonicalProblem.VariableNumber; j < auxiliaryProblemObjCoeffs.FirstDimension; j++)
                auxiliaryProblemObjCoeffs[j, 1] = -1;
            for (int i = 0; i < canonicalProblem.LimitationNumber; i++)
            {
                auxiliaryOldSimplexTable[i, 0] = canonicalProblem.limitationVector[i];
                for (int j = 0; j < canonicalProblem.VariableNumber; j++)
                    auxiliaryOldSimplexTable[i, j + 1] = canonicalProblem.limitationMatrix[i, j];
                auxiliaryOldSimplexTable[i, canonicalProblem.limitationMatrix.SecondDimension + 1 + i] = 1;
            }
            while (true)
            {
                for (int i = 0; i < basisObjCoeffs.FirstDimension; i++)
                    for (int j = 0; j < basisObjCoeffs.SecondDimension; j++)
                        basisObjCoeffs[i, j] = auxiliaryProblemObjCoeffs[basisIndexes[i], j];
                estimations = auxiliaryOldSimplexTable.GetColumns(1).T * basisObjCoeffs - auxiliaryProblemObjCoeffs;
                if (AlgorithmPrint)
                    AllSimplexTables.Add(FillStringSimplexTable(auxiliaryProblemObjCoeffs, auxiliaryOldSimplexTable, estimations, basisIndexes));
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
                    if ((estimations[j, 1] < -Epsilon ||
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
                                                estimations.GetColumn(1).FirstMinIndex : -1;
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
            if (auxiliaryProblemSolution == null)
            {
                solution = null;
                Solvability = 0;
                return;
            }
            Solvability = 1;
            for (int j = canonicalProblem.VariableNumber; j < auxiliaryProblemSolution.Dimension; j++)
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
            solution = new Vector(TrimCanonicalSolution(canonicalSolution));
        }

        // Clone() override
        // 
        public object Clone() => new LinearProgrammingProblem(this);

        // Auxiliary method used in order to print valid results of artificial basis algorithm
        // 
        private string ConvertSpecialVectorToString(Vector specialVector)
        {
            string res = "";
            if (Math.Abs(specialVector[1]) > Epsilon)
                res += Math.Abs(specialVector[1] - 1) < Epsilon ? "M" :
                    (Math.Abs(specialVector[1] + 1) < Epsilon ? "-M" : Convert.ToString(specialVector[1]) + "M");
            if (Math.Abs(specialVector[0]) > Epsilon)
                res += (specialVector[0] > Epsilon && Math.Abs(specialVector[1]) > Epsilon ? "+" : "") +
                    Convert.ToString(specialVector[0]);
            if (Math.Abs(specialVector[0]) < Epsilon && Math.Abs(specialVector[1]) < Epsilon)
                res = "0";
            return res;
        }

        // Method returning string which contains used simplex table
        // 
        private string[,] FillStringSimplexTable(Vector objCoefficients,
            Matrix simplexTableCoordinates, Vector estimations, int[] basisIndexes)
        {
            string[,] table = new string[LimitationNumber + 3, VariableNumber + 4];
            table[0, 0] = "C";
            table[0, 3] = table[0, 2] = table[0, 1] = "";
            for (int j = 4; j < table.GetLength(1); j++)
                table[0, j] = Convert.ToString(objCoefficients[j - 4]);
            table[1, 0] = "#";
            table[1, 1] = "B";
            table[1, 2] = "Cb";
            for (int j = 3; j < table.GetLength(1); j++)
                table[1, j] = "A" + Convert.ToString(j - 3);
            for (int i = 2; i < table.GetLength(0) - 1; i++)
            {
                for (int j = 0; j < table.GetLength(1); j++)
                {
                    if (j == 0)
                        table[i, j] = Convert.ToString(i - 1);
                    else if (j == 1)
                        table[i, j] = "A" + Convert.ToString(basisIndexes[i - 2] + 1);
                    else if (j == 2)
                        table[i, j] = Convert.ToString(objCoefficients[basisIndexes[i - 2]]);
                    else
                        table[i, j] = Convert.ToString(simplexTableCoordinates[i - 2, j - 3]);
                }
            }
            table[table.GetLength(0) - 1, 0] = "E";
            table[table.GetLength(0) - 1, 2] = table[table.GetLength(0) - 1, 1] = "";
            table[table.GetLength(0) - 1, 3] = Convert.ToString(objCoefficients[basisIndexes] * simplexTableCoordinates.GetColumn(0));
            for (int j = 4; j < table.GetLength(1); j++)
                table[table.GetLength(0) - 1, j] = Convert.ToString(estimations[j - 4]);
            return table;
        }

        // Method returning string which contains used simplex table
        // 
        private string[,] FillStringSimplexTable(Matrix objCoefficients, Matrix simplexTableCoordinates, Matrix estimations, int[] basisIndexes)
        {
            string[,] table = new string[LimitationNumber + 3, simplexTableCoordinates.SecondDimension + 3];
            table[0, 0] = "C";
            table[0, 3] = table[0, 2] = table[0, 1] = "";
            for (int j = 4; j < table.GetLength(1); j++)
                table[0, j] = ConvertSpecialVectorToString(objCoefficients.GetRow(j - 4));
            table[1, 0] = "#";
            table[1, 1] = "B";
            table[1, 2] = "Cb";
            for (int j = 3; j < table.GetLength(1); j++)
                table[1, j] = "A" + Convert.ToString(j - 3);
            for (int i = 2; i < table.GetLength(0) - 1; i++)
            {
                for (int j = 0; j < table.GetLength(1); j++)
                {
                    if (j == 0)
                        table[i, j] = Convert.ToString(i);
                    else if (j == 1)
                        table[i, j] = "A" + Convert.ToString(basisIndexes[i - 2] + 1);
                    else if (j == 2)
                        table[i, j] = ConvertSpecialVectorToString(objCoefficients.GetRow(basisIndexes[i - 2]));
                    else
                        table[i, j] = Convert.ToString(simplexTableCoordinates[i - 2, j - 3]);
                }
            }
            table[table.GetLength(0) - 1, 0] = "E";
            table[table.GetLength(0) - 1, 2] = table[table.GetLength(0) - 1, 1] = "";
            table[table.GetLength(0) - 1, 3] = "";
            for (int j = 4; j < table.GetLength(1); j++)
                table[table.GetLength(0) - 1, j] = ConvertSpecialVectorToString(estimations.GetRow(j - 4));
            return table;
        }

        // Method which check inputed data and return true, if it could define LPP, and false - otherwise
        // 
        private static bool InputDataValid
        (Matrix limitationMatrix, Vector limitationVector,
         Vector objectiveFunctionCoefficients, short[] relationArray = null,
         short[] signArray = null)
        {
            bool res = objectiveFunctionCoefficients.Dimension == limitationMatrix.SecondDimension
                && limitationVector.Dimension == limitationMatrix.FirstDimension;
            res = res && objectiveFunctionCoefficients != Vector.NullVector(objectiveFunctionCoefficients.Dimension);
            for (int i = 0; i < limitationMatrix.FirstDimension; i++)
                if (limitationMatrix.GetRow(i) == Vector.NullVector(limitationMatrix.SecondDimension))
                    return false;
            if (relationArray != null)
            {
                res = res && relationArray.Length == limitationMatrix.FirstDimension;
                for (int i = 0; i < relationArray.Length; i++)
                    res = res && (relationArray[i] == -1 || relationArray[i] == 0 || relationArray[i] == 1);
            }
            if (signArray != null)
            {
                res = res && signArray.Length == limitationMatrix.SecondDimension;
                for (int j = 0; j < signArray.Length; j++)
                    res = res && (signArray[j] == 1 || signArray[j] == 0 || signArray[j] == -1);
            }
            return res;
        }

        // Setter for default basis
        // 
        public void SetDefaultBasis(int[] defaultBasisIndexes)
        {
            defaultBasisSolution       = AllowableBasis(defaultBasisIndexes);
            if (defaultBasisSolution == null)
                throw new ArgumentException("Correspond basis is not allowed!", nameof(defaultBasisIndexes));
            this.defaultBasisIndexes   = (int[])defaultBasisIndexes.Clone();
            defaultBasis               = new Vector[this.defaultBasisIndexes.Length];
            Vector[] limitationVectors = (Vector[])limitationMatrix.Vectors.Clone();
            for (int i = 0; i < defaultBasis.Length; i++)
                defaultBasis[i] = limitationVectors[defaultBasisIndexes[i]];
        }

        // Setter for default basis
        // 
        public void SetDefaultBasis(Vector[] defaultBasisVectors)
        {
            defaultBasisSolution       = AllowableBasis(defaultBasisVectors);
            if (defaultBasisSolution == null)
                throw new ArgumentException("Correspond basis is not allowed!", nameof(defaultBasisVectors));
            defaultBasis               = (Vector[])defaultBasisVectors.Clone();
            defaultBasisIndexes        = new int[defaultBasisVectors.Length];
            List<int> indexesAllowable = new List<int>();
            int indexTmp               = 0;
            Vector[] limitationVectors = (Vector[])limitationMatrix.Vectors.Clone();
            for (int j = 0; j < VariableNumber; j++)
                indexesAllowable.Add(j);
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

        // Setter for default basis solution
        // 
        public void SetDefaultBasisSolution(Vector defaultBasisSolution)
        {
            if (!Canonical)
                throw new InvalidOperationException("For noncanonical problems basis doesn't exist!");
            if (!AllowableSolution(defaultBasisSolution))
                throw new ArgumentException("Inputed solution is not allowable!", nameof(defaultBasisSolution));
            List<int> allowableIndexes        = new List<int>();
            int       i                       = 0;
            Vector[]  limitationVectors       = (Vector[])limitationMatrix.Vectors.Clone();
            int       positiveComponentsCount = 0;
            for (int j = 0; j < defaultBasisSolution.Dimension; j++)
                if (defaultBasisSolution[j] > Epsilon)
                    positiveComponentsCount++;
            if (positiveComponentsCount > LimitationNumber)
                throw new ArgumentException("Inputed solution can not be basis!", nameof(defaultBasisSolution));
            this.defaultBasisSolution = new Vector(defaultBasisSolution);
            defaultBasisIndexes       = new int[LimitationNumber];
            defaultBasis              = new Vector[LimitationNumber];
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
            while (i < LimitationNumber)
            {
                defaultBasisIndexes[i] = allowableIndexes[0];
                defaultBasis[i] = limitationVectors[allowableIndexes[0]];
                if (Matrix.UniteVectors(defaultBasis).Rank == i + 1)
                    i++;
                allowableIndexes.RemoveAt(0);
            }

        }

        // Simplex algorithm which is used, if default basis or default basis solution were inputed
        // 
        private void SimplexAlgorithmWithDefaultSolution()
        {
            if (defaultBasis == null || defaultBasisIndexes == null || defaultBasisSolution == null)
                throw new ArgumentNullException(nameof(defaultBasis), "One or more arguments were null!");
            AllSimplexTables.Clear();
            int[]     basisIndexes                       = (int[])defaultBasisIndexes.Clone();
            Matrix    basisMatrix                        = Matrix.UniteVectors(defaultBasis);
            Vector    basisObjectiveFunctionCoefficients = new Vector(defaultBasis.Length);
            Vector    estimations                        = new Vector(VariableNumber);
            int       indexTmp                           = 0;
            int       inputedVectorIndex                 = 0;
            Matrix    newSimplexTable                    = new Matrix(LimitationNumber,
                                                                      VariableNumber + 1);
            Matrix    oldSimplexTable                    = new Matrix(LimitationNumber,
                                                                      VariableNumber + 1);
            int       outputedVectorIndex                = 0;
            List<int> positiveCoordinatesIndexes         = new List<int>();
            Vector    simplexRelations                   = null;
            Vector[]  tmpVectors                         = new Vector[VariableNumber + 1];
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
                if (AlgorithmPrint)
                    AllSimplexTables.Add(FillStringSimplexTable(objectiveFunctionCoefficients, oldSimplexTable, estimations, basisIndexes));
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
                if (Solvability == 0)
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

        // Method solving LPP
        // 
        public void Solve()
        {
            if (defaultBasis != null && defaultBasisIndexes != null && defaultBasisSolution != null)
                SimplexAlgorithmWithDefaultSolution();
            else
                ArtificialBasisMethod();
        }

        // Method converting LPP to string
        //
        public override string ToString()
        {
            string res = "";
            for (int j = 0, firstPrint = 0; j < objectiveFunctionCoefficients.Dimension; j++)
            {
                if (objectiveFunctionCoefficients[j] > Epsilon)
                    res += j == firstPrint ? "" : " + ";
                else if (objectiveFunctionCoefficients[j] < -Epsilon)
                    res += j == firstPrint ? "-" : " - ";
                else
                {
                    if (j == firstPrint) firstPrint++;
                    continue;
                }
                res += (Math.Abs(Math.Abs(objectiveFunctionCoefficients[j]) - 1) < Epsilon ? "" :
                    Convert.ToString(Math.Abs(objectiveFunctionCoefficients[j]))) + variableChar + Convert.ToString(j + 1);
            }
            res += " --> ";
            if (maxObjectiveValue)
                res += "max\n";
            else
                res += "min\n";
            for (int i = 0; i < limitationMatrix.FirstDimension; i++)
            {
                for (int j = 0, firstPrint = 0; j < limitationMatrix.SecondDimension; j++)
                {
                    if (limitationMatrix[i, j] > Epsilon)
                        res += j == firstPrint ? "" : " + ";
                    else if (limitationMatrix[i, j] < -Epsilon)
                        res += j == firstPrint ? "-" : " - ";
                    else
                    {
                        if (j == firstPrint) firstPrint++;
                        continue;
                    }
                    res += (Math.Abs(Math.Abs(limitationMatrix[i, j]) - 1) < Epsilon ? "" :
                    Convert.ToString(Math.Abs(limitationMatrix[i, j]))) + variableChar + Convert.ToString(j + 1);
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
            {
                if (signArray[j] == 1)
                    res += variableChar + Convert.ToString(j + 1) + " >= 0    ";
                else if (signArray[j] == -1)
                    res += variableChar + Convert.ToString(j + 1) + " <= 0    ";
            }
            return res;
        }

        // Method converting solution of equal canonical LPP to solution of this
        // 
        public Vector TrimCanonicalSolution(Vector canonicalSolution)
        {
            if (canonicalSolution == null)
                return null;
            else if (!EqualCanonicalProblem.AllowableSolution(canonicalSolution))
                throw new ArgumentException("Inputed solution is not allowable!", nameof(canonicalSolution));
            Vector thisSolution = new Vector(VariableNumber);
            for (int j = 0, tmp = 0; j < VariableNumber; j++)
            {
                if (signArray[j] == 1)
                    thisSolution[j] = canonicalSolution[j + tmp];
                else if (signArray[j] == -1)
                    thisSolution[j] = -canonicalSolution[j + tmp];
                else
                {
                    thisSolution[j] = canonicalSolution[j] - canonicalSolution[j + 1];
                    tmp++;
                }
            }
            return thisSolution;
        }

        #endregion

        #region Properties --- Свойства

        // Property defining is it necessary to store intermediate simplex tables
        // 
        public bool            AlgorithmPrint   { get; private set; }

        // List containing all simplex tables appeared in simplex algorithm
        // 
        public List<string[,]> AllSimplexTables { get; private set; }

        // Property, returning true, if LPP is canonical, and false - otherwise
        // 
        public bool            Canonical
        {
            get
            {
                foreach (short relation in relationArray)
                    if (relation != 0)
                        return false;
                foreach (short sign in signArray)
                    if (sign != 1)
                        return false;
                return maxObjectiveValue &&
                    limitationMatrix.Rank == limitationMatrix.FirstDimension &&
                    limitationMatrix.FirstDimension != limitationMatrix.SecondDimension;
            }
        }

        // Property creating equal canonical LPP
        // 
        public LinearProgrammingProblem EqualCanonicalProblem
        {
            get
            {
                List<int> indexesList       = new List<int>();
                int       indexTmp          = 0;
                Matrix    limitationMatrix  = new Matrix(this.limitationMatrix);
                Vector    limitationVector  = new Vector(this.limitationVector);
                bool      maxObjectiveValue = true;
                Vector    objCoeffsTmp      = new Vector(this.objectiveFunctionCoefficients);
                Random    random            = new Random();
                short[]   relationArray     = new short[this.relationArray.Length];
                Vector[]  tmpVectorArray;
                char      variableChar      = this.variableChar;
                while (variableChar == this.variableChar)
                    variableChar = validCharString[random.Next(validCharNum)];
                for (int j = 0; j < this.limitationMatrix.SecondDimension; j++)
                {
                    if (this.signArray[j] == -1)
                    {
                        objCoeffsTmp[j] = -objCoeffsTmp[j];
                        for (int i = 0; i < this.limitationMatrix.FirstDimension; i++)
                            limitationMatrix[i, j] = -limitationMatrix[i, j];
                    }
                }
                for (int j = 0; j < this.signArray.Length; j++)
                    if (this.signArray[j] == 0)
                        indexesList.Add(j);
                foreach (int index in indexesList)
                {
                    tmpVectorArray = (Vector[])limitationMatrix.Vectors.Clone();
                    Array.Resize(ref tmpVectorArray, tmpVectorArray.Length + 1);
                    tmpVectorArray[index + indexTmp + 1] = -tmpVectorArray[index + indexTmp];
                    Array.Copy(limitationMatrix.Vectors, index + indexTmp + 1, tmpVectorArray,
                               index + indexTmp + 2, VariableNumber - index - indexTmp - 1);
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
                short[] signArray = new short[VariableNumber];
                for (int i = 0; i < signArray.Length; i++)
                    signArray[i] = 1;
                Vector objectiveFunctionCoefficients = new Vector(VariableNumber);
                for (int i = 0, newIndex = 0; i < objCoeffsTmp.Dimension; i++, newIndex++)
                {
                    objectiveFunctionCoefficients[newIndex] = objCoeffsTmp[i];
                    if (indexesList.Contains(i))
                    {
                        newIndex++;
                        objectiveFunctionCoefficients[newIndex] = -objCoeffsTmp[i];
                    }
                }
                for (int i = objCoeffsTmp.Dimension + indexesList.Count; i < objectiveFunctionCoefficients.Dimension; i++)
                    objectiveFunctionCoefficients[i] = 0;
                if (!this.maxObjectiveValue)
                    objectiveFunctionCoefficients = -objectiveFunctionCoefficients;
                return new LinearProgrammingProblem(limitationMatrix, limitationVector,
                                                    objectiveFunctionCoefficients,
                                                    AlgorithmPrint, maxObjectiveValue,
                                                    relationArray, signArray, variableChar);
            }
        }

        // Property for limitationMatrix
        // 
        public Matrix LimitationMatrix              { get { return limitationMatrix; } }

        // Property returning number of limitation of LPP
        // 
        public int    LimitationNumber              { get { return limitationMatrix.FirstDimension; } }

        // Property for limitationVector
        // 
        public Vector LimitationVector              { get { return limitationVector; } }

        // Property for objectiveFunctionCoefficients
        // 
        public Vector ObjectiveFunctionCoefficients { get { return objectiveFunctionCoefficients; } }

        // Property for solution of LPP (= null if LPP is unsolvable)
        // 
        public Vector Solution                      { get { return solution; } }

        // Property which return 1, if LPP is solvable, 0 - otherwise, and -1, if LPP hasn't been solved yet
        // 
        public short  Solvability                   { get; private set; }

        // Property returning number of variables of LPP
        // 
        public int    VariableNumber                { get { return limitationMatrix.SecondDimension; } }

        // Property for variableChar
        //
        public char   VariableSymbol                { get { return variableChar; } }

        #endregion
    }
}
