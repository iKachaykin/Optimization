﻿using System;
using System.Collections.Generic;

namespace SimplexMethod
{
    public class Vector : ICloneable
    {
        const double Epsilon = 1E-8;
        private int dimension;
        private double[] vector;

        public Vector(int dimension = 16, double defaultValue = 0.0)
        {
            if (dimension <= 0)
                throw new InvalidOperationException();
            this.dimension = dimension;
            vector = new double[dimension];
            for (int i = 0; i < dimension; i++)
                vector[i] = defaultValue;
        }

        public Vector(Vector other)
        {
            dimension = other.dimension;
            vector = (double[])other.vector.Clone();
        }

        public Vector(double[] vector)
        {
            dimension = vector.Length;
            this.vector = new double[dimension];
            for (int i = 0; i < dimension; i++)
                this.vector[i] = vector[i];
        }

        public Vector(List<double> list)
        {
            dimension = list.Count;
            vector = new double[dimension];
            for (int i = 0; i < dimension; i++)
                vector[i] = list[i];
        }

        public object Clone() { return new Vector(this); }

        public double this[int i]
        {
            get
            {
                if (i < 0 || i >= dimension)
                    throw new IndexOutOfRangeException();
                return vector[i];
            }
            set
            {
                if (i < 0 || i >= dimension)
                    throw new IndexOutOfRangeException();
                vector[i] = value;
            }
        }

        public Vector this[int[] indexes]
        {
            get
            {
                foreach(int index in indexes)
                {
                    if (index < 0 || index >= dimension)
                        throw new IndexOutOfRangeException();
                }
                Vector res = new Vector(indexes.Length);
                for (int i = 0; i < res.dimension; i++)
                    res.vector[i] = vector[indexes[i]];
                return res;
            }
        }

        public static Vector operator +(Vector v) { return new Vector(v); }

        public static Vector operator -(Vector v)
        {
            Vector res = new Vector(v.dimension);
            for (int i = 0; i < v.dimension; i++)
                res[i] = -v[i];
            return res;
        }

        public static Vector operator +(Vector left, Vector right)
        {
            if (left.dimension != right.dimension)
                throw new InvalidOperationException();
            Vector res = new Vector(left.dimension, 0.0);
            for (int i = 0; i < res.dimension; i++)
                res.vector[i] = left.vector[i] + right.vector[i];
            return res;
        }

        public static Vector operator -(Vector left, Vector right)
        {
            if (left.dimension != right.dimension)
                throw new InvalidOperationException();
            Vector res = new Vector(left.dimension, 0.0);
            for (int i = 0; i < res.dimension; i++)
                res.vector[i] = left.vector[i] - right.vector[i];
            return res;
        }

        public static Vector operator *(double c, Vector v)
        {
            Vector res = new Vector(v);
            for (int i = 0; i < res.dimension; i++)
                res.vector[i] *= c;
            return res;
        }

        public static Vector operator *(Vector v, double c)
        {
            Vector res = new Vector(v);
            for (int i = 0; i < res.dimension; i++)
                res.vector[i] *= c;
            return res;
        }

        public static double operator *(Vector left, Vector right)
        {
            if (left.dimension != right.dimension)
                throw new InvalidOperationException();
            double res = 0.0;
            for (int i = 0; i < left.dimension; i++)
                res += left[i] * right[i];
            return res;
        }

        public static Vector operator /(Vector v, double c)
        {
            Vector res = new Vector(v);
            for (int i = 0; i < res.dimension; i++)
                res.vector[i] /= c;
            return res;
        }

        public static bool operator ==(Vector left, Vector right)
        {
            if ((object)left == null && (object)right == null)
                return true;
            if ((object)left == null || (object)right == null)
                return false;
            if (left.dimension != right.dimension)
                return false;
            for (int i = 0; i < left.dimension; i++)
                if (Math.Abs(left[i] - right[i]) > Epsilon)
                    return false;
            return left.dimension == right.dimension;
        }

        public static bool operator !=(Vector left, Vector right)
        {
            if ((object)left == null && (object)right == null)
                return false;
            if ((object)left == null || (object)right == null)
                return true;
            if (left.dimension != right.dimension)
                return true;
            for (int i = 0; i < left.dimension; i++)
                if (Math.Abs(left[i] - right[i]) > Epsilon)
                    return true;
            return false;
        }

        public static explicit operator double(Vector v)
        {
            if (v.dimension != 1)
                throw new InvalidOperationException();
            return v.vector[0];
        }

        public static explicit operator Vector(double c) { return new Vector(1, c); }

        public static explicit operator Matrix(Vector v)
        {
            Matrix res = new Matrix(1, v.Dimension);
            for (int i = 0; i < v.Dimension; i++)
                res[0, i] = v.vector[i];
            return res;
        }

        public override bool Equals(object obj)
        {
            if (obj == null || GetType() != obj.GetType())
                return false;
            Vector v = (Vector)obj;
            return this == v;
        }

        public override int GetHashCode() { return Convert.ToInt32(EuclidNorm); }

        public override string ToString()
        {
            String res = "(";
            double delt = 0;
            for (int i = 0; i < dimension; i++)
            {
                delt = Math.Abs(vector[i] - Math.Round(vector[i]));
                if (i != dimension - 1)
                    res += Convert.ToString(delt < Epsilon ? Convert.ToInt32(Math.Round(vector[i])) : vector[i]) + "; ";
                else
                    res += Convert.ToString(delt < Epsilon ? Convert.ToInt32(Math.Round(vector[i])) : vector[i]);
            }
            return res + ")";
            
                    
        }

        public double EuclidNorm { get { return Math.Sqrt(this * this); } }

        public int Dimension { get { return dimension; } }

        public double Sum 
        {
            get
            {
                double res = 0;
                foreach (double elem in vector)
                    res += elem;
                return res;
            }
        }

        public Vector Floor
        {
            get
            {
                Vector res = new Vector(this);
                for (int i = 0; i < res.dimension; i++)
                    res.vector[i] = Math.Floor(res.vector[i]);
                return res;
            }
        }

        public Vector Ceil
        {
            get
            {
                Vector res = new Vector(this);
                for (int i = 0; i < res.dimension; i++)
                    res.vector[i] = Math.Ceiling(res.vector[i]);
                return res;
            }
        }

        public Vector Abs
        {
            get
            {
                Vector res = new Vector(this);
                for (int i = 0; i < res.Dimension; i++)
                    if (res.vector[i] < 0)
                        res.vector[i] = -res.vector[i];
                return res;
            }
        }

        public double Max
        {
            get
            {
                double res = vector[0];
                for (int i = 1; i < vector.Length; i++)
                    if (vector[i] > res)
                        res = vector[i];
                return res;
            }
        }

        public double Min
        {
            get
            {
                double res = vector[0];
                for (int i = 1; i < vector.Length; i++)
                    if (vector[i] < res)
                        res = vector[i];
                return res;   
            }
        }

        public int FirstMaxIndex
        {
            get
            {
                int res = 0;
                for (int i = 1; i < vector.Length; i++)
                    if (vector[i] - vector[res] > Epsilon)
                        res = i;
                return res;
            }
        }

        public int FirstMinIndex
        {
            get
            {
                int res = 0;
                for (int i = 1; i < vector.Length; i++)
                    if (vector[i] - vector[res] < -Epsilon)
                        res = i;
                return res;
            }
        }

        public int[] MaxIndexes
        {
            get
            {
                double max = Max;
                List<int> tmpLst = new List<int>();
                for (int i = 0; i < dimension; i++)
                    if (Math.Abs(vector[i] - max) < Epsilon)
                        tmpLst.Add(i);
                int[] res = new int[tmpLst.Count];
                for (int i = 0; i < res.Length; i++)
                    res[i] = tmpLst[i];
                return res;
            }
        }

        public int[] MinIndexes
        {
            get
            {
                double min = Min;
                List<int> tmpLst = new List<int>();
                for (int i = 0; i < dimension; i++)
                    if (Math.Abs(vector[i] - min) < Epsilon)
                        tmpLst.Add(i);
                int[] res = new int[tmpLst.Count];
                for (int i = 0; i < res.Length; i++)
                    res[i] = tmpLst[i];
                return res;
            }
        }

        public Vector Pow(double pow)
        {
            Vector res = new Vector(this);
            for (int i = 0; i < res.dimension; i++)
                res.vector[i] = Math.Pow(res.vector[i], pow);
            return res;
        }

        public static bool DimensionEquality(params Vector[] vectors)
        {
            int dim = vectors[0].Dimension;
            for (int i = 1; i < vectors.Length; i++)
                if (vectors[i].Dimension != dim)
                    return false;
            return true;
        }

        public static Vector NullVector(int dim = 16)
        {
            if (dim <= 0)
                throw new InvalidOperationException();
            return new Vector(dim, 0.0);
        }

        public static Vector OnesVector(int dim = 16)
        {
            if (dim <= 0)
                throw new InvalidOperationException();
            return new Vector(dim, 1.0); 
        }

        public static Vector UnitVector(int dim = 16, int index = 0)
        {
            if (dim <= 0 || index < 0 || index >= dim)
                throw new InvalidOperationException();
            Vector res = new Vector(dim);
            res.vector[index] = 1.0;
            return res;
        }

        public static Vector RandomVector(int dim = 16)
        {
            Random random = new Random();
            Vector res = new Vector(dim);
            for (int i = 0; i < dim; i++)
                res.vector[i] = random.NextDouble();
            return res;
        }

        public static Vector RandomUnitVector(int dim = 16)
        {
            Vector res = new Vector(RandomVector(dim));
            return res / res.EuclidNorm;
        }
    }
}