using System;
using System.Drawing;
using System.Windows.Forms;

namespace SimplexMethod
{
    public partial class FormInitialConditions : Form
    {
        private bool maxObj, algorithmPrint;
        private int firstBoxLocation = 12, boxWidth = 35, boxHeight = 23, verticalBoxDistance = 20, distanceBetweenControls = 5,
            labelWidth = 30, labelHeight = 23, comboBoxWidth = 45, comboBoxHeight = 23, buttonHeight = 60, distanceBetweenButtons = 10;
        private ComboBox[] comboBoxesRelations;
        private ComboBox[] comboBoxesSigns;
        private ComboBox comboBoxMax;
        private TextBox[,] textBoxesLimitationMatrix;
        private TextBox[] textBoxesLimitationVector;
        private TextBox[] textBoxesObjectiveCoeffs;
        private Label[,] labels;
        private Label[] labelsObj;
        private Label[] labelsSigns;
        private Button continueButton;
        private Button toPreviousFormButton;
        public Form1 previousForm;

        public Matrix LimitationMatrix { get; private set; }
        public Vector LimitationVector { get; private set; }
        public Vector ObjectiveFunctionCoefficients{ get; private set; }
        public LinearProgrammingProblem Problem { get; private set; }
        public LinearProgrammingProblem CanonicalProblem { get; private set; }

        public short[] RelationArray { get; private set; }
        public short[] SignArray { get; private set; }

        public short MethodNumber { get; private set; }

        private void FormInitialConditions_Load(object sender, EventArgs e)
        {
            for (int j = 0; j < textBoxesObjectiveCoeffs.Length; j++)
                textBoxesObjectiveCoeffs[j].TabIndex = j;
            comboBoxMax.TabIndex = textBoxesObjectiveCoeffs[textBoxesObjectiveCoeffs.Length - 1].TabIndex + 1;
            for (int i = 0; i < textBoxesLimitationMatrix.GetLength(0); i++)
            {
                for (int j = 0; j < textBoxesLimitationMatrix.GetLength(1); j++)
                    textBoxesLimitationMatrix[i, j].TabIndex = comboBoxMax.TabIndex + (textBoxesLimitationMatrix.GetLength(1) + 2) * i + j + 1;
                comboBoxesRelations[i].TabIndex = textBoxesLimitationMatrix[i, textBoxesLimitationMatrix.GetLength(1) - 1].TabIndex + 1;
                textBoxesLimitationVector[i].TabIndex = comboBoxesRelations[i].TabIndex + 1;
            }
            for (int j = 0; j < textBoxesObjectiveCoeffs.Length; j++)
                comboBoxesSigns[j].TabIndex = textBoxesLimitationVector[textBoxesLimitationVector.Length - 1].TabIndex + 1 + j;
            int formWidth = Size.Width, tmp = firstBoxLocation;
            Controls.Add(comboBoxMax);
            for (int j = 0; j < textBoxesObjectiveCoeffs.Length; j++)
            {
                Controls.Add(textBoxesObjectiveCoeffs[j]);
                Controls.Add(labelsObj[j]);
                Controls.Add(labelsSigns[j]);
                Controls.Add(comboBoxesSigns[j]);
                tmp += labelsSigns[j].Width;
            }
            tmp += (2 * textBoxesObjectiveCoeffs.Length - 1) * distanceBetweenControls + textBoxesObjectiveCoeffs.Length * Convert.ToInt32(1.5 * comboBoxWidth) + firstBoxLocation;
            if (tmp > formWidth)
                formWidth = tmp;
            for (int i = 0; i < textBoxesLimitationMatrix.GetLength(0); i++)
            {
                tmp = firstBoxLocation;
                for (int j = 0; j < textBoxesLimitationMatrix.GetLength(1); j++)
                {
                    tmp += labels[i, j].Size.Width; 
                    Controls.Add(textBoxesLimitationMatrix[i, j]);
                    Controls.Add(labels[i, j]);
                }
                Controls.Add(comboBoxesRelations[i]);
                Controls.Add(textBoxesLimitationVector[i]);
                tmp += (2 * textBoxesLimitationMatrix.GetLength(1) + 5) * distanceBetweenControls + (textBoxesLimitationMatrix.GetLength(1) + 1) * boxWidth + comboBoxWidth;
                if (tmp > formWidth)
                    formWidth = tmp;
            }
            continueButton.Location = new Point(firstBoxLocation, labelsSigns[labelsSigns.Length - 1].Location.Y + comboBoxHeight + verticalBoxDistance);
            continueButton.Size = new Size((formWidth - 2 * firstBoxLocation - distanceBetweenButtons) / 2, buttonHeight);
            continueButton.Font = new Font("Microsoft Sans Serif", 15F);
            continueButton.Text = "Продолжить";
            continueButton.UseVisualStyleBackColor = true;
            continueButton.Click += new EventHandler(continueButton_Click);
            continueButton.TabIndex = comboBoxesSigns[comboBoxesSigns.Length - 1].TabIndex + 1;
            AcceptButton = continueButton;
            CancelButton = toPreviousFormButton;
            Controls.Add(continueButton);
            toPreviousFormButton.Location = new Point(firstBoxLocation + continueButton.Width + distanceBetweenButtons, continueButton.Location.Y);
            toPreviousFormButton.Size = new Size((formWidth - 2 * firstBoxLocation - distanceBetweenButtons) / 2, buttonHeight);
            toPreviousFormButton.Font = new Font("Microsoft Sans Serif", 15F);
            toPreviousFormButton.Text = "Назад";
            toPreviousFormButton.UseVisualStyleBackColor = true;
            toPreviousFormButton.Click += new EventHandler(toPreviousFormButton_Click);
            toPreviousFormButton.TabIndex = continueButton.TabIndex + 1;
            Controls.Add(toPreviousFormButton);
            Size = new Size(formWidth + 10,
                (textBoxesLimitationMatrix.GetLength(0) + 1) * verticalBoxDistance + comboBoxHeight +
                3 * firstBoxLocation + (textBoxesLimitationMatrix.GetLength(0) + 1) * boxHeight + 39 + continueButton.Size.Height);
            MaximumSize = MinimumSize = Size;
        }

        public FormInitialConditions(int variableNumber, int limitationNumber, short methodNumber, bool algorithmPrint, Form1 previousForm)
        {
            int tmp = 0;
            this.algorithmPrint = algorithmPrint;
            this.MethodNumber = methodNumber;
            this.previousForm = previousForm;
            LimitationMatrix = new Matrix(limitationNumber, variableNumber);
            LimitationVector = new Vector(limitationNumber);
            ObjectiveFunctionCoefficients = new Vector(variableNumber);
            RelationArray = new short[limitationNumber];
            SignArray = new short[variableNumber];
            Size = new Size((boxWidth + firstBoxLocation) * variableNumber + firstBoxLocation, 200);
            textBoxesLimitationMatrix = new TextBox[limitationNumber, variableNumber];
            labels = new Label[limitationNumber, variableNumber];
            comboBoxesRelations = new ComboBox[limitationNumber];
            textBoxesLimitationVector = new TextBox[limitationNumber];
            textBoxesObjectiveCoeffs = new TextBox[variableNumber];
            labelsObj = new Label[variableNumber];
            continueButton = new Button();
            toPreviousFormButton = new Button();
            for (int j = 0; j < variableNumber; j++)
            {
                textBoxesObjectiveCoeffs[j] = new TextBox();
                textBoxesObjectiveCoeffs[j].Size = new Size(boxWidth, boxHeight);
                textBoxesObjectiveCoeffs[j].Location = new Point(firstBoxLocation + j * (2 * distanceBetweenControls + boxWidth + labelWidth), firstBoxLocation);
                textBoxesObjectiveCoeffs[j].Text = "0";
                labelsObj[j] = new Label();
                labelsObj[j].Text = "x" + Convert.ToString(j + 1);
                labelsObj[j].Size = new Size(labelWidth, labelHeight);
                labelsObj[j].Location = new Point(textBoxesObjectiveCoeffs[j].Location.X + textBoxesObjectiveCoeffs[j].Size.Width + distanceBetweenControls, 
                    firstBoxLocation);
                if (j != variableNumber - 1)
                    labelsObj[j].Text += " + ";
                else
                    labelsObj[j].Text += " -> ";
            }
            for (int i = 0; i < limitationNumber; i++)
            {
                for (int j = 0; j < variableNumber; j++)
                {
                    textBoxesLimitationMatrix[i, j] = new TextBox();
                    textBoxesLimitationMatrix[i, j].Size = new Size(boxWidth, boxHeight);
                    labels[i, j] = new Label();
                    labels[i, j].Text = "x" + Convert.ToString(j + 1);
                    labels[i, j].Size = new Size(labelWidth, labelHeight);
                    if (j != variableNumber - 1)
                        labels[i, j].Text += " + ";
                    textBoxesLimitationMatrix[i, j].Location = new Point(firstBoxLocation + (boxWidth + labels[i, j].Size.Width + 2 * distanceBetweenControls) * j, 
                        2 * firstBoxLocation + boxHeight + i * (boxHeight + verticalBoxDistance));
                    textBoxesLimitationMatrix[i, j].Text = "0";
                    if (j == 0)
                        labels[i, j].Location = new Point(firstBoxLocation + boxWidth + distanceBetweenControls, 
                            2 * firstBoxLocation + boxHeight + i * (boxHeight + verticalBoxDistance));
                    else
                        labels[i, j].Location = new Point(labels[i, j - 1].Location.X + labels[i, j - 1].Size.Width + boxWidth + 2 * distanceBetweenControls, 
                            2 * firstBoxLocation + boxHeight + i * (boxHeight + verticalBoxDistance));
                }
                comboBoxesRelations[i] = new ComboBox();
                comboBoxesRelations[i].DropDownStyle = ComboBoxStyle.DropDownList;
                comboBoxesRelations[i].Size = new Size(comboBoxWidth, comboBoxHeight);
                comboBoxesRelations[i].Location = new Point(labels[i, variableNumber - 1].Location.X + labels[i, variableNumber - 1].Size.Width + distanceBetweenControls,
                    2 * firstBoxLocation + boxHeight + i * (boxHeight + verticalBoxDistance));
                comboBoxesRelations[i].Items.Add(">=");
                comboBoxesRelations[i].Items.Add("=");
                comboBoxesRelations[i].Items.Add("<=");
                textBoxesLimitationVector[i] = new TextBox();
                textBoxesLimitationVector[i].Size = new Size(boxWidth, boxHeight);
                textBoxesLimitationVector[i].Location = new Point(comboBoxesRelations[i].Location.X + comboBoxesRelations[i].Size.Width + 2 * distanceBetweenControls,
                     2 * firstBoxLocation + boxHeight + i * (boxHeight + verticalBoxDistance));
                textBoxesLimitationVector[i].Text = "0";
            }
            comboBoxMax = new ComboBox();
            comboBoxMax.DropDownStyle = ComboBoxStyle.DropDownList;
            comboBoxMax.Size = new Size(comboBoxWidth, comboBoxHeight);
            tmp = 0;
            for (int j = 0; j < variableNumber; j++)
                tmp += labelsObj[j].Width;
            comboBoxMax.Location = new Point(comboBoxesRelations[0].Location.X, firstBoxLocation);
            comboBoxMax.Items.Add("max");
            comboBoxMax.Items.Add("min");
            labelsSigns = new Label[variableNumber];
            comboBoxesSigns = new ComboBox[variableNumber];
            for (int j = 0; j < variableNumber; j++)
            {
                labelsSigns[j] = new Label();
                labelsSigns[j].Text = "x" + Convert.ToString(j + 1);
                labelsSigns[j].Size = new Size(labelHeight, labelWidth);
                if (j == 0)
                    labelsSigns[j].Location = new Point(firstBoxLocation, 
                    textBoxesLimitationMatrix[limitationNumber - 1, j].Location.Y + verticalBoxDistance + boxHeight);
                else
                    labelsSigns[j].Location = new Point(labelsSigns[j - 1].Location.X  + labelsSigns[j - 1].Width + 2 * distanceBetweenControls + Convert.ToInt32(1.5 * comboBoxWidth),
                    textBoxesLimitationMatrix[limitationNumber - 1, j].Location.Y + verticalBoxDistance + boxHeight);
                comboBoxesSigns[j] = new ComboBox();
                comboBoxesSigns[j].DropDownStyle = ComboBoxStyle.DropDownList;
                comboBoxesSigns[j].Size = new Size(Convert.ToInt32(1.5 * comboBoxWidth), comboBoxHeight);
                comboBoxesSigns[j].Location = new Point(labelsSigns[j].Location.X + labelsSigns[j].Width + distanceBetweenControls, labelsSigns[j].Location.Y);
                comboBoxesSigns[j].Items.Add(" >= 0");
                comboBoxesSigns[j].Items.Add(" <= 0");
                comboBoxesSigns[j].Items.Add(" not ");
            }
            InitializeComponent();
        }

        private void continueButton_Click(object sender, EventArgs e)
        {
            int i = 0, j = 0;
            try
            {
                for (j = 0; j < textBoxesObjectiveCoeffs.Length; j++)
                    ObjectiveFunctionCoefficients[j] = Convert.ToDouble(textBoxesObjectiveCoeffs[j].Text);
            }
            catch (FormatException)
            {
                MessageBox.Show("Коэффицинт введенный при x" + Convert.ToString(j + 1) + " в формуле целевой функции не является числом!");
                textBoxesObjectiveCoeffs[j].Text = "0";
                return;
            }
            catch (OverflowException)
            {
                MessageBox.Show("Коэффицинт введенный при x" + Convert.ToString(j + 1) + " в формуле целевой функции, к сожалению, слишком велик!");
                textBoxesObjectiveCoeffs[j].Text = "0";
                return;
            }
            catch (Exception)
            {
                MessageBox.Show("Во время введения коэффициента при x" + Convert.ToString(j + 1) + " в формуле целевой функции возникла неизвестная ошибка!");
                textBoxesObjectiveCoeffs[j].Text = "0";
                return;
            }
            if (comboBoxMax.Text.Replace(" ", "") == "max")
                maxObj = true;
            else if (comboBoxMax.Text.Replace(" ", "") == "min")
                maxObj = false;
            else
            {
                MessageBox.Show("Не указан экстремум целевой функции!");
                return;
            }
            try
            {
                for (i = 0; i < textBoxesLimitationMatrix.GetLength(0); i++)
                    for (j = 0; j < textBoxesLimitationMatrix.GetLength(1); j++)
                        LimitationMatrix[i, j] = Convert.ToDouble(textBoxesLimitationMatrix[i, j].Text);
            }
            catch(FormatException)
            {
                MessageBox.Show("Коэффицинт введенный при x" + Convert.ToString(j + 1) + " в уравнении/неравенстве №" + Convert.ToString(i + 1) + " не является числом!");
                textBoxesLimitationMatrix[i, j].Text = "0";
                return;
            }
            catch(OverflowException)
            {
                MessageBox.Show("Коэффицинт введенный при x" + Convert.ToString(j + 1) + " в уравнении/неравенстве №" + Convert.ToString(i + 1) + ", к сожалению, слишком велик!");
                textBoxesLimitationMatrix[i, j].Text = "0";
                return;
            }
            catch(Exception)
            {
                MessageBox.Show("Во время введения коэффициента при x" + Convert.ToString(j + 1) + " в уравнении/неравенстве №" + Convert.ToString(i + 1) + " возникла неизвестная ошибка!");
                textBoxesLimitationMatrix[i, j].Text = "0";
                return;
            }
            try
            {
                for (i = 0; i < textBoxesLimitationVector.Length; i++)
                    LimitationVector[i] = Convert.ToDouble(textBoxesLimitationVector[i].Text);
            }
            catch(FormatException)
            {
                MessageBox.Show("Свободный член введенный в уравнении/неравенстве №" + Convert.ToString(i + 1) + " не является числом!");
                textBoxesLimitationVector[i].Text = "0";
                return;
            }
            catch (OverflowException)
            {
                MessageBox.Show("Свободный член введенный в уравнении/неравенстве №" + Convert.ToString(i + 1) + ", к сожалению, слишком велик!");
                textBoxesLimitationVector[i].Text = "0";
                return;
            }
            catch (Exception)
            {
                MessageBox.Show("Во время введения свободного члена в уравнении/неравенстве №" + Convert.ToString(i + 1) + " возникла неизвестная ошибка!");
                textBoxesLimitationVector[i].Text = "0";
                return;
            }
            for (i = 0; i < comboBoxesRelations.Length; i++)
            {
                if (comboBoxesRelations[i].Text == "")
                {
                    MessageBox.Show("Не указан тип отношения в уравнении/неравенстве №" + Convert.ToString(i + 1) + "!");
                    return;
                }
            }
            for (i = 0; i < RelationArray.Length; i++)
            {
                if (comboBoxesRelations[i].Text.Replace(" ", "") == "<=")
                    RelationArray[i] = -1;
                else if (comboBoxesRelations[i].Text.Replace(" ", "") == ">=")
                    RelationArray[i] = 1;
                else
                    RelationArray[i] = 0;
            }
            for (j = 0; j < comboBoxesSigns.Length; j++)
            {
                if (comboBoxesSigns[j].Text == "")
                {
                    MessageBox.Show("Не указан знак для переменной x" + Convert.ToString(j + 1) + "!");
                    return;
                }
            }
            for (j = 0; j < SignArray.Length; j++)
            {
                if (comboBoxesSigns[j].Text.Replace(" ", "") == "<=0")
                    SignArray[j] = -1;
                else if (comboBoxesSigns[j].Text.Replace(" ", "") == ">=0")
                    SignArray[j] = 1;
                else
                    SignArray[j] = 0;
            }
            try
            {
                Problem = new LinearProgrammingProblem(LimitationMatrix, LimitationVector, ObjectiveFunctionCoefficients,
                    algorithmPrint, maxObj, RelationArray, SignArray);
            }
            catch(ArgumentException)
            {
                MessageBox.Show("Введенная ЗЛП не имеет смысла, или содержит ограничения, которые, очевидно, тождественно истинные или ложные!");
                return;
            }
            CanonicalProblem = new LinearProgrammingProblem(Problem.EqualCanonicalProblem);
            if (MethodNumber == 0)
            {
                Hide();
                DefaultBasisInput basisInput = new DefaultBasisInput(CanonicalProblem, this);
                basisInput.Show();
            }
            else if(MethodNumber == 1)
            {
                Hide();
                DefaultSolutionInput solutionInput = new DefaultSolutionInput(CanonicalProblem, this);
                solutionInput.Show();
            }
            else
            {
                Hide();
                ResultForm result = new ResultForm(Problem, this);
                result.Show();
            }
        }

        private void toPreviousFormButton_Click(object sender, EventArgs e)
        {
            Close();
            previousForm.Show();
        }
    }
}
