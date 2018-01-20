using System;
using System.Drawing;
using System.Windows.Forms;

namespace SimplexMethod
{
    public partial class DefaultSolutionInput : Form
    {
        private FormInitialConditions previousForm;
        private LinearProgrammingProblem canonicalProblem;
        private int firstControlLocation = 12, verticalDistanceBetweenControls = 10, buttonHeight = 60, distanceBetweenButtons = 10, 
            horizontalDistanceBetweenControls = 5, textBoxWidth = 35, textBoxHeight = 23;
        private Vector defaultSolution;
        private Label titleLabel1, titleLabel2;
        private Label canonicalProblemLabel;
        private Label[] defaultSolutionLabels;
        private Button continueButton;
        private Button toPreviousFormButton;
        private TextBox[] defaultSolutionTextBoxes;

        private void DefaultSolutionInput_Load(object sender, EventArgs e)
        {
            int formWidth = 0;
            titleLabel1 = new Label();
            titleLabel1.Location = new Point(firstControlLocation, firstControlLocation);
            titleLabel1.Text = "Каноническая форма задачи, введенной вами:";
            titleLabel1.AutoSize = true;
            titleLabel1.Font = new Font("Microsoft Sans Serif", 10F);
            canonicalProblemLabel = new Label();
            canonicalProblemLabel.Location = new Point(firstControlLocation, firstControlLocation + titleLabel1.Height + verticalDistanceBetweenControls);
            canonicalProblemLabel.Text = canonicalProblem.ToString();
            canonicalProblemLabel.AutoSize = true;
            canonicalProblemLabel.Font = new Font("Microsoft Sans Serif", 10F);
            titleLabel2 = new Label();
            titleLabel2.Location = new Point(firstControlLocation, canonicalProblemLabel.Location.Y + (canonicalProblem.LimitationNumber + 2) * canonicalProblemLabel.Height + verticalDistanceBetweenControls);
            titleLabel2.Text = "Пожалуйста, введите допустимое опорное решение:";
            titleLabel2.AutoSize = true;
            titleLabel2.Font = new Font("Microsoft Sans Serif", 10F);
            defaultSolutionTextBoxes = new TextBox[canonicalProblem.VariableNumber];
            defaultSolutionLabels = new Label[canonicalProblem.VariableNumber + 1];
            defaultSolutionLabels[0] = new Label();
            defaultSolutionLabels[0].Location = new Point(firstControlLocation, titleLabel2.Location.Y + titleLabel2.Height + verticalDistanceBetweenControls);
            defaultSolutionLabels[0].Text = canonicalProblem.VariableSymbol + "* = (";
            defaultSolutionLabels[0].AutoSize = true;
            defaultSolutionLabels[0].Font = new Font("Microsoft Sans Serif", 10F);
            for(int j = 0; j < defaultSolutionTextBoxes.Length; j++)
            {
                defaultSolutionTextBoxes[j] = new TextBox();
                defaultSolutionTextBoxes[j].Location = new Point(defaultSolutionLabels[j].Location.X + defaultSolutionLabels[j].Width + horizontalDistanceBetweenControls,
                    defaultSolutionLabels[0].Location.Y);
                defaultSolutionTextBoxes[j].Size = new Size(textBoxWidth, textBoxHeight);
                defaultSolutionTextBoxes[j].Text = "0";
                defaultSolutionLabels[j + 1] = new Label();
                defaultSolutionLabels[j + 1].Location = new Point(defaultSolutionTextBoxes[j].Location.X + defaultSolutionTextBoxes[j].Width + horizontalDistanceBetweenControls, 
                    defaultSolutionLabels[0].Location.Y);
                defaultSolutionLabels[j + 1].Text = j == defaultSolutionTextBoxes.Length - 1 ? ")" : ";";
                defaultSolutionLabels[j + 1].AutoSize = true;
                defaultSolutionLabels[j + 1].Font = new Font("Microsoft Sans Serif", 10F);
            }
            formWidth = 2 * firstControlLocation + defaultSolutionLabels[0].Width;
            for (int j = 0; j < defaultSolutionTextBoxes.Length; j++)
                formWidth += 2 * horizontalDistanceBetweenControls + textBoxWidth + defaultSolutionLabels[j + 1].Width;
            continueButton = new Button();
            continueButton.Location = new Point(firstControlLocation, defaultSolutionLabels[0].Location.Y + textBoxHeight + verticalDistanceBetweenControls);
            continueButton.Size = new Size((formWidth - 2 * firstControlLocation - distanceBetweenButtons) / 2, buttonHeight);
            continueButton.Font = new Font("Microsoft Sans Serif", 15F);
            continueButton.Text = "Продолжить";
            continueButton.UseVisualStyleBackColor = true;
            continueButton.Click += new EventHandler(continueButton_Click);
            toPreviousFormButton = new Button();
            toPreviousFormButton.Location = new Point(firstControlLocation + continueButton.Width + distanceBetweenButtons, continueButton.Location.Y);
            toPreviousFormButton.Size = new Size((formWidth - 2 * firstControlLocation - distanceBetweenButtons) / 2, buttonHeight);
            toPreviousFormButton.Font = new Font("Microsoft Sans Serif", 15F);
            toPreviousFormButton.Text = "Назад";
            toPreviousFormButton.UseVisualStyleBackColor = true;
            toPreviousFormButton.Click += new EventHandler(toPreviousFormButton_Click);
            Text = "Ввод исходного опорного решения";
            AutoSize = true;
            Controls.Add(titleLabel1);
            Controls.Add(canonicalProblemLabel);
            Controls.Add(titleLabel2);
            foreach (TextBox textBox in defaultSolutionTextBoxes)
                Controls.Add(textBox);
            foreach (Label label in defaultSolutionLabels)
                Controls.Add(label);
            Controls.Add(continueButton);
            Controls.Add(toPreviousFormButton);
        }

        public DefaultSolutionInput(LinearProgrammingProblem canonicalProblem, FormInitialConditions previousForm)
        {
            this.canonicalProblem = canonicalProblem;
            this.previousForm = previousForm;
            defaultSolution = new Vector(canonicalProblem.VariableNumber);
            InitializeComponent();
        }


        private void continueButton_Click(object sender, EventArgs e)
        {
            int j = 0;
            try
            {
                for (j = 0; j < defaultSolution.Dimension; j++)
                    defaultSolution[j] = Convert.ToDouble(defaultSolutionTextBoxes[j].Text);
            }
            catch(FormatException)
            {
                MessageBox.Show("Введенная вами " + Convert.ToString(j + 1) + "-я компонента не является числом!");
                defaultSolutionTextBoxes[j].Text = "0";
                return;
            }
            catch(OverflowException)
            {
                MessageBox.Show("Введенная вами " + Convert.ToString(j + 1) + "-я компонента, к сожалению, является слишком большой!");
                defaultSolutionTextBoxes[j].Text = "0";
                return;
            }
            catch(Exception)
            {
                MessageBox.Show("Во время ввода " + Convert.ToString(j + 1) + "-й компоненты произошла неизвестная ошибка!");
                defaultSolutionTextBoxes[j].Text = "0";
                return;
            }
            try
            {
                canonicalProblem.SetDefaultBasisSolution(defaultSolution);
            }
            catch(ArgumentException)
            {
                MessageBox.Show("Введенный вектор, к сожалению, не может быть опорным!");
                return;
            }
            ResultForm result = new ResultForm(canonicalProblem, previousForm);
            Close();
            result.Show();
        }

        private void toPreviousFormButton_Click(object sender, EventArgs e)
        {
            Close();
            previousForm.Show();
        }
    }
}
