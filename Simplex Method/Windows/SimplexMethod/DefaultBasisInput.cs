using System;
using System.Drawing;
using System.Windows.Forms;

namespace SimplexMethod
{
    public partial class DefaultBasisInput : Form
    {
        FormInitialConditions previousForm;
        LinearProgrammingProblem canonicalProblem;
        int [] defaultBasisIndexes;
        int firstControlLocation = 12, verticalDistanceBetweenControls = 10, buttonHeight = 60, distanceBetweenButtons = 10;
        private Label titleLabel1, titleLabel2, titleLabel3;
        private Label canonicalProblemLabel;
        private Label limitationVectorLabel;
        private CheckBox[] vectorCheckBoxes;
        private Button continueButton;
        private Button toPreviousFormButton;

        private void DefaultBasisInput_Load(object sender, EventArgs e)
        {
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
            titleLabel2.Text = "Векторы, образованные в данной ЗЛП";
            titleLabel2.AutoSize = true;
            titleLabel2.Font = new Font("Microsoft Sans Serif", 10F);
            vectorCheckBoxes = new CheckBox[canonicalProblem.VariableNumber];
            limitationVectorLabel = new Label();
            limitationVectorLabel.Location = new Point(firstControlLocation, titleLabel2.Location.Y + titleLabel2.Height + 2 * verticalDistanceBetweenControls);
            limitationVectorLabel.Text = "A0: " + Convert.ToString(canonicalProblem.LimitationVector);
            limitationVectorLabel.AutoSize = true;
            limitationVectorLabel.Font = new Font("Microsoft Sans Serif", 10F);
            for (int j = 0; j < vectorCheckBoxes.Length; j++)
            {
                vectorCheckBoxes[j] = new CheckBox();
                if (j == 0)
                    vectorCheckBoxes[j].Location = new Point(firstControlLocation, limitationVectorLabel.Location.Y + limitationVectorLabel.Height + verticalDistanceBetweenControls);
                else
                    vectorCheckBoxes[j].Location = new Point(firstControlLocation, vectorCheckBoxes[j - 1].Location.Y + vectorCheckBoxes[j - 1].Height + verticalDistanceBetweenControls);
                vectorCheckBoxes[j].Text = "A" + Convert.ToString(j + 1) + ": " + Convert.ToString(canonicalProblem.LimitationMatrix.GetColumn(j));
                vectorCheckBoxes[j].AutoSize = true;
                vectorCheckBoxes[j].Font = new Font("Microsoft Sans Serif", 10F);
                vectorCheckBoxes[j].CheckedChanged += new EventHandler(checkBoxes_CheckedChanged);
            }
            titleLabel3 = new Label();
            titleLabel3.Location = new Point(firstControlLocation, vectorCheckBoxes[vectorCheckBoxes.Length - 1].Location.Y +
                vectorCheckBoxes[vectorCheckBoxes.Length - 1].Height + verticalDistanceBetweenControls);
            titleLabel3.Text = "Пожалуйста выберите вектора, которые будут базисными.";
            titleLabel3.AutoSize = true;
            titleLabel3.Font = new Font("Microsoft Sans Serif", 10F);
            AutoSize = true;
            continueButton = new Button();
            continueButton.Location = new Point(firstControlLocation, titleLabel3.Location.Y + titleLabel3.Height + verticalDistanceBetweenControls);
            continueButton.Size = new Size((Width - 2 * firstControlLocation - distanceBetweenButtons), buttonHeight);
            continueButton.Font = new Font("Microsoft Sans Serif", 15F);
            continueButton.Text = "Продолжить";
            continueButton.UseVisualStyleBackColor = true;
            continueButton.Click += new EventHandler(continueButton_Click);
            toPreviousFormButton = new Button();
            toPreviousFormButton.Location = new Point(firstControlLocation + continueButton.Width + distanceBetweenButtons, continueButton.Location.Y);
            toPreviousFormButton.Size = new Size((Width - 2 * firstControlLocation - distanceBetweenButtons) / 2, buttonHeight);
            toPreviousFormButton.Font = new Font("Microsoft Sans Serif", 15F);
            toPreviousFormButton.Text = "Назад";
            toPreviousFormButton.UseVisualStyleBackColor = true;
            toPreviousFormButton.Click += new EventHandler(toPreviousFormButton_Click);
            Text = "Ввод исходного базиса";
            AcceptButton = continueButton;
            CancelButton = toPreviousFormButton;
            Controls.Add(titleLabel1);
            Controls.Add(canonicalProblemLabel);
            Controls.Add(titleLabel2);
            Controls.Add(limitationVectorLabel);
            foreach (CheckBox checkBox in vectorCheckBoxes)
                Controls.Add(checkBox);
            Controls.Add(titleLabel3);
            Controls.Add(continueButton);
            Controls.Add(toPreviousFormButton);
            MaximumSize = MinimumSize = Size;
        }

        public DefaultBasisInput(LinearProgrammingProblem canonicalProblem, FormInitialConditions previousForm)
        {
            this.canonicalProblem = canonicalProblem;
            this.previousForm = previousForm;
            defaultBasisIndexes = new int[canonicalProblem.LimitationNumber];
            InitializeComponent();
        }

        private void checkBoxes_CheckedChanged(object sender, EventArgs e)
        {
            int checkedCount = 0;
            foreach (CheckBox checkBox in vectorCheckBoxes)
            {
                if (checkBox.Checked)
                    checkedCount++;
            }
            if (checkedCount == canonicalProblem.LimitationNumber)
            {
                foreach (CheckBox checkBox in vectorCheckBoxes)
                {
                    if (!checkBox.Checked)
                        checkBox.Enabled = false;
                }
            }
            else if (checkedCount < canonicalProblem.LimitationNumber)
            {
                foreach (CheckBox checkBox in vectorCheckBoxes)
                {
                    checkBox.Enabled = true;
                }
            }
        }

        private void continueButton_Click(object sender, EventArgs e)
        {
            int checkedCount = 0;
            foreach (CheckBox checkBox in vectorCheckBoxes)
            {
                if (checkBox.Checked)
                    checkedCount++;
            }
            if(checkedCount != canonicalProblem.LimitationNumber)
            {
                MessageBox.Show("Вы указали недостаточное количество векторов, для того, чтобы они образовывали базис!");
                return;
            }
            for (int i = 0, indexBasis = 0; i < vectorCheckBoxes.Length; i++)
                if (vectorCheckBoxes[i].Checked)
                    defaultBasisIndexes[indexBasis++] = i;
            try
            {
                canonicalProblem.SetDefaultBasis(defaultBasisIndexes);
            }
            catch (ArgumentException)
            {
                MessageBox.Show("К сожалению указанные вами вектора линейно-зависимы и не могут образовывать базис!");
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
