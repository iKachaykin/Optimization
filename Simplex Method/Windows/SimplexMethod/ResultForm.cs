using System;
using System.Drawing;
using System.Windows.Forms;

namespace SimplexMethod
{
    public partial class ResultForm : Form
    {
        private int firstControlLocation = 12, verticalDistanceBetweenControls = 10, 
            distanceBetweenButtons = 5, buttonHeight = 60, formWidth = 0;
        private LinearProgrammingProblem problem, givenProblem;
        private FormInitialConditions formWithProblem;
        private Label Label1, Label2, Label3;
        private Label givenProblemLabel;
        private Label solutionLabel;
        private Label objectiveFunctionValueLabel;
        private Vector solution;
        private Button toPreviousFormButton;
        private Button toStartFormButton;
        private Button exitButton;
        
        public void ResultForm_Load(object sender, EventArgs e)
        {
            Label1 = new Label();
            Label1.Location = new Point(firstControlLocation, firstControlLocation);
            Label1.Text = "Первоначальная задача линейного программирования:";
            Label1.Font = new Font("Microsoft Sans Serif", 10F);
            Label1.AutoSize = true;
            givenProblemLabel = new Label();
            givenProblemLabel.Location = new Point(firstControlLocation, Label1.Location.Y + Label1.Height + verticalDistanceBetweenControls);
            givenProblemLabel.Text = Convert.ToString(givenProblem);
            givenProblemLabel.Font = new Font("Microsoft Sans Serif", 10F);
            givenProblemLabel.AutoSize = true;
            Label2 = new Label();
            Label2.Location = new Point(firstControlLocation, givenProblemLabel.Location.Y + 
                (givenProblem.LimitationNumber + 1)* givenProblemLabel.Height + verticalDistanceBetweenControls);
            Label2.Text = "Значение аргумента, в котором функция достигает своего экстремума:";
            Label2.Font = new Font("Microsoft Sans Serif", 10F);
            Label2.AutoSize = true;
            if (formWithProblem.MethodNumber == 0 || formWithProblem.MethodNumber == 1)
                solution = givenProblem.TrimCanonicalSolution(problem.Solution);
            else
                solution = problem.Solution;
            solutionLabel = new Label();
            solutionLabel.Location = new Point(firstControlLocation, Label2.Location.Y + Label2.Height + verticalDistanceBetweenControls);
            solutionLabel.Text = solution != null ? "x* = " + Convert.ToString(solution) : "введенная ЗЛП имеет допустимых решений.";
            solutionLabel.Font = new Font("Microsoft Sans Serif", 10F);
            solutionLabel.AutoSize = true;
            Label3 = new Label();
            Label3.Location = new Point(firstControlLocation, solutionLabel.Location.Y + solutionLabel.Height + verticalDistanceBetweenControls);
            Label3.Text = "Оптимальное значение целевой функции:";
            Label3.Font = new Font("Microsoft Sans Serif", 10F);
            Label3.AutoSize = true;
            objectiveFunctionValueLabel = new Label();
            objectiveFunctionValueLabel.Location = new Point(firstControlLocation, Label3.Location.Y + Label3.Height + verticalDistanceBetweenControls);
            objectiveFunctionValueLabel.Text = solution != null ? "f(x*) = " + Convert.ToString(solution * givenProblem.ObjectiveFunctionCoefficients) : "отсутствует.";
            objectiveFunctionValueLabel.Font = new Font("Microsoft Sans Serif", 10F);
            objectiveFunctionValueLabel.AutoSize = true;
            formWidth = 0;
            Controls.Add(Label1);
            Controls.Add(givenProblemLabel);
            Controls.Add(Label2);
            Controls.Add(solutionLabel);
            Controls.Add(Label3);
            Controls.Add(objectiveFunctionValueLabel);
            toPreviousFormButton = new Button();
            toPreviousFormButton.Location = new Point(firstControlLocation, objectiveFunctionValueLabel.Location.Y + objectiveFunctionValueLabel.Height + verticalDistanceBetweenControls);
            toPreviousFormButton.Font = new Font("Microsoft Sans Serif", 15F);
            toPreviousFormButton.Text = "Назад";
            toPreviousFormButton.UseVisualStyleBackColor = true;
            toPreviousFormButton.Click += new EventHandler(toPreviousFormButton_Click);
            toStartFormButton = new Button();
            toStartFormButton.Font = new Font("Microsoft Sans Serif", 15F);
            toStartFormButton.Text = "В начало";
            toStartFormButton.UseVisualStyleBackColor = true;
            toStartFormButton.Click += new EventHandler(toStartFormButton_Click);
            exitButton = new Button();
            exitButton.Font = new Font("Microsoft Sans Serif", 15F);
            exitButton.Text = "Выход";
            exitButton.UseVisualStyleBackColor = true;
            exitButton.Click += new EventHandler(exitButton_Click);
            if (givenProblem.AlgorithmPrint)
            {
                int count = 1;
                DataGridView simplexTables = new DataGridView();
                simplexTables.Location = new Point(firstControlLocation, objectiveFunctionValueLabel.Location.Y +
                    objectiveFunctionValueLabel.Height + verticalDistanceBetweenControls);
                simplexTables.Size = new Size(100 * problem.AllSimplexTables[0].GetLength(1) + 60, 
                    25 * (problem.AllSimplexTables[0].GetLength(0) + 2));
                for (int j = 0; j < problem.AllSimplexTables[0].GetLength(1); j++)
                {
                    simplexTables.Columns.Add("", "");
                    simplexTables.Columns[j].ReadOnly = true;
                }
                foreach (string[,] table in problem.AllSimplexTables)
                {
                    simplexTables.Rows.Add("Итерация №" + Convert.ToString(count++));
                    for (int i = 0; i < table.GetLength(0); i++)
                        simplexTables.Rows.Add(GetRow(table, i));
                }
                simplexTables.TabIndex = 4;
                Controls.Add(simplexTables);
                formWidth = simplexTables.Width + 2 * firstControlLocation;
                toPreviousFormButton.Location = new Point(firstControlLocation, simplexTables.Location.Y + simplexTables.Height + verticalDistanceBetweenControls);
            }
            formWidth = Max(new int[] {formWidth,
                Max(new int[] {Label1.Width, givenProblemLabel.Width, Label2.Width, solutionLabel.Width, Label3.Width, objectiveFunctionValueLabel.Width})});
            toPreviousFormButton.Size = new Size((formWidth - 2 * firstControlLocation - 2 * distanceBetweenButtons) / 3, buttonHeight);
            toStartFormButton.Size = new Size((formWidth - 2 * firstControlLocation - 2 * distanceBetweenButtons) / 3, buttonHeight);
            exitButton.Size = new Size((formWidth - 2 * firstControlLocation - 2 * distanceBetweenButtons) / 3, buttonHeight);
            toStartFormButton.Location = new Point(toPreviousFormButton.Location.X + toPreviousFormButton.Width + distanceBetweenButtons,
                toPreviousFormButton.Location.Y);
            exitButton.Location = new Point(toStartFormButton.Location.X + toStartFormButton.Width + distanceBetweenButtons,
                toStartFormButton.Location.Y);
            toStartFormButton.TabIndex = 1;
            toPreviousFormButton.TabIndex = 2;
            exitButton.TabIndex = 3;
            AcceptButton = toStartFormButton;
            CancelButton = toPreviousFormButton;
            Controls.Add(toPreviousFormButton);
            Controls.Add(toStartFormButton);
            Controls.Add(exitButton);
            AutoSize = true;
            MaximumSize = MinimumSize = Size;
        }

        public ResultForm(LinearProgrammingProblem problem, FormInitialConditions formWithProblem)
        {
            this.problem = problem;
            this.formWithProblem = formWithProblem;
            givenProblem = formWithProblem.Problem;
            InitializeComponent();
        }

        private void toPreviousFormButton_Click(object sender, EventArgs e)
        {
            Close();
            formWithProblem.Show();
        }

        private void toStartFormButton_Click(object sender, EventArgs e)
        {
            Hide();
            formWithProblem.previousForm.Show();
        }

        private void exitButton_Click(object sender, EventArgs e)
        {
            formWithProblem.previousForm.Close();
            formWithProblem.Close();
            Close();
            Application.Exit();
        }
        private string [] GetRow(string[,] table, int index)
        {
            string[] result = new string[table.GetLength(1)];
            for (int j = 0; j < table.GetLength(1); j++)
                result[j] = table[index, j];
            return result;
        }

        private string [] GetColumn(string[,] table, int index)
        {
            string[] result = new string[table.GetLength(0)];
            for (int i = 0; i < table.GetLength(0); i++)
                result[i] = table[i, index];
            return result;
        }

        private int Max(int[] array)
        {
            int max = array[0];
            for (int i = 1; i < array.Length; i++)
                if (array[i] > max)
                    max = array[i];
            return max;
        }
    }
}
