using System;
using System.Windows.Forms;

namespace SimplexMethod
{
    public partial class Form1 : Form
    {
        public int LimitationNumber { get; private set; }
        public short MethodNumber { get; private set; }
        public bool PrintAlgorithms { get; private set; }
        public int VariableNumber { get; private set; }

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            MinimumSize = MaximumSize = Size;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            try
            {
                VariableNumber = Convert.ToInt32(textBox1.Text);
            }
            catch(FormatException)
            {
                MessageBox.Show("Введенное значение для количества переменных не является натуральным числом!");
                textBox1.Text = "";
                return;
            }
            catch(OverflowException)
            {
                MessageBox.Show("Введенное значение, к сожалению, слишком велико!");
                textBox1.Text = "";
                return;
            }
            catch(Exception)
            {
                MessageBox.Show("Неизвестная ошибка!");
                textBox1.Text = "";
                return;
            }
            if (VariableNumber <= 0)
            {
                MessageBox.Show("Введенное значение для количества переменных не является натуральным числом!");
                textBox1.Text = "";
                return;
            }
            try
            {
                LimitationNumber = Convert.ToInt32(textBox2.Text);
            }
            catch (FormatException)
            {
                MessageBox.Show("Введенное значение для количества ограничений не является натуральным числом!");
                textBox2.Text = "";
                return;
            }
            catch (OverflowException)
            {
                MessageBox.Show("Введенное значение, к сожалению, слишком велико!");
                textBox2.Text = "";
                return;
            }
            catch (Exception)
            {
                MessageBox.Show("Неизвестная ошибка!");
                textBox2.Text = "";
                return;
            }
            if (!radioButton1.Checked && !radioButton2.Checked && !radioButton3.Checked)
            {
                MessageBox.Show("Выберите один из методов!");
                return;
            }
            if (LimitationNumber <= 0)
            {
                MessageBox.Show("Введенное значение для количества ограничений не является натуральным числом!");
                textBox2.Text = "";
                return;
            }
            else if (radioButton1.Checked)
                MethodNumber = 0;
            else if (radioButton2.Checked)
                MethodNumber = 1;
            else
                MethodNumber = 2;
            PrintAlgorithms = checkBox1.Checked;
            Hide();
            FormInitialConditions initForm = new FormInitialConditions(VariableNumber, LimitationNumber, MethodNumber, PrintAlgorithms, this);
            initForm.Show();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            Hide();
            UserGuide guide = new UserGuide(this);
            guide.Show();
        }
    }
}
