using System;
using System.Windows.Forms;

namespace SimplexMethod
{
    public partial class UserGuide : Form
    {
        Form1 baseForm;
        public UserGuide(Form1 baseForm)
        {
            this.baseForm = baseForm;
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Hide();
            baseForm.Show();
        }
    }
}
