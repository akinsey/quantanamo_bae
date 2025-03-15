import unittest
import os
import subprocess

class TestMainExecution(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Run the main program and capture output."""
        cls.output_log = "tests/output.log"
        os.makedirs("tests", exist_ok=True)  # Ensure tests folder exists

        with open(cls.output_log, "w") as f:
            result = subprocess.run(["python", "main.py"], stdout=f, stderr=subprocess.STDOUT)
            cls.exit_code = result.returncode  # Capture exit code

    def test_exit_message(self):
        """Check if the exit message appears in output."""
        with open(self.output_log, "r") as f:
            logs = f.read()
        self.assertIn("We hope you enjoyed your stay", logs, "Exit message not found in logs!")

    def test_plot_image_exists(self):
        """Check if the plot image was generated."""
        plot_path = "plots/WMT_strategy.png"
        self.assertTrue(os.path.exists(plot_path), f"Plot image not generated: {plot_path}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test files (optional)."""
        if os.path.exists(cls.output_log):
            os.remove(cls.output_log)

if __name__ == "__main__":
    unittest.main()
