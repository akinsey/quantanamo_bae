import unittest # tests/test_main.py
import os
import subprocess
import glob  # Import to find files with a wildcard

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

    def test_any_png_generated(self):
        """Check if at least one .png file exists in the plots directory."""
        plot_dir = "plots"
        png_files = glob.glob(os.path.join(plot_dir, "*.png"))  # Find any .png files

        # Debugging: Print directory contents if no .png files found
        if not png_files:
            print(f"DEBUG: No .png files found in {plot_dir}. Listing directory contents:")
            print(os.listdir(plot_dir) if os.path.exists(plot_dir) else "Directory does not exist.")

        self.assertGreater(len(png_files), 0, "No plot images (.png) were generated.")

    @classmethod
    def tearDownClass(cls):
        """Clean up test files (optional)."""
        if os.path.exists(cls.output_log):
            os.remove(cls.output_log)

if __name__ == "__main__":
    unittest.main()
