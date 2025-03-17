import unittest
import os
import subprocess
import glob

class TestMainExecution(unittest.TestCase):
    """
    Unit tests to validate main program execution and output generation.
    """

    def run_main_with_flags(self, strategy="SMA", stock="AAPL", use_ai=True, plot=False):
        """
        Helper function to run the main program with different command-line arguments.
        """
        cmd = ["python", "main.py", "--strategy", strategy, "--stock", stock]

        if not use_ai:
            cmd.append("--disable-ai")
        if plot:
            cmd.append("--plot")

        output_log = "tests/output.log"
        os.makedirs("tests", exist_ok=True)  # Ensure tests directory exists

        with open(output_log, "w") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

        return output_log, result.returncode  # Return log file path and exit code

    def test_default_execution(self):
        """
        Test default execution (SMA strategy, AAPL stock, AI enabled, no plotting).
        """
        output_log, exit_code = self.run_main_with_flags()
        self.assertEqual(exit_code, 0, "Program did not exit cleanly")

        with open(output_log, "r") as f:
            logs = f.read()

        self.assertIn("We hope you enjoyed your stay", logs, "Exit message not found!")

    def test_macd_strategy(self):
        """
        Test execution with MACD strategy.
        """
        output_log, exit_code = self.run_main_with_flags(strategy="MACD")
        self.assertEqual(exit_code, 0, "Program did not exit cleanly")

    def test_rsi_strategy(self):
        """
        Test execution with RSI strategy.
        """
        output_log, exit_code = self.run_main_with_flags(strategy="RSI")
        self.assertEqual(exit_code, 0, "Program did not exit cleanly")

    def test_disable_ai(self):
        """
        Test execution with AI disabled.
        """
        output_log, exit_code = self.run_main_with_flags(use_ai=False)
        self.assertEqual(exit_code, 0, "Program did not exit cleanly")

    def test_plotting_enabled(self):
        """
        Test execution with plotting enabled.
        """
        output_log, exit_code = self.run_main_with_flags(plot=True)
        self.assertEqual(exit_code, 0, "Program did not exit cleanly")

        # Verify PNG files were created
        plot_dir = "plots"
        png_files = glob.glob(os.path.join(plot_dir, "*.png"))  # Find all .png files

        if not png_files:
            print(f"DEBUG: No .png files found in {plot_dir}. Listing directory contents:")
            print(os.listdir(plot_dir) if os.path.exists(plot_dir) else "Directory does not exist.")

        self.assertGreater(len(png_files), 0, "No plot images (.png) were generated.")

    def test_all_flags_combined(self):
        """
        Test execution with all flags set (MACD strategy, custom stock, AI disabled, plotting enabled).
        """
        output_log, exit_code = self.run_main_with_flags(strategy="MACD", stock="TSLA", use_ai=False, plot=True)
        self.assertEqual(exit_code, 0, "Program did not exit cleanly")

        # Verify PNG files were created
        plot_dir = "plots"
        png_files = glob.glob(os.path.join(plot_dir, "*.png"))  # Find all .png files
        self.assertGreater(len(png_files), 0, "No plot images (.png) were generated.")

    def test_invalid_strategy(self):
        """
        Test execution with an invalid strategy (should fail).
        """
        output_log, exit_code = self.run_main_with_flags(strategy="INVALID_STRATEGY")
        self.assertNotEqual(exit_code, 0, "Program should fail on an invalid strategy")

    @classmethod
    def tearDownClass(cls):
        """
        Clean up test files after execution.
        """
        output_log = "tests/output.log"
        if os.path.exists(output_log):
            os.remove(output_log)

if __name__ == "__main__":
    unittest.main()
