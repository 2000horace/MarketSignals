import unittest

if __name__ == "__main__":
    # Discover all tests in the 'tests/unit' directory
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("tests/unit", pattern="test_*.py")

    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_runner.run(test_suite)