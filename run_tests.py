import unittest

_DIR_PARENT = 'tests/integration'

if __name__ == "__main__":

    # Discover all tests in the 'tests/unit' directory
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(_DIR_PARENT, pattern="test_*.py")

    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_runner.run(test_suite)