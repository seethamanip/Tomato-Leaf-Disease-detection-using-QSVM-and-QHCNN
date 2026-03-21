import os
import sys
import runpy

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Run the training module
runpy.run_module('src.train_qsvm', run_name='__main__')
