"""Reuse the frozen-input groin protocol with the new law; no legacy run mutation."""
import os
from pathlib import Path
from stage_camenen192 import ROOT,OUT,TARGET
import stage_formula190
stage_formula190.OUT=OUT/os.environ.get('BRAIDS_RUN_TAG','qualification')
stage_formula190.TARGET=Path(os.environ.get('BRAIDS_TEST_RUNTIME_OVERRIDE',TARGET)).resolve()
import run_formula190
if __name__=='__main__':run_formula190.main()
