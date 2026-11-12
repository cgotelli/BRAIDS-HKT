"""Bounded wet/dry-AMR groin, inlet oracle, cancellation and exact restart."""
import os
from pathlib import Path
from stage_wong193 import OUT,TARGET
import stage_formula190
stage_formula190.OUT=OUT/os.environ.get('BRAIDS_RUN_TAG','qualification')
stage_formula190.TARGET=Path(os.environ.get('BRAIDS_TEST_RUNTIME_OVERRIDE',TARGET))
from run_formula190 import main
if __name__=='__main__':main()
