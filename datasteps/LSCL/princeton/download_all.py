# builds every file under data/datasteps/LSCL/princeton/. run from repo root: python -m datasteps.LSCL.princeton.download_all

import datasteps.LSCL.princeton.kvr as kvr
import datasteps.LSCL.princeton.popqa as popqa
import datasteps.LSCL.princeton.triviaqa as triviaqa
import datasteps.LSCL.princeton.lama as lama
import datasteps.LSCL.princeton.entityquestions as entityquestions

STEPS = [kvr, popqa, triviaqa, lama, entityquestions]  # [module with main()]

for step in STEPS:
    print(f"=== {step.__name__.rsplit('.', 1)[1]} ===")
    step.main()
