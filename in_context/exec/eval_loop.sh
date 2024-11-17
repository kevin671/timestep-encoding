
#!/bin/sh
#PJM -L rscgrp=share-debug
#PJM -L gpu=1
#PJM -g gg45
#PJM -j
#PJM --fs /work

source /work/gg45/g45004/.bashrc
conda activate loop_tf
mkdir -p /work/gg45/g45004/timestep-encoding/in_context/results2/Figures

python scripts/eval_loop.py