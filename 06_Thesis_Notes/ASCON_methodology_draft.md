# ASCON Methodology Draft

This study extended the profiled side-channel key recovery methodology to ASCON-AEAD-128. A selectable tight-trigger firmware was prepared so that the power leakage of one selected key byte could be isolated during the ASCON key-dependent initialization or key-injection operation. Random keys were used during profiling, and the selected key byte value was stored as ground truth. The collected traces were intended for training 256-class key-byte classifiers and evaluating Top-k candidate ranking.
