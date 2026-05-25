# 01_collection

Main notebook:

`ASCON_AEAD128_AutoCollect_TightTrigger_30k_AllBytes.ipynb`

Run order:

1. Build/flash ASCON tight-trigger firmware.
2. Run notebook in TEST_MODE first.
3. Collect 100 traces for byte 00 and byte 01.
4. Verify trigger count and trace shape.
5. Collect all 16 bytes only after sanity checks pass.
