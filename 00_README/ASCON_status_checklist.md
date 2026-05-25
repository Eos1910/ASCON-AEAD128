# ASCON_FYP Status Checklist

## Firmware

- [ ] Real ASCON-AEAD-128 C implementation added to `01_Firmware/full_ascon`.
- [ ] Full ASCON firmware builds for CWLITEXMEGA.
- [ ] Tight-trigger firmware builds for CWLITEXMEGA.
- [ ] SimpleSerial commands confirmed: `k`, `n`, `p`, `r`.
- [ ] Trigger count is non-zero for target bytes 0 and 1.

## Collection

- [ ] Run collection notebook in TEST_MODE.
- [ ] Verify byte_00 and byte_01 traces.
- [ ] Verify metadata columns.
- [ ] Collect all 16 bytes only after test mode passes.

## Training

- [ ] Verify dataset completeness.
- [ ] Train byte 00 first.
- [ ] Train bytes 00 to 15.
- [ ] Compare Top-k results with ACORNv3.
