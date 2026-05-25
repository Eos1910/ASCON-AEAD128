# ASCON_FYP

This folder mirrors the ACORN_FYP structure and is prepared for ASCON-AEAD-128 side-channel experiments using ChipWhisperer-Lite 2-Part with XMEGA.

## Main goal

The intended experiment is a profiled power side-channel key-byte recovery study:

1. Implement ASCON-AEAD-128 on XMEGA.
2. Add selectable tight-trigger firmware for one key byte at a time.
3. Collect random-key traces with known labels.
4. Train 256-class key-byte models.
5. Evaluate Top-k candidate ranking and key-space reduction.

## Important warning

The included firmware files are scaffolds. They are not a complete verified ASCON-AEAD-128 implementation yet. Replace the placeholder crypto implementation with a real ASCON-AEAD-128 implementation before collecting final thesis data.
