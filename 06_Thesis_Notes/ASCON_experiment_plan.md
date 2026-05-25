# ASCON-AEAD-128 Experiment Plan

## Objective

Extend the ACORNv3 key-byte recovery methodology to ASCON-AEAD-128.

## Proposed workflow

1. Port or add real ASCON-AEAD-128 C implementation.
2. Build full ASCON firmware and confirm encryption works.
3. Build tight-trigger selected key-byte firmware.
4. Collect test-mode traces for byte 00 and byte 01.
5. Verify traces and metadata.
6. Train byte 00 model first.
7. Collect full 30k-per-byte dataset only if leakage is learnable.
