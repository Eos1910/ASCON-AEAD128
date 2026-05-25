# simpleserial-ascon-keyabsorb

This folder mirrors ACORN's `tight_trigger_keybyte/simpleserial-acorn-keyabsorb` folder.

Expected SimpleSerial interface:

- `k`: 16-byte ASCON key
- `n`: 16-byte ASCON nonce
- `p`: 1-byte target key byte index, 0 to 15
- `r`: 1-byte checksum/status response

Important:
The current implementation is a scaffold. Replace the placeholder ASCON key-byte operation with a real selected-byte operation from ASCON initialization or finalization before final collection.
