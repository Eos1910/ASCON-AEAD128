# ============================================================
# ASCON-AEAD128 tight-trigger key-byte firmware
# Target board: ChipWhisperer-Lite XMEGA
# ============================================================

TARGET = simpleserial-ascon-keyabsorb

SRC = simpleserial-ascon-keyabsorb.c
SRC += encrypt.c

OPT = s

MCU = atxmega128d3
BOARD = CWLITEXMEGA

FIRMWAREPATH = /mnt/c/Users/thetp/ChipWhisperer/chipwhisperer/firmware/mcu

CRYPTO_LIB =
CRYPTO_TARGET = NONE

CFLAGS += -DASCON_ENABLE_KEYBYTE_TRIGGER

include $(FIRMWAREPATH)/Makefile.inc