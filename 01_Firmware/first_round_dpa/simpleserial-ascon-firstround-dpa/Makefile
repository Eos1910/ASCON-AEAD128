# ============================================================
# ASCON-AEAD128 first-round DPA/CPA firmware
# Target board: ChipWhisperer-Lite XMEGA
# ============================================================

TARGET = simpleserial-ascon-firstround-dpa

SRC = simpleserial-ascon-firstround-dpa.c
SRC += encrypt.c

OPT = s

MCU = atxmega128d3
BOARD = CWLITEXMEGA

FIRMWAREPATH = /mnt/c/Users/thetp/ChipWhisperer/chipwhisperer/firmware/mcu

CRYPTO_LIB =
CRYPTO_TARGET = NONE

CFLAGS += -DASCON_ENABLE_FIRSTROUND_DPA

include $(FIRMWAREPATH)/Makefile.inc