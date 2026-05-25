# ============================================================
# ASCON-AEAD-128 tight-trigger key-byte firmware
# Target board: ChipWhisperer-Lite XMEGA
# ============================================================

# Target output name
TARGET = simpleserial-ascon-keyabsorb

# Source files
SRC = main.c
SRC += encrypt.c

# Optimization level
OPT = s

# Target board / MCU
MCU = atxmega128d3
BOARD = CWLITEXMEGA

# ChipWhisperer firmware MCU path
FIRMWAREPATH = C:/Users/thetp/ChipWhisperer/chipwhisperer/firmware/mcu

# Do not use ChipWhisperer crypto library
CRYPTO_LIB =
CRYPTO_TARGET = NONE

# Enable trigger inside ascon_keybyte_absorb_tighttrigger()
# This is important for byte-wise tight-trigger dataset collection.
CFLAGS += -DASCON_ENABLE_KEYBYTE_TRIGGER

# Include ChipWhisperer build system
include $(FIRMWAREPATH)/Makefile.inc