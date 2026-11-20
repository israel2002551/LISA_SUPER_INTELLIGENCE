#pragma once
#include <stdint.h>
#include "gemma4_w25q128_config.h"

// ==============================================================================
// W25Q128 (16 MiB) partition map for the edge node
// ==============================================================================
//   0x000000  4 KiB   boot control, copy 0 ┐ ping-pong: the copy with the higher
//   0x001000  4 KiB   boot control, copy 1 ┘ valid sequence number wins
//   0x002000  56 KiB  reserved
//   0x010000  6 MiB   Partition A  model slot 0
//   0x610000  6 MiB   Partition B  model slot 1 (OTA staging when A is active)
//   0xC10000  ~3.9 MiB Partition C  ring log (interactions, device profiles, errors)
//
// NOTE: this differs from the chat build (gemma4_w25q128_cluster), which keeps a
// single image at address 0. Re-provision the chip when switching builds.
// ==============================================================================

#define W25_SECTOR_BYTES     0x1000u
#define W25_CTRL_ADDR_0      0x000000u
#define W25_CTRL_ADDR_1      0x001000u
#define W25_SLOT_A_ADDR      0x010000u
#define W25_SLOT_BYTES       0x600000u
#define W25_SLOT_B_ADDR      (W25_SLOT_A_ADDR + W25_SLOT_BYTES)
#define W25_RING_ADDR        (W25_SLOT_B_ADDR + W25_SLOT_BYTES)
#define W25_RING_END         0x1000000u
#define W25_RING_SECTORS     ((W25_RING_END - W25_RING_ADDR) / W25_SECTOR_BYTES)

static_assert((W25_SLOT_A_ADDR % 0x10000u) == 0 && (W25_SLOT_B_ADDR % 0x10000u) == 0, "slots must be 64 KiB aligned");
static_assert(W25_MODEL_HEADER_BYTES + GEMMA_MODEL_PAYLOAD_BYTES <= W25_SLOT_BYTES, "model image must fit a slot");
static_assert(W25_RING_ADDR < W25_RING_END && W25_RING_SECTORS >= 256, "ring log needs at least 1 MiB");

static inline uint32_t w25_slot_addr(int slot) { return slot == 0 ? W25_SLOT_A_ADDR : W25_SLOT_B_ADDR; }
