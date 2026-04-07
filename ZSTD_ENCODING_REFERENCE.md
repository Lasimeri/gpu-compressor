# Zstd FSE Sequence Encoding — Exact Bit-Level Specification

## 1. Overview

Zstd encodes sequences (Literal Length, Match Length, Offset) using Finite State Entropy (FSE). This document provides the exact encoding specification from RFC 8878 and the Facebook zstd reference implementation, critical for GPU compression.

**Key Facts:**
- Sequences are encoded in **reverse** (LIFO stack order)
- First sequence encoded is **last** to be decoded
- Bitstream is written forward, read backward
- FSE states are flushed AFTER all sequences are encoded
- Extra bits are interleaved with FSE symbols in a specific order

---

## 2. RFC 8878 — Sequence Encoding Specification

### 2.1 Sequence Data Structure

From RFC 8878 Section 3.1.1.3:

A sequence consists of three components:
- **Literal Length (LL)**: Number of literal bytes before the match
- **Match Length (ML)**: Length of the match (minimum 3 bytes, stored as ML - 3)
- **Offset (OF)**: Distance to the match source

The offset uses **repcode encoding**:
- Offsets 1, 2, 3 are special: they reference the last 3 used offsets (repeat offsets)
- Raw offsets start at 4

### 2.2 Sequence Code Tables

Each sequence value (LL, ML, OF) is mapped to a **code** (0-35) via a fixed table. The code determines:
1. **Baseline value** (BL): Starting value
2. **Number of extra bits** (NB): How many bits to read from bitstream

#### Literal Length Code Table (RFC 8878 §3.1.1.3.2.1.1)

```
Code   Baseline   Number_of_Bits
0-15   0-15       0
16     16         1
17     18         1
18     20         1
19     22         1
20     24         2
21     28         2
22     32         3
23     40         3
24     48         4
25     64         6
26     128        7
27     256        8
28     512        9
29     1024       10
30     2048       11
31     4096       12
32     8192       13
33     16384      14
34     32768      15
35     65536      16
```

#### Match Length Code Table (RFC 8878 §3.1.1.3.2.1.1)

```
Code   Baseline   Number_of_Bits
0-31   3-34       0
32     35         1
33     37         1
34     39         1
35     41         1
36     43         2
37     47         2
38     51         3
39     59         3
40     67         4
41     83         6
42     131        7
43     259        8
44     515        9
45     1027       10
46     2051       11
47     4099       12
48     8195       13
49     16387      14
50     32771      15
51     65539      16
52     131075     17 (not used in standard)
```

#### Offset Code Table (RFC 8878 §3.1.1.3.2.1.1)

```
Code   Baseline   Number_of_Bits
0      0          0
1      1          0
2      2          0
3      3          0
4      4          1
5      6          1
6      8          2
7      12         2
8      16         3
9      24         3
10     32         4
11     48         4
12     64         5
13     96         5
14     128        6
15     192        6
16     256        7
17     384        7
18     512        8
19     768        8
20     1024       9
21     1536       9
22     2048       10
23     3072       10
24     4096       11
25     6144       11
26     8192       12
27     12288      12
28     16384      13
29     24576      13
30     32768      14
31     49152      14
```

### 2.3 Sequence Execution Flow

From RFC 8878 §3.1.1.3.2.1 (Sequence Execution / Decoding):

1. **Build FSE tables** from Literal Length, Match Length, and Offset FSE descriptions
2. **Initialize FSE states** with the last sequence's codes (encoded first, decoded last)
3. **Process sequences in reverse** (decoder reads backward, encoder wrote forward)
4. For each sequence:
   - Decode LL code → look up baseline + read extra bits → final LL value
   - Decode ML code → look up baseline + read extra bits → final ML value  
   - Decode OF code → look up baseline + read extra bits → final OF value (or repcode)
5. **Flush final FSE states** (sentinel bits)

---

## 3. Facebook/Meta Zstd Reference Implementation

### 3.1 ZSTD_encodeSequences_body() Function

**Location:** `lib/compress/zstd_compress_sequences.c`

This is the authoritative encoder. Key structure:

```c
FORCE_INLINE_TEMPLATE size_t
ZSTD_encodeSequences_body(
    void* dst, size_t dstCapacity,
    FSE_CTable const* CTable_MatchLength, BYTE const* mlCodeTable,
    FSE_CTable const* CTable_OffsetBits, BYTE const* ofCodeTable,
    FSE_CTable const* CTable_LitLength, BYTE const* llCodeTable,
    SeqDef const* sequences, size_t nbSeq, int longOffsets)
```

**Critical Algorithm:**

```
1. Initialize bitstream:
   BIT_initCStream(&blockStream, dst, dstCapacity)

2. Initialize FSE states with LAST sequence (index nbSeq-1):
   FSE_initCState2(&stateMatchLength, CTable_MatchLength, mlCodeTable[nbSeq-1])
   FSE_initCState2(&stateOffsetBits, CTable_OffsetBits, ofCodeTable[nbSeq-1])
   FSE_initCState2(&stateLitLength, CTable_LitLength, llCodeTable[nbSeq-1])

3. Add LAST sequence's extra bits (before FSE symbols):
   BIT_addBits(&blockStream, sequences[nbSeq-1].litLength, LL_bits[llCodeTable[nbSeq-1]])
   if (MEM_32bits()) BIT_flushBits(&blockStream)
   BIT_addBits(&blockStream, sequences[nbSeq-1].mlBase, ML_bits[mlCodeTable[nbSeq-1]])
   if (MEM_32bits()) BIT_flushBits(&blockStream)
   BIT_addBits(&blockStream, sequences[nbSeq-1].offBase, OF_bits[ofCodeTable[nbSeq-1]])
   BIT_flushBits(&blockStream)

4. For each remaining sequence (n = nbSeq-2 down to 0):
   a. Encode FSE symbols (in order LL, OF, ML):
      FSE_encodeSymbol(&blockStream, &stateLitLength, llCodeTable[n])
      FSE_encodeSymbol(&blockStream, &stateOffsetBits, ofCodeTable[n])
      FSE_encodeSymbol(&blockStream, &stateMatchLength, mlCodeTable[n])
   
   b. Add extra bits (in order LL, ML, OF):
      BIT_addBits(&blockStream, sequences[n].litLength, LL_bits[llCodeTable[n]])
      BIT_addBits(&blockStream, sequences[n].mlBase, ML_bits[mlCodeTable[n]])
      BIT_addBits(&blockStream, sequences[n].offBase, OF_bits[ofCodeTable[n]])
   
   c. Flush periodically to prevent register overflow

5. Flush FSE final states:
   FSE_flushCState(&blockStream, &stateMatchLength)
   FSE_flushCState(&blockStream, &stateOffsetBits)
   FSE_flushCState(&blockStream, &stateLitLength)

6. Close bitstream (adds sentinel 1-bit):
   BIT_closeCStream(&blockStream)
```

### 3.2 FSE_encodeSymbol() Implementation

**Location:** `lib/common/fse.h`

```c
MEM_STATIC void FSE_encodeSymbol(BIT_CStream_t* bitC, FSE_CState_t* statePtr, unsigned symbol)
{
    FSE_symbolCompressionTransform const symbolTT = 
        ((const FSE_symbolCompressionTransform*)(statePtr->symbolTT))[symbol];
    const U16* const stateTable = (const U16*)(statePtr->stateTable);
    
    // Calculate how many bits to output from current state
    U32 const nbBitsOut = (U32)((statePtr->value + symbolTT.deltaNbBits) >> 16);
    
    // Output lower nbBitsOut bits of current state value
    BIT_addBits(bitC, (BitContainerType)statePtr->value, nbBitsOut);
    
    // Transition to next state (based on symbol)
    statePtr->value = stateTable[(statePtr->value >> nbBitsOut) + symbolTT.deltaFindState];
}
```

### 3.3 FSE_flushCState() Implementation

```c
MEM_STATIC void FSE_flushCState(BIT_CStream_t* bitC, const FSE_CState_t* statePtr)
{
    // Output all remaining bits of final state
    BIT_addBits(bitC, (BitContainerType)statePtr->value, statePtr->stateLog);
    BIT_flushBits(bitC);
}
```

### 3.4 BIT_addBits() — The Core Bit Packing

**Location:** `lib/common/bitstream.h`

```c
MEM_STATIC void BIT_addBits(BIT_CStream_t* bitC, BitContainerType value, unsigned nbBits)
{
    // CRITICAL: Bits are accumulated in a register from LSB to MSB
    // value is shifted left by current bit position and OR'ed in
    assert(nbBits < BIT_MASK_SIZE);
    assert(nbBits + bitC->bitPos < sizeof(bitC->bitContainer) * 8);
    
    // Shift value left by current position, OR into container
    bitC->bitContainer |= BIT_getLowerBits(value, nbBits) << bitC->bitPos;
    bitC->bitPos += nbBits;
}
```

**Key Detail:** Bits are added to the **right** (lower positions) of the register. This is LSB-first accumulation.

### 3.5 BIT_flushBits() — Writing to Memory

```c
MEM_STATIC void BIT_flushBits(BIT_CStream_t* bitC)
{
    size_t const nbBytes = bitC->bitPos >> 3;  // Number of complete bytes
    
    // Write little-endian to memory
    MEM_writeLEST(bitC->ptr, bitC->bitContainer);
    bitC->ptr += nbBytes;
    
    // Keep remaining bits (0-7)
    bitC->bitPos &= 7;
    bitC->bitContainer >>= nbBytes*8;
}
```

**Critical:** Uses **little-endian** byte order. Bytes are written in forward order; bits within bytes are LSB-first.

### 3.6 BIT_closeCStream() — Sequence Terminator

```c
MEM_STATIC size_t BIT_closeCStream(BIT_CStream_t* bitC)
{
    BIT_addBitsFast(bitC, 1, 1);  // Add sentinel 1-bit
    BIT_flushBits(bitC);
    if (bitC->ptr >= bitC->endPtr) return 0; // overflow
    return (size_t)(bitC->ptr - bitC->startPtr) + (bitC->bitPos > 0);
}
```

**Sentinel Bit:** A final 1-bit is added to mark end of sequence bitstream. Decoder uses this to detect when all bits have been consumed.

---

## 4. Bit-Level Bitstream Format

### 4.1 Encoder's Forward Writing

The encoder writes bits in **forward** direction (left to right in memory):

```
Memory layout (increasing byte addresses →):
┌──────────────┬──────────────┬──────────────┐
│   Byte N     │  Byte N+1    │  Byte N+2    │
├──────────────┼──────────────┼──────────────┤
│ Bits 7-0     │ Bits 7-0     │ Bits 7-0     │
└──────────────┴──────────────┴──────────────┘
  (LSB first)    (LSB first)    (LSB first)
```

**Within each byte:** Bits 0-7 are written LSB-first (bit 0 is written first, bit 7 last).

### 4.2 Decoder's Backward Reading

The decoder reads bits in **backward** direction (right to left in memory):

```
Initialization:
- ptr = srcBuffer + srcSize - sizeof(BitContainer)
- Loads bitContainer from memory at ptr
- bitsConsumed tracks how many bits have been consumed from container
- Last byte determines sentinel bit position

Reading bits:
- Extract from high bits of current container
- Shift down as bits consumed
- When container exhausted, reload from ptr-=nbBytes
```

**Key:** Decoder reads the **last** bytes first, then earlier bytes. Within each byte, it reads from high bits (bit 7) to low bits (bit 0).

### 4.3 The Critical Relationship

**Encoding path:**
```
Bit stream (conceptually): [1st-bit] [2nd-bit] ... [Nth-bit] [sentinel=1]
Memory order (forward):     Byte0     Byte1     ... ByteK    (last byte)
Within-byte (LSB-first):    0→7       0→7       ... 0→7
```

**Decoding path:**
```
Read order:                [sentinel=1] ... [Nth-bit] [2nd-bit] [1st-bit]
Memory order (backward):   ByteK       ... Byte1     Byte0     (first byte)
Within-byte (high→low):    detect 1    ... 7→0      7→0       7→0
```

### 4.4 Example: Encoding Two Bits "10" (Binary)

Encoder writes: `10` (binary 0b10 = 2 decimal)

```
Initial register: 0x00
After addBits(2, 2):
  bitContainer = 0x02 (bits = 0b10 in positions 0-1)
  bitPos = 2

Next addBits(5, 3):
  bitContainer = 0x2A (bits = 0b101010, 5 shifted left by 2 = 0b10100, OR with 0x02)
  bitPos = 5

BIT_flushBits() writes:
  MEM_writeLEST(ptr, 0x2A)  → Memory byte = 0x2A
  ptr+=1, bitPos=0, bitContainer=0
```

Decoder reads backward from that byte:
```
Loads last byte: 0x2A = 0b00101010
bitsConsumed counts from highest bit
Extract 2 bits: 0b10 = 2 ✓
Extract 3 bits: 0b101 = 5 ✓
```

---

## 5. Sequence Encoding Step-by-Step Walkthrough

### Example: Three Sequences

Given input sequences:
```
Seq[0]: LL=5, ML=4, OF=8
Seq[1]: LL=10, ML=5, OF=16
Seq[2]: LL=3, ML=3, OF=1
```

Assume FSE tables exist and produce these codes:
```
LL Code Table: LL 0-5 → Code 0, LL 10 → Code 10, LL 3 → Code 3
ML Code Table: ML 3 → Code 0, ML 4 → Code 1, ML 5 → Code 2
OF Code Table: OF 1 → Code 1, OF 8 → Code 6, OF 16 → Code 10
```

And baseline/bits (from tables above):
```
LL_bits[0]=0, LL_bits[3]=0, LL_bits[10]=0
ML_bits[0]=0, ML_bits[1]=0, ML_bits[2]=0
OF_bits[1]=0, OF_bits[6]=2, OF_bits[10]=4
```

#### Step 1: Initialize with Last Sequence (Seq[2])

```
llCodeTable[2] = 3  (LL=3)
mlCodeTable[2] = 0  (ML=3)
ofCodeTable[2] = 1  (OF=1, repcode)

FSE_initCState2(&stateLitLength, CTable_LitLength, 3)
FSE_initCState2(&stateMatchLength, CTable_MatchLength, 0)
FSE_initCState2(&stateOffsetBits, CTable_OffsetBits, 1)
```

#### Step 2: Encode Last Sequence's Extra Bits

```
LL code 3 → LL_bits[3] = 0 bits to add
ML code 0 → ML_bits[0] = 0 bits to add
OF code 1 → OF_bits[1] = 0 bits to add
(No extra bits for this sequence)
```

#### Step 3: Encode Middle Sequence (Seq[1])

```
a) Encode FSE symbols:
   FSE_encodeSymbol(&stream, &stateLitLength, 10)   → Variable bits (from table)
   FSE_encodeSymbol(&stream, &stateOffsetBits, 10)  → Variable bits
   FSE_encodeSymbol(&stream, &stateMatchLength, 2)  → Variable bits

b) Add extra bits:
   LL_bits[10] = 0 bits
   ML_bits[2] = 0 bits
   OF_bits[10] = 4 bits → BIT_addBits(&stream, 16, 4) → Actually adds 0b0000 (16 fits in 4 bits)
   
c) Flush if needed
```

#### Step 4: Encode First Sequence (Seq[0])

```
a) Encode FSE symbols:
   FSE_encodeSymbol(&stream, &stateLitLength, 0)
   FSE_encodeSymbol(&stream, &stateOffsetBits, 6)
   FSE_encodeSymbol(&stream, &stateMatchLength, 1)

b) Add extra bits:
   LL_bits[0] = 0 bits
   ML_bits[1] = 0 bits
   OF_bits[6] = 2 bits → BIT_addBits(&stream, 8, 2) → Adds 0b00 (from 8)

c) Flush
```

#### Step 5: Flush FSE States

```
FSE_flushCState(&stream, &stateMatchLength)  → Outputs stateLog bits
FSE_flushCState(&stream, &stateOffsetBits)
FSE_flushCState(&stream, &stateLitLength)
```

#### Step 6: Close with Sentinel

```
BIT_closeCStream() → Adds sentinel 1-bit, flushes
```

#### Result Bitstream (Conceptual)

```
Written forward (to memory, LSB-first within bytes):
[FSE(Seq[2])] [FSE(Seq[1])+Extras] [FSE(Seq[0])+Extras] [FSE-states] [Sentinel=1]

Decoded backward:
[Sentinel=1] ← [FSE-states] ← [FSE(Seq[0])+Extras] ← [FSE(Seq[1])+Extras] ← [FSE(Seq[2])]
```

---

## 6. Extra Bits Encoding Details

### 6.1 How Extra Bits are Used

For each sequence type (LL, ML, OF), the **code** determines both the baseline and how many extra bits follow.

**Example: LL code 24**
- LL_bits[24] = 4 (number of extra bits)
- LL_base[24] = 48 (baseline)
- Final LL value = 48 + (4-bit value read from bitstream)
- If 4 bits = 0b0111 = 7, then LL = 48 + 7 = 55

### 6.2 Offset Encoding: Special Case

Offsets have special handling in the encoder:

```c
if (longOffsets) {
    U32 const ofBits = ofCodeTable[nbSeq-1];
    unsigned const extraBits = ofBits - MIN(ofBits, STREAM_ACCUMULATOR_MIN-1);
    if (extraBits) {
        BIT_addBits(&blockStream, sequences[nbSeq-1].offBase, extraBits);
        BIT_flushBits(&blockStream);  // Force flush
    }
    BIT_addBits(&blockStream, sequences[nbSeq-1].offBase >> extraBits,
                ofBits - extraBits);
} else {
    BIT_addBits(&blockStream, sequences[nbSeq-1].offBase, ofCodeTable[nbSeq-1]);
}
```

**Key:** For long offsets (>2^31), the offset bits may be split and flushed separately.

### 6.3 ML and LL Encoding

For Match Length, the stored value is **mlBase** = matchLength - 3 (since minimum match is 3 bytes).

```c
BIT_addBits(&blockStream, sequences[n].mlBase, ML_bits[mlCodeTable[n]])
```

Similar for Literal Length (stored directly).

---

## 7. FSE State Encoding

### 7.1 FSE State Representation

Each FSE state is a 16-bit value in the state table.

```c
typedef struct {
    ptrdiff_t value;        // Current state value
    const void* stateTable; // Table of next states
    const void* symbolTT;   // Symbol transition table
    unsigned stateLog;      // Log of table size (usually 5-9)
} FSE_CState_t;
```

### 7.2 State Initialization (FSE_initCState2)

```c
MEM_STATIC void FSE_initCState2(FSE_CState_t* statePtr, const FSE_CTable* ct, U32 symbol)
{
    FSE_initCState(statePtr, ct);
    {
        const FSE_symbolCompressionTransform symbolTT = 
            ((const FSE_symbolCompressionTransform*)(statePtr->symbolTT))[symbol];
        const U16* stateTable = (const U16*)(statePtr->stateTable);
        
        // Initialize to smallest state for this symbol
        U32 nbBitsOut = (U32)((symbolTT.deltaNbBits + (1<<15)) >> 16);
        statePtr->value = (nbBitsOut << 16) - symbolTT.deltaNbBits;
        statePtr->value = stateTable[(statePtr->value >> nbBitsOut) + symbolTT.deltaFindState];
    }
}
```

### 7.3 State Output (FSE_flushCState)

When flushing, the full state value is output:

```c
MEM_STATIC void FSE_flushCState(BIT_CStream_t* bitC, const FSE_CState_t* statePtr)
{
    BIT_addBits(bitC, (BitContainerType)statePtr->value, statePtr->stateLog);
    BIT_flushBits(bitC);
}
```

For example, if stateLog = 6, then 6 bits of the state value are written to indicate the final state.

---

## 8. Critical Implementation Notes for GPU Encoding

### 8.1 Bit Ordering (THE MOST COMMON BUG)

**WRONG:**
```c
// Incorrect: bits written MSB-first
bitContainer |= value << (bitPos ^ (sizeof(bitContainer)*8 - 1));  // WRONG!
```

**CORRECT:**
```c
// Correct: bits written LSB-first, accumulate from lower positions
bitContainer |= BIT_getLowerBits(value, nbBits) << bitPos;  // RIGHT
```

### 8.2 FSE State Register Overflow

The bitstream register must be flushed when approaching maximum:

```c
if (nbBits + bitPos >= 64 - stateLog) {
    BIT_flushBits(&blockStream);
}
```

Failing to flush causes state bits to overwrite data bits.

### 8.3 Sentinel Bit Requirement

The last bit of the sequence bitstream MUST be a 1-bit:

```c
BIT_addBits(bitC, 1, 1);  // Sentinel
BIT_flushBits(bitC);
```

The decoder looks for the highest set bit in the last byte to determine how many bits are valid. Omitting this causes bitstream corruption.

### 8.4 FSE State Ordering

States are flushed in order: **ML, OF, LL** (from the code):

```c
FSE_flushCState(&blockStream, &stateMatchLength);  // First
FSE_flushCState(&blockStream, &stateOffsetBits);   // Second
FSE_flushCState(&blockStream, &stateLitLength);    // Third
```

Decoder reads them in **reverse** during initialization, so LL state is read first.

### 8.5 Register Width Differences

On 32-bit systems:
- STREAM_ACCUMULATOR_MIN = 25 bits
- Flushes required more frequently

On 64-bit systems:
- STREAM_ACCUMULATOR_MIN = 57 bits
- More data can accumulate before flush

Use `MEM_32bits()` macro to branch if needed.

---

## 9. Common Pitfalls in Custom Implementations

### Pitfall 1: Wrong Bit Order

**Symptom:** FSE table descriptions decode correctly, but sequences have garbage values.

**Cause:** Bits written in wrong order (MSB vs LSB first).

**Fix:** Always use LSB-first accumulation. Verify by encoding then decoding a known sequence.

### Pitfall 2: Missing Flushes

**Symptom:** FSE states overwrite sequence bits, producing random corruption.

**Cause:** Not flushing bitstream before outputting FSE states.

**Fix:** Insert explicit `BIT_flushBits()` before `FSE_flushCState()`.

### Pitfall 3: Sentinel Bit Omitted

**Symptom:** Bitstream decodes but decoder doesn't know when sequence ends.

**Cause:** No 1-bit sentinel added by `BIT_closeCStream()`.

**Fix:** Ensure final code is:
```c
BIT_addBits(stream, 1, 1);
BIT_flushBits(stream);
```

### Pitfall 4: Sequence Order Reversed

**Symptom:** Sequence values are correct but in reverse order.

**Cause:** Iterating forward instead of backward through sequences.

**Fix:** Loop from `n = nbSeq-2` **down to** 0:
```c
for (n = nbSeq-2; n < nbSeq; n--) {  // Loop wraps, counts down
```

### Pitfall 5: Offset Code Encoding Wrong

**Symptom:** Offsets are off by 3, or repcode offsets (1-3) are treated as raw offsets.

**Cause:** Not understanding offset encoding (raw offset = value - ZSTD_REP_NUM).

**Fix:** Verify:
- Repcode 1-3: stored as 1-3 directly
- Raw offset N: stored as N + ZSTD_REP_NUM (= N + 3)

---

## 10. References

- **RFC 8878** — Zstandard Compression Format (February 2021)
  - Section 3.1.1.3.2.1 — Sequences Section
  - Section 3.1.1.3.2.1.1 — Sequence Codes for Lengths and Offsets
  - Section 4.1 — FSE Table Description
  - Section 4.2 — Sequence Encoding

- **Facebook/Meta zstd GitHub** — https://github.com/facebook/zstd
  - `lib/compress/zstd_compress_sequences.c` — ZSTD_encodeSequences_body()
  - `lib/common/fse.h` — FSE state encoding (FSE_encodeSymbol, FSE_flushCState)
  - `lib/common/bitstream.h` — BIT_addBits, BIT_flushBits, BIT_closeCStream

- **Nigel Tao Zstandard Worked Examples** — https://nigeltao.github.io
  - Part 5: Finite State Entropy Codes
  - Part 6: Sequences

---

## Appendix: Verification Checklist

To verify your GPU FSE sequence encoder is correct, test these in order:

- [ ] **Encode single sequence with all zeros (LL=0, ML=3, OF=1)** — Should produce minimal bitstream
- [ ] **Encode sequence and verify backward** — Decode the bitstream, check values match
- [ ] **Test bit order** — Manually trace one FSE symbol encoding, verify bits in memory match expected
- [ ] **Check sentinel bit** — Last bit of sequence section must be 1
- [ ] **Verify FSE state flush** — After encoding, FSE states must appear in bitstream
- [ ] **Test multiple sequences** — Ensure reverse iteration produces correct order
- [ ] **Test offset encoding** — Verify repcode (1-3) vs raw offset (4+) distinction
- [ ] **Test long offsets** — If using longOffsets=1, verify split extra bits handling
- [ ] **Test 32-bit register behavior** — If implementing 32-bit mode, verify extra flushes
- [ ] **Cross-check with reference decoder** — Run decoder on your encoded streams

