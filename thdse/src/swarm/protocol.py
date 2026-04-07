"""VSA2VSA Communication Protocol — Data structures and serialization for
inter-agent communication via raw FHRR phase arrays.

No text. No parsing. Only phase arrays (List[float]) and minimal metadata.
The JSON header is metadata only; the float payload uses struct.pack for
exact precision preservation.

Layer phases (AST, CFG, Data) use uint16 quantization to reduce bandwidth:
  phase in [-pi, pi] -> uint16 in [0, 65535]
  Precision: ~0.0001 radians (~0.006 degrees), sufficient for correlation.
  Bandwidth: 2 bytes/dim vs 4 bytes/dim = 50% reduction per layer.
  Final phases remain float32 (full precision for decoding).
"""

import json
import math
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_TWO_PI = 2.0 * math.pi


@dataclass
class PhaseMessage:
    """A single message in the VSA2VSA protocol."""
    sender_id: int                    # Agent index
    message_type: str                 # "candidate" | "wall" | "target" | "ack"
    phases: List[float]               # The phase array (d floats) — final_handle phases
    metadata: Dict[str, Any]          # fitness, entropy, source_context, etc.
    timestamp: int                    # Monotonic counter (not wall clock)
    # Per-layer phase arrays for layered consensus
    ast_phases: Optional[List[float]] = None
    cfg_phases: Optional[List[float]] = None
    data_phases: Optional[List[float]] = None


@dataclass
class SwarmConfig:
    """Configuration for a swarm run."""
    n_agents: int                     # Number of agents
    dimension: int                    # FHRR dimension (all agents share this)
    arena_capacity: int               # Per-agent arena capacity
    consensus_threshold: float        # rho threshold for clique consensus (default: 0.85)
    fitness_threshold: float          # Minimum fitness for vocab expansion (default: 0.4)
    max_rounds: int                   # Maximum orchestration rounds
    stagnation_limit: int             # Consecutive zero-progress rounds before halt
    corpus_paths: List[List[str]]     # Per-agent corpus directory paths
    # corpus_paths[i] = list of directories for agent i
    # MUST have len(corpus_paths) == n_agents
    # Each agent's corpus MUST be different
    drift_reconciliation_interval: int = 5  # Rounds between drift reconciliation
    drift_grace_period: int = 3             # Rounds before a wall is eligible for propagation


# ── Phase Quantization ───────────────────────────────────────────

def _quantize_phases(phases: List[float]) -> List[int]:
    """Quantize phases from [-pi, pi] to uint16 [0, 65535].

    Precision: ~0.0001 radians per quantum.
    For d=256: 512 bytes instead of 1024 bytes (50% reduction).
    """
    return [int(((p + math.pi) / _TWO_PI) * 65535 + 0.5) & 0xFFFF for p in phases]


def _dequantize_phases(quants: List[int]) -> List[float]:
    """Dequantize uint16 [0, 65535] back to phases in [-pi, pi]."""
    return [(q / 65535.0) * _TWO_PI - math.pi for q in quants]


# ── Serialization ────────────────────────────────────────────────

# Wire format:
#   [4B header_len][header JSON]
#   [4B ast_len][ast uint16s]     <- quantized (2 bytes per phase)
#   [4B cfg_len][cfg uint16s]     <- quantized
#   [4B data_len][data uint16s]   <- quantized
#   [remaining: final float32s]   <- full precision

def serialize_message(msg: PhaseMessage) -> bytes:
    """Serialize a PhaseMessage to bytes for IPC.

    Format: JSON header (message_type, sender_id, metadata, timestamp)
    followed by quantized per-layer uint16 arrays (with 4-byte length
    prefixes) and full-precision final phases as float32.

    Layer phases use uint16 quantization (50% bandwidth reduction).
    Final phases remain float32 for decoding precision.
    """
    header = json.dumps({
        "sender_id": msg.sender_id,
        "message_type": msg.message_type,
        "metadata": msg.metadata,
        "timestamp": msg.timestamp,
    }, separators=(",", ":")).encode("utf-8")

    header_len = len(header)

    # Pack per-layer arrays as quantized uint16 with length prefixes
    def pack_quantized_phases(phases: Optional[List[float]]) -> bytes:
        if phases is None or len(phases) == 0:
            return struct.pack(">I", 0)
        quants = _quantize_phases(phases)
        packed = struct.pack(f">{len(quants)}H", *quants)
        return struct.pack(">I", len(packed)) + packed

    ast_packed = pack_quantized_phases(msg.ast_phases)
    cfg_packed = pack_quantized_phases(msg.cfg_phases)
    data_packed = pack_quantized_phases(msg.data_phases)

    # Final phases at full float32 precision (no length prefix — remaining bytes)
    phases_packed = struct.pack(f">{len(msg.phases)}f", *msg.phases)

    return (
        struct.pack(">I", header_len) + header
        + ast_packed + cfg_packed + data_packed
        + phases_packed
    )


def deserialize_message(data: bytes) -> PhaseMessage:
    """Deserialize bytes back to PhaseMessage."""
    if len(data) < 4:
        raise ValueError("Message too short: missing header length")

    offset = 0

    # Read header
    header_len = struct.unpack(">I", data[offset:offset + 4])[0]
    offset += 4
    if len(data) < offset + header_len:
        raise ValueError("Message truncated: header incomplete")
    header_bytes = data[offset:offset + header_len]
    header = json.loads(header_bytes.decode("utf-8"))
    offset += header_len

    # Read per-layer quantized uint16 arrays
    def read_quantized_phases(off: int):
        if off + 4 > len(data):
            return None, off
        layer_len = struct.unpack(">I", data[off:off + 4])[0]
        off += 4
        if layer_len == 0:
            return None, off
        n_shorts = layer_len // 2
        quants = list(struct.unpack(f">{n_shorts}H", data[off:off + layer_len]))
        off += layer_len
        return _dequantize_phases(quants), off

    ast_phases, offset = read_quantized_phases(offset)
    cfg_phases, offset = read_quantized_phases(offset)
    data_phases, offset = read_quantized_phases(offset)

    # Read final phases (remaining bytes, float32)
    phases_bytes = data[offset:]
    n_floats = len(phases_bytes) // 4
    if n_floats > 0:
        phases = list(struct.unpack(f">{n_floats}f", phases_bytes))
    else:
        phases = []

    return PhaseMessage(
        sender_id=header["sender_id"],
        message_type=header["message_type"],
        phases=phases,
        metadata=header.get("metadata", {}),
        timestamp=header["timestamp"],
        ast_phases=ast_phases,
        cfg_phases=cfg_phases,
        data_phases=data_phases,
    )
