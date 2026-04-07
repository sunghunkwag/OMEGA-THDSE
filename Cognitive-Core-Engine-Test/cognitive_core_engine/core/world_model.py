"""
WorldModel — environment dynamics and Q-value estimation.

BN-01 fix: replace linear feature-hashing Q-learning with a tiny
transformer backbone.  The transformer generalises across unseen
(obs, action) pairs and provides richer uncertainty estimates.

Public API is backward-compatible with the original WorldModel:
  encode_state(obs) -> str
  features(obs, action) -> Dict[str, float]       # kept for compat
  q_value(obs, action) -> float
  confidence(obs, action) -> float
  update(obs, action, reward, next_obs, action_space) -> None

The original implementation is re-exported as LegacyWorldModel so
existing tests that import it directly continue to work.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cognitive_core_engine.core.utils import stable_hash


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Transformer architecture
_VOCAB_SIZE = 512          # hash-bucket count for token embedding
_EMBED_DIM = 32            # embedding / model dimension
_N_HEADS = 2               # multi-head attention heads
_FF_DIM = 64               # feedforward hidden dimension
_N_LAYERS = 2              # number of transformer encoder layers
_MAX_SEQ_LEN = 8           # max tokens per (obs, action) encoding

# Training
_LR = 3e-3                 # Adam learning rate
_GAMMA = 0.9               # TD discount factor
_REPLAY_CAPACITY = 500     # experience replay buffer capacity
_BATCH_SIZE = 16           # replay mini-batch size
_REPLAY_ALPHA = 0.6        # prioritisation exponent (0 = uniform)
_MC_DROPOUT_PASSES = 5     # forward passes for uncertainty estimate
_DROPOUT_RATE = 0.1        # dropout probability during MC sampling


# ---------------------------------------------------------------------------
# Minimal NumPy transformer primitives
# ---------------------------------------------------------------------------

class _Embedding:
    """Token embedding table: vocab_size × embed_dim."""

    def __init__(self, vocab_size: int, embed_dim: int, rng: np.random.Generator) -> None:
        scale = math.sqrt(2.0 / embed_dim)
        self.W = rng.normal(0.0, scale, (vocab_size, embed_dim)).astype(np.float32)
        # Adam state
        self.m = np.zeros_like(self.W)
        self.v = np.zeros_like(self.W)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """token_ids: (seq_len,) int → (seq_len, embed_dim)."""
        return self.W[token_ids]

    def backward_and_update(self, token_ids: np.ndarray, grad: np.ndarray,
                             t: int, lr: float) -> None:
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for i, idx in enumerate(token_ids):
            g = grad[i]
            self.m[idx] = beta1 * self.m[idx] + (1 - beta1) * g
            self.v[idx] = beta2 * self.v[idx] + (1 - beta2) * (g * g)
            m_hat = self.m[idx] / (1 - beta1 ** t)
            v_hat = self.v[idx] / (1 - beta2 ** t)
            self.W[idx] -= lr * m_hat / (np.sqrt(v_hat) + eps)


class _LayerNorm:
    """Per-feature layer normalisation."""

    def __init__(self, dim: int) -> None:
        self.gamma = np.ones(dim, dtype=np.float32)
        self.beta = np.zeros(dim, dtype=np.float32)
        self._last_x: Optional[np.ndarray] = None
        self._last_mean: Optional[np.ndarray] = None
        self._last_std: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + 1e-6
        self._last_x, self._last_mean, self._last_std = x, mean, std
        return self.gamma * (x - mean) / std + self.beta


class _MultiHeadSelfAttention:
    """Scaled dot-product multi-head self-attention (seq_len, d_model)."""

    def __init__(self, d_model: int, n_heads: int, rng: np.random.Generator) -> None:
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        scale = math.sqrt(2.0 / d_model)
        # Query, Key, Value, Output projections
        self.Wq = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.Wk = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.Wv = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.Wo = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        # Adam state (simplified: one slot per weight matrix)
        self._params = [self.Wq, self.Wk, self.Wv, self.Wo]
        self._m = [np.zeros_like(p) for p in self._params]
        self._v = [np.zeros_like(p) for p in self._params]

    def forward(self, x: np.ndarray, dropout_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """x: (seq, d_model) → (seq, d_model)."""
        seq = x.shape[0]
        Q = x @ self.Wq  # (seq, d_model)
        K = x @ self.Wk
        V = x @ self.Wv

        # Reshape to (heads, seq, d_k)
        Q = Q.reshape(seq, self.n_heads, self.d_k).transpose(1, 0, 2)
        K = K.reshape(seq, self.n_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(seq, self.n_heads, self.d_k).transpose(1, 0, 2)

        scores = Q @ K.transpose(0, 2, 1) / math.sqrt(self.d_k)  # (heads, seq, seq)
        attn = _softmax(scores, axis=-1)
        if dropout_mask is not None:
            attn = attn * dropout_mask
        ctx = attn @ V  # (heads, seq, d_k)
        ctx = ctx.transpose(1, 0, 2).reshape(seq, self.d_model)
        return ctx @ self.Wo

    def adam_update(self, param_idx: int, grad: np.ndarray, t: int, lr: float) -> None:
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self._m[param_idx] = beta1 * self._m[param_idx] + (1 - beta1) * grad
        self._v[param_idx] = beta2 * self._v[param_idx] + (1 - beta2) * (grad * grad)
        m_hat = self._m[param_idx] / (1 - beta1 ** t)
        v_hat = self._v[param_idx] / (1 - beta2 ** t)
        self._params[param_idx] -= lr * m_hat / (np.sqrt(v_hat) + eps)


class _FeedForward:
    """Position-wise feed-forward: d_model → ff_dim → d_model."""

    def __init__(self, d_model: int, ff_dim: int, rng: np.random.Generator) -> None:
        scale1 = math.sqrt(2.0 / d_model)
        scale2 = math.sqrt(2.0 / ff_dim)
        self.W1 = rng.normal(0, scale1, (d_model, ff_dim)).astype(np.float32)
        self.b1 = np.zeros(ff_dim, dtype=np.float32)
        self.W2 = rng.normal(0, scale2, (ff_dim, d_model)).astype(np.float32)
        self.b2 = np.zeros(d_model, dtype=np.float32)
        self._params = [self.W1, self.b1, self.W2, self.b2]
        self._m = [np.zeros_like(p) for p in self._params]
        self._v = [np.zeros_like(p) for p in self._params]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (seq, d_model) → (seq, d_model)."""
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return h @ self.W2 + self.b2

    def adam_update(self, param_idx: int, grad: np.ndarray, t: int, lr: float) -> None:
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self._m[param_idx] = beta1 * self._m[param_idx] + (1 - beta1) * grad
        self._v[param_idx] = beta2 * self._v[param_idx] + (1 - beta2) * (grad * grad)
        m_hat = self._m[param_idx] / (1 - beta1 ** t)
        v_hat = self._v[param_idx] / (1 - beta2 ** t)
        self._params[param_idx] -= lr * m_hat / (np.sqrt(v_hat) + eps)


class _OutputHead:
    """Linear projection from pooled embedding to scalar Q-value."""

    def __init__(self, d_model: int, rng: np.random.Generator) -> None:
        self.W = rng.normal(0, math.sqrt(2.0 / d_model), (d_model, 1)).astype(np.float32)
        self.b = np.zeros(1, dtype=np.float32)
        self._m_W = np.zeros_like(self.W)
        self._v_W = np.zeros_like(self.W)
        self._m_b = np.zeros_like(self.b)
        self._v_b = np.zeros_like(self.b)

    def forward(self, pooled: np.ndarray) -> float:
        """pooled: (d_model,) → scalar."""
        return float((pooled @ self.W + self.b).squeeze())

    def backward(self, pooled: np.ndarray, grad_out: float) -> np.ndarray:
        """Returns grad w.r.t. pooled."""
        return (grad_out * self.W.T).squeeze()

    def adam_update(self, pooled: np.ndarray, grad_out: float, t: int, lr: float) -> None:
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        gW = pooled.reshape(-1, 1) * grad_out
        gb = np.array([grad_out])
        for m, v, p, g in [
            (self._m_W, self._v_W, self.W, gW),
            (self._m_b, self._v_b, self.b, gb),
        ]:
            m[:] = beta1 * m + (1 - beta1) * g
            v[:] = beta2 * v + (1 - beta2) * (g * g)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def _tokenise(obs: Dict[str, Any], action: str, max_len: int = _MAX_SEQ_LEN) -> np.ndarray:
    """Convert (obs, action) to a fixed-length integer token sequence.

    Each field is hashed to a bucket in [0, _VOCAB_SIZE).  The sequence is
    padded with token 0 or truncated to max_len.
    """
    fields = [
        str(obs.get("task", "")),
        str(obs.get("domain", "")),
        str(int(obs.get("difficulty", 0))),
        str(int(obs.get("budget", 0)) // 10),  # coarse budget bucket
        str(obs.get("phase", "")),
        action,
    ]
    tokens = [abs(hash(f)) % _VOCAB_SIZE for f in fields]
    # Pad or truncate
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    return np.array(tokens[:max_len], dtype=np.int32)


# ---------------------------------------------------------------------------
# Replay buffer with priority sampling
# ---------------------------------------------------------------------------

@dataclass
class _Experience:
    obs: Dict[str, Any]
    action: str
    reward: float
    next_obs: Dict[str, Any]
    action_space: List[str]
    priority: float = 1.0


class _PrioritisedReplayBuffer:
    """Fixed-capacity experience replay with TD-error priority sampling."""

    def __init__(self, capacity: int) -> None:
        self._buf: deque[_Experience] = deque(maxlen=capacity)

    def push(self, exp: _Experience) -> None:
        self._buf.append(exp)

    def sample(self, n: int, rng: random.Random) -> List[_Experience]:
        if len(self._buf) < n:
            return list(self._buf)
        total = sum(e.priority ** _REPLAY_ALPHA for e in self._buf)
        probs = [(e.priority ** _REPLAY_ALPHA) / total for e in self._buf]
        buf_list = list(self._buf)
        indices = rng.choices(range(len(buf_list)), weights=probs, k=n)
        return [buf_list[i] for i in indices]

    def update_priority(self, idx: int, td_error: float) -> None:
        if 0 <= idx < len(self._buf):
            list(self._buf)[idx].priority = abs(td_error) + 1e-6

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# Main TransformerWorldModel
# ---------------------------------------------------------------------------

class TransformerWorldModel:
    """Two-layer transformer Q-value estimator.

    Replaces the linear feature-hashing model (BN-01).  Same public API.
    """

    def __init__(
        self,
        gamma: float = _GAMMA,
        lr: float = _LR,
        seed: int = 42,
    ) -> None:
        self.gamma = gamma
        self.lr = lr
        self._rng_np = np.random.default_rng(seed)
        self._rng_py = random.Random(seed)
        self._t = 0  # Adam step counter

        # Architecture
        self._embed = _Embedding(_VOCAB_SIZE, _EMBED_DIM, self._rng_np)
        self._layers: List[Tuple[_MultiHeadSelfAttention, _FeedForward, _LayerNorm, _LayerNorm]] = []
        for _ in range(_N_LAYERS):
            attn = _MultiHeadSelfAttention(_EMBED_DIM, _N_HEADS, self._rng_np)
            ff = _FeedForward(_EMBED_DIM, _FF_DIM, self._rng_np)
            ln1 = _LayerNorm(_EMBED_DIM)
            ln2 = _LayerNorm(_EMBED_DIM)
            self._layers.append((attn, ff, ln1, ln2))
        self._head = _OutputHead(_EMBED_DIM, self._rng_np)

        # Replay buffer
        self._replay = _PrioritisedReplayBuffer(_REPLAY_CAPACITY)

        # Visitation counts (for confidence)
        self._sa_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Internal forward pass
    # ------------------------------------------------------------------

    def _forward(self, tokens: np.ndarray,
                 dropout_rate: float = 0.0) -> Tuple[float, np.ndarray]:
        """tokens: (seq_len,) → (q_value, pooled_embedding)."""
        x = self._embed.forward(tokens).astype(np.float32)  # (seq, embed)

        for attn, ff, ln1, ln2 in self._layers:
            # MC-dropout mask on attention weights
            mask = None
            if dropout_rate > 0:
                shape = (_N_HEADS, len(tokens), len(tokens))
                mask = (self._rng_np.random(shape) > dropout_rate).astype(np.float32)
                mask /= (1.0 - dropout_rate + 1e-9)

            residual = x
            x = ln1.forward(x)
            x = residual + attn.forward(x, dropout_mask=mask)

            residual = x
            x = ln2.forward(x)
            x = residual + ff.forward(x)

        pooled = x.mean(axis=0)  # mean-pool over sequence
        q = self._head.forward(pooled)
        return q, pooled

    # ------------------------------------------------------------------
    # Public API (backward-compatible with original WorldModel)
    # ------------------------------------------------------------------

    def encode_state(self, obs: Dict[str, Any]) -> str:
        key = {
            "task": obs.get("task", ""),
            "domain": obs.get("domain", ""),
            "difficulty": int(obs.get("difficulty", 0)),
            "budget": int(obs.get("budget", 0)),
            "phase": obs.get("phase", ""),
        }
        return stable_hash(key)

    def features(self, obs: Dict[str, Any], action: str) -> Dict[str, float]:
        """Return token embedding as a flat feature dict.

        Retained for backward compatibility.  Callers that iterate over
        features() should still work; the values are now continuous
        embedding dimensions rather than binary indicator features.
        """
        tokens = _tokenise(obs, action)
        x = self._embed.forward(tokens)  # (seq, embed)
        pooled = x.mean(axis=0)           # (embed,)
        return {f"emb_{i}": float(pooled[i]) for i in range(len(pooled))}

    def q_value(self, obs: Dict[str, Any], action: str) -> float:
        tokens = _tokenise(obs, action)
        q, _ = self._forward(tokens)
        return q

    def confidence(self, obs: Dict[str, Any], action: str) -> float:
        """MC-dropout uncertainty estimate: 1 − std over multiple forward passes.

        A higher value means the model is more certain about this Q-value.
        """
        tokens = _tokenise(obs, action)
        samples = [self._forward(tokens, dropout_rate=_DROPOUT_RATE)[0]
                   for _ in range(_MC_DROPOUT_PASSES)]
        std = float(np.std(samples))
        # Normalise: std ~ 0 → confidence ~ 1; std ~ 1 → confidence ~ 0
        return max(0.0, 1.0 - min(1.0, std * 2.0))

    def update(
        self,
        obs: Dict[str, Any],
        action: str,
        reward: float,
        next_obs: Dict[str, Any],
        action_space: List[str],
    ) -> None:
        """TD(0) update with experience replay."""
        # Compute TD error for current transition
        td_error = self._td_error(obs, action, reward, next_obs, action_space)

        # Store in replay buffer with priority = |td_error|
        self._replay.push(_Experience(
            obs=obs, action=action, reward=reward,
            next_obs=next_obs, action_space=action_space,
            priority=abs(td_error) + 1e-6,
        ))

        # Gradient step on current transition
        self._gradient_step(obs, action, reward, next_obs, action_space)

        # Update visitation count
        key = self.encode_state(obs) + "|" + action
        self._sa_counts[key] = self._sa_counts.get(key, 0) + 1

        # Experience replay mini-batch
        if len(self._replay) >= _BATCH_SIZE:
            batch = self._replay.sample(_BATCH_SIZE, self._rng_py)
            for exp in batch:
                self._gradient_step(
                    exp.obs, exp.action, exp.reward,
                    exp.next_obs, exp.action_space,
                )

    def _td_error(
        self,
        obs: Dict[str, Any],
        action: str,
        reward: float,
        next_obs: Dict[str, Any],
        action_space: List[str],
    ) -> float:
        current_q = self.q_value(obs, action)
        next_best = max(self.q_value(next_obs, a) for a in action_space)
        target = reward + self.gamma * next_best
        return target - current_q

    def _gradient_step(
        self,
        obs: Dict[str, Any],
        action: str,
        reward: float,
        next_obs: Dict[str, Any],
        action_space: List[str],
    ) -> None:
        """One Adam gradient step minimising the squared TD error."""
        self._t += 1
        tokens = _tokenise(obs, action)
        q, pooled = self._forward(tokens)

        next_best = max(self.q_value(next_obs, a) for a in action_space)
        target = reward + self.gamma * next_best
        td_error = target - q

        # Gradient of 0.5 * (target - q)^2 w.r.t. q  is  -(td_error)
        grad_q = -td_error  # scalar

        # Backprop through output head → pooled embedding
        grad_pooled = self._head.backward(pooled, grad_q)
        self._head.adam_update(pooled, grad_q, self._t, self.lr)

        # Backprop through mean-pool → each token embedding
        seq_len = len(tokens)
        grad_seq = np.stack([grad_pooled / seq_len] * seq_len)  # (seq, embed)

        # Update embedding table rows
        self._embed.backward_and_update(tokens, grad_seq, self._t, self.lr)


# ---------------------------------------------------------------------------
# Backward compatibility: keep original implementation as LegacyWorldModel
# ---------------------------------------------------------------------------

@dataclass
class TransitionSummary:
    count: int = 0


class LegacyWorldModel:
    """Original linear feature-hashing Q-learning world model.

    Retained for backward compatibility and as a baseline for ablation.
    New code should use TransformerWorldModel (aliased as WorldModel below).
    """

    def __init__(self, gamma: float = 0.9, lr: float = 0.08) -> None:
        self.gamma = gamma
        self.lr = lr
        self._weights: Dict[str, float] = {}
        self._sa_counts: Dict[Tuple[str, str], TransitionSummary] = {}
        self.replay_buffer: List[Tuple] = []
        self.max_buffer_size = 200

    def _feature_bucket(self, budget: int) -> int:
        return min(5, max(0, budget // 10))

    def encode_state(self, obs: Dict[str, Any]) -> str:
        key = {
            "task": obs.get("task", ""),
            "domain": obs.get("domain", ""),
            "difficulty": int(obs.get("difficulty", 0)),
            "budget": int(obs.get("budget", 0)),
            "phase": obs.get("phase", ""),
        }
        return stable_hash(key)

    def features(self, obs: Dict[str, Any], action: str) -> Dict[str, float]:
        task = str(obs.get("task", ""))
        domain = str(obs.get("domain", ""))
        diff = int(obs.get("difficulty", 0))
        phase = str(obs.get("phase", ""))
        budget = int(obs.get("budget", 0))
        bucket = self._feature_bucket(budget)
        return {
            "bias": 1.0,
            f"task:{task}": 1.0,
            f"domain:{domain}": 1.0,
            f"diff:{diff}": 1.0,
            f"phase:{phase}": 1.0,
            f"action:{action}": 1.0,
            f"task_action:{task}|{action}": 1.0,
            f"budget_bucket:{bucket}": 1.0,
            f"diff_action:{diff}|{action}": 1.0,
            f"domain_diff:{domain}|{diff}": float(diff) / 5.0,
            f"task_phase:{task}|{phase}": 1.0,
        }

    def q_value(self, obs: Dict[str, Any], action: str) -> float:
        feats = self.features(obs, action)
        return sum(self._weights.get(k, 0.0) * v for k, v in feats.items())

    def confidence(self, obs: Dict[str, Any], action: str) -> float:
        s = self.encode_state(obs)
        count = self._sa_counts.get((s, action), TransitionSummary()).count
        return 1.0 - (1.0 / math.sqrt(count + 1.0))

    def update(self, obs: Dict[str, Any], action: str, reward: float,
               next_obs: Dict[str, Any], action_space: List[str]) -> None:
        self.replay_buffer.append((obs, action, reward, next_obs, action_space))
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer.pop(0)
        feats = self.features(obs, action)
        current = self.q_value(obs, action)
        next_best = max(self.q_value(next_obs, a) for a in action_space)
        target = reward + self.gamma * next_best
        td_error = target - current
        for k, v in feats.items():
            self._weights[k] = self._weights.get(k, 0.0) + self.lr * td_error * v
        if len(self.replay_buffer) >= 10:
            samples = random.sample(self.replay_buffer, min(5, len(self.replay_buffer)))
            for s_obs, s_action, s_reward, s_next_obs, s_action_space in samples:
                s_feats = self.features(s_obs, s_action)
                s_current = self.q_value(s_obs, s_action)
                s_next_best = max(self.q_value(s_next_obs, a) for a in s_action_space)
                s_target = s_reward + self.gamma * s_next_best
                s_td_error = s_target - s_current
                for k, v in s_feats.items():
                    self._weights[k] = self._weights.get(k, 0.0) + (self.lr * 0.5) * s_td_error * v
        s = self.encode_state(obs)
        entry = self._sa_counts.get((s, action))
        if entry is None:
            entry = TransitionSummary()
            self._sa_counts[(s, action)] = entry
        entry.count += 1


# ---------------------------------------------------------------------------
# Default export: TransformerWorldModel as WorldModel
# ---------------------------------------------------------------------------

#: WorldModel now refers to the transformer-based implementation.
#: Import LegacyWorldModel explicitly when the linear baseline is needed.
WorldModel = TransformerWorldModel
