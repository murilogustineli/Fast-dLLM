import os
import types
import inspect
from typing import Dict, List, Optional


def dbg(*a, **k):
    # Toggle with: export DEBUG_LAYER_REUSE=1
    if os.environ.get("DEBUG_LAYER_REUSE", "0") == "1":
        print("[LAYER-REUSE-DEBUG]", *a, **k)


def _shape_of(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return tuple(x.shape)
    except Exception:
        pass
    return None


def _brief_arg(a):
    s = type(a).__name__
    shp = _shape_of(a)
    if shp is not None:
        s += f"[shape={shp}]"
    return s


def _get_seq_len_from_args(args, kwargs):
    # Expect first positional arg to be hidden_states: [B, T, D]
    if args and _shape_of(args[0]) is not None and len(args[0].shape) >= 2:
        try:
            return int(args[0].shape[1])
        except Exception:
            pass
    # Fallbacks if you ever pass hidden_states by kw (rare in HF)
    hs = kwargs.get("hidden_states", None)
    if hs is not None and _shape_of(hs) is not None and len(hs.shape) >= 2:
        try:
            return int(hs.shape[1])
        except Exception:
            pass
    return None


class LayerReuseController:
    """
    Wraps a subset of decoder layers to reuse the previous forward output
    for (reuse_k - 1) calls, and refresh on every kth call.
    """

    def __init__(self, model_with_layers, subset: Optional[str], reuse_k: int):
        # model_with_layers must expose .model.layers or .layers; here we use .model.layers if available else .layers
        if hasattr(model_with_layers, "model") and hasattr(
            model_with_layers.model, "layers"
        ):
            self.layers = model_with_layers.model.layers
            dbg("Using model.model.layers")
        else:
            self.layers = model_with_layers.layers
            dbg("Using model.layers")

        self.reuse_k = max(int(reuse_k), 1)
        self.enabled = False
        self.counter = 0
        self.cache: Dict[int, object] = {}
        self.orig_forwards: Dict[int, object] = {}

        n = len(self.layers)
        subset_size = min(12, n)
        if subset is None:
            self.reuse_layers: List[int] = []
        elif subset == "first":
            self.reuse_layers = list(range(0, subset_size))
        elif subset == "middle":
            start = max(0, n // 2 - subset_size // 2)
            self.reuse_layers = list(range(start, start + subset_size))
        elif subset == "last":
            self.reuse_layers = list(range(n - subset_size, n))
        else:
            # treat iterable/list of indices as-is
            self.reuse_layers = list(subset)

        dbg(
            f"Init: n_layers={n} | subset='{subset}' => reuse_layers={self.reuse_layers} | reuse_k={self.reuse_k}"
        )

    def _make_wrapper(self, layer_idx: int, orig_forward):
        is_bound = (
            hasattr(orig_forward, "__self__") and orig_forward.__self__ is not None
        )

        def wrapped(self_layer, *args, **kwargs):
            dbg(
                f"CALL layer={layer_idx} | enabled={self.enabled} | step={self.counter} | "
                f"reuse_k={self.reuse_k} | args_len={len(args)} | kwargs_keys={list(kwargs.keys())}"
            )

            # ---- Inspect basics
            x = args[0] if args else None
            T_cur = None
            if x is not None:
                shp = _shape_of(x)
                if shp is not None and len(shp) >= 3:
                    # (B, T, C) or similar
                    T_cur = shp[1]
                    dbg(f"Current input seq_len (T) for layer {layer_idx}: {T_cur}")

            # ---- Detect stateful KV/block-cache usage
            use_block_cache = bool(kwargs.get("use_block_cache", False))
            has_past = kwargs.get("past_key_value", None) is not None
            has_block_past = kwargs.get("block_past_key_values", None) is not None
            touches_positions = any(
                k in kwargs
                for k in ("cache_position", "replace_position", "position_embeddings")
            )

            stateful = (
                use_block_cache or has_past or has_block_past or touches_positions
            )
            if stateful:
                dbg(
                    f"Layer {layer_idx}: stateful=True (use_block_cache={use_block_cache}, "
                    f"has_past={has_past}, has_block_past={has_block_past}, touches_positions={touches_positions}) "
                    f"→ bypassing reuse AND caching"
                )

            # ---- Check if reuse would be valid
            can_consider_reuse = (
                self.enabled
                and (layer_idx in self.reuse_layers)
                and (self.reuse_k > 1)
                and (self.counter % self.reuse_k != 0)
                and not stateful  # ← hard stop: never reuse on stateful calls
            )

            # Optional: cache metadata to guard on T changes
            meta = self.cache.get(("meta", layer_idx))
            last_T = meta.get("T") if isinstance(meta, dict) else None
            same_len = (T_cur is None) or (last_T is None) or (T_cur == last_T)

            if can_consider_reuse and same_len and (layer_idx in self.cache):
                dbg(f"Cache HIT for layer {layer_idx} at step={self.counter}")
                return self.cache[layer_idx]
            else:
                dbg(
                    f"{'Stateful call' if stateful else 'Cache MISS/REFRESH'} for layer {layer_idx} "
                    f"at step={self.counter} (enabled={self.enabled}, in_subset={layer_idx in self.reuse_layers}, "
                    f"same_len={same_len})"
                )

            # ---- Call original forward
            try:
                out = (
                    orig_forward(*args, **kwargs)
                    if is_bound
                    else orig_forward(self_layer, *args, **kwargs)
                )
            except TypeError as te:
                sig = None
                try:
                    sig = inspect.signature(orig_forward)
                except Exception:
                    pass
                dbg("---- TypeError CONTEXT DUMP ----")
                dbg(f"layer_idx={layer_idx} | is_bound={is_bound}")
                dbg(f"orig_forward={orig_forward}")
                dbg(f"signature={sig}")
                dbg(f"args_len={len(args)} | kwargs={list(kwargs.keys())}")
                if args:
                    for i, a in enumerate(args[:8]):
                        dbg(f"  arg[{i}] -> {_brief_arg(a)}")
                if "attention_mask" in kwargs:
                    dbg(
                        f"  kwargs['attention_mask'] -> {_brief_arg(kwargs['attention_mask'])}"
                    )
                dbg("---- END CONTEXT DUMP ----")
                raise TypeError(
                    f"[LayerReuse Wrapper] Forward failed on layer={layer_idx} | "
                    f"is_bound={is_bound} | reuse_k={self.reuse_k} | step={self.counter} | error={te}"
                ) from te

            # ---- Only cache on non-stateful paths, and when length is stable
            if (
                self.enabled
                and (layer_idx in self.reuse_layers)
                and (not stateful)
                and same_len
            ):
                self.cache[layer_idx] = out
                self.cache[("meta", layer_idx)] = {"T": T_cur}
                dbg(f"Updated cache for layer {layer_idx} (seq_len={T_cur})")
            else:
                if stateful:
                    dbg(f"Skipped caching for layer {layer_idx} due to stateful call")
                elif not same_len:
                    dbg(
                        f"Skipped caching for layer {layer_idx} due to seq_len change: last_T={last_T} → T_cur={T_cur}"
                    )

            return out

        wrapped.__orig_forward__ = orig_forward
        wrapped.__controller__ = self
        return wrapped

    def enable_reuse(self):
        if self.enabled:
            dbg("enable_reuse(): already enabled")
            return
        self.enabled = True
        self.counter = 0
        self.cache.clear()
        dbg(f"Enabling reuse on layers={self.reuse_layers} | reuse_k={self.reuse_k}")

        for idx, layer in enumerate(self.layers):
            if idx in self.reuse_layers:
                self.orig_forwards[idx] = layer.forward
                layer.forward = types.MethodType(
                    self._make_wrapper(idx, layer.forward), layer
                )
                dbg(f"Patched layer {idx}.forward")

    def disable_reuse(self):
        if not self.enabled:
            dbg("disable_reuse(): already disabled")
            return
        dbg("Disabling reuse and restoring original forwards")
        for idx, layer in enumerate(self.layers):
            if idx in self.reuse_layers and idx in self.orig_forwards:
                layer.forward = self.orig_forwards[idx]
                dbg(f"Restored layer {idx}.forward")
        self.orig_forwards.clear()
        self.cache.clear()
        self.enabled = False
        self.counter = 0

    def step(self):
        """
        Call this after each *sampling micro-step* (i.e., after you advance decoding).
        Every reuse_k steps, we drop the cache to refresh.
        """
        if not self.enabled:
            dbg("step(): controller not enabled; ignoring")
            return
        self.counter += 1
        dbg(f"step(): counter -> {self.counter} (reuse_k={self.reuse_k})")
        if self.counter % self.reuse_k == 0:
            self.cache.clear()
            dbg("step(): counter % reuse_k == 0 -> cache cleared")
