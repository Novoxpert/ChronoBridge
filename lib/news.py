import os, numpy as np, pandas as pd, torch,logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def add_onehot_columns(df_news: pd.DataFrame, tradable_symbols: list) -> pd.DataFrame:
    """
    Convert `assets` column (list of dicts) to one-hot columns for all tradable symbols.
    Only symbols in tradable_symbols are used; missing symbols will have 0.
    """
    #syms = [s.replace('USDT','') for s in tradable_symbols]
    #syms = [s.replace('BINANCE:','') for s in tradable_symbols]
    syms = tradable_symbols

    def parse_symbols(asset_list):
        if not isinstance(asset_list, list):
            return []
        return [a['symbol'] for a in asset_list if isinstance(a, dict) and 'symbol' in a and a['symbol'] in syms]

    # safely handle missing / non-list entries
    df_news["asset_symbols"] = df_news["assets"].apply(lambda x: x if isinstance(x, list) else []).apply(parse_symbols)

    # create one-hot
    onehot = df_news["asset_symbols"].explode().str.get_dummies().groupby(level=0).sum()

    # ensure all tradable symbols columns exist
    for s in syms:
        if s not in onehot.columns:
            onehot[s] = 0
    onehot = onehot[syms]

    df_news = df_news.drop(columns=[c for c in onehot.columns if c in df_news.columns], errors='ignore')
    df_news = df_news.join(onehot)

    return df_news.reset_index(drop=True)



def load_text_encoder(model_path: str, device: str = None, max_len: int = 2048):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_path)
    #Edited by "Elham Esmaeilnia" (2025 sep 7): 
    #mdl = AutoModel.from_pretrained(model_path).to(device).eval()
    mdl = AutoModel.from_pretrained(model_path).to(device).eval()
    return tok, mdl, device, max_len

#One-time global perf knobs (safe to set once per process)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@torch.no_grad()
def embed_texts(
    texts,
    tok,
    mdl,
    device="cuda",
    max_len=256,
    pooling="mean",        # "mean" or "cls"
    batch_size=512,        # tune up/down; will auto-backoff on OOM
    show_progress=True
):
    """
    Efficient text embedding with GPU, AMP, and OOM-safe batching.

    - Cleans None/NaN -> "" (or a special token if you prefer)
    - Uses HF fast tokenizer, padding+truncation to max_len
    - Mixed precision on CUDA for 1.5–3x speedup
    - Non-blocking tensor transfers
    - Avoids per-batch .cpu().numpy() until the end
    - Auto backoff on CUDA OOM by halving batch_size
    """

    # --- Clean input texts cheaply (vectorized-ish) ---
    def _to_str(x):
        if x is None:
            return ""
        if isinstance(x, float) and np.isnan(x):
            return ""
        s = x if isinstance(x, str) else str(x)
        s = s.strip()
        return s  # allow "" after strip

    texts = [ _to_str(t) for t in texts ]
    n = len(texts)
    if n == 0:
        return np.zeros((0, mdl.config.hidden_size), dtype=np.float32)

    mdl.eval()
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    # Use AMP on CUDA
    use_amp = (dev.type == "cuda")
    scaler_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else torch.autocast(enabled=False, device_type="cpu")

    # Progress bar
    rng = range(0, n, batch_size)
    if show_progress:
        rng = tqdm(rng, desc="Embedding news", total=(n + batch_size - 1) // batch_size)

    # Accumulate as tensors -> single CPU copy at the end
    emb_chunks = []
    i = 0
    cur_bs = batch_size

    while i < n:
        j = min(i + cur_bs, n)
        batch = texts[i:j]

        try:
            # Tokenize on CPU; create PT tensors
            enc = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len
            )

            # Move to GPU with non_blocking copies (uses pinned memory path)
            for k in enc:
                enc[k] = enc[k].to(dev, non_blocking=True)

            with scaler_ctx:
                out = mdl(**enc).last_hidden_state  # [B, T, H]

                if pooling == "mean":
                    mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
                    denom = mask.sum(1).clamp(min=1)
                    pooled = (out * mask).sum(1) / denom        # [B, H]
                else:  # "cls"
                    pooled = out[:, 0, :]                        # [B, H]

            # Keep on GPU; move to CPU only once at the end
            emb_chunks.append(pooled.detach())

            # Advance window & progress
            i = j
            if show_progress:
                # Manually update by the number of items processed in this (possibly smaller) batch
                tqdm.write(f"Processed {i}/{n}") if (i % (cur_bs * 4) == 0) else None

            # Optional: do NOT empty the cache every loop; it hurts perf.
            # torch.cuda.empty_cache()  # avoid unless you are truly memory constrained

            # If we had previously reduced batch size due to OOM and now succeed a few times,
            # you could consider slowly increasing it again. Kept simple here.

        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and dev.type == "cuda":
                # Back off batch size and retry the same range
                new_bs = max(1, cur_bs // 2)
                if new_bs == cur_bs:
                    raise  # cannot reduce further
                logging.warning(f"OOM at batch_size={cur_bs}. Reducing to {new_bs} and retrying...")
                cur_bs = new_bs
                torch.cuda.empty_cache()
                continue
            else:
                raise

    # Concatenate on GPU, single D2H copy, then to numpy
    emb = torch.cat(emb_chunks, dim=0)          # [N, H] on GPU (or CPU)
    emb = emb.to("cpu", non_blocking=True)      # one big transfer
    return emb.numpy()


def resample_news_3m(df_news: pd.DataFrame, no_news_vec: np.ndarray, rule: str = "3min") -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    # Ensure no_news_vec is float32
    no_news_vec = np.asarray(no_news_vec, dtype=np.float32)

    df = df_news.copy()
    df["releasedAt"] = pd.to_datetime(df["releasedAt"], errors="coerce")
    df = df.dropna(subset=["releasedAt"])
    df["t3"] = df["releasedAt"].dt.floor(rule)

    base = {"releasedAt", "content", "embedding", "news_count", "asset_symbols", "t3"}
    asset_cols = [c for c in df.columns if c not in base and df[c].dtype != "O"]

    if "news_count" not in df.columns:
        df["news_count"] = 1

    g = df.groupby("t3", sort=True)
    agg = g[asset_cols + ["news_count"]].sum()

    # Function to average embeddings safely
    def _avg(arr):
        arr = [np.asarray(x, dtype=np.float32) for x in arr if isinstance(x, (list, np.ndarray))]
        if not arr:
            return np.zeros_like(no_news_vec, dtype=np.float32)
        shapes = {a.shape for a in arr}
        if len(shapes) != 1:
            raise ValueError(f"Embedding length mismatch {shapes}")
        return np.mean(np.stack(arr, axis=0), axis=0).astype(np.float32)

    emb = g["embedding"].apply(_avg)
    news_3m = agg.join(emb.rename("embedding"))

    # Reindex to regular 3-minute intervals
    idx = pd.date_range(news_3m.index.min(), news_3m.index.max(), freq=rule)
    news_3m = news_3m.reindex(idx)

    # Fill missing values in numeric columns
    news_3m["news_count"] = news_3m["news_count"].fillna(0).astype(int)
    for c in asset_cols:
        news_3m[c] = news_3m[c].fillna(0).astype(int)

    # Fill missing embeddings with no_news_vec (ensuring float32)
    mask = news_3m["embedding"].isna()
    news_3m.loc[mask, "embedding"] = news_3m.loc[mask, "embedding"].apply(
        lambda _: np.copy(no_news_vec)
    )

    # Ensure all embeddings are float32 (in case some were different)
    news_3m["embedding"] = news_3m["embedding"].apply(lambda x: np.asarray(x, dtype=np.float32))

    news_3m.index.name = "t3"
    return news_3m
