from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _as_token_id_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        ids: list[int] = []
        for item in value:
            ids.extend(_as_token_id_list(item))
        return ids
    token_id = int(value)
    return [token_id] if token_id >= 0 else []


def resolve_generation_stop_token_ids(tokenizer, llm=None) -> list[int]:
    """EOS / end-of-turn ids `generate()` should stop on.

    Training labels now use the chat-template closer (`tokenizer.eos_token`, Gemma
    `<turn|>`, Harmony `<|return|>`). Model `generation_config.eos_token_id` can
    disagree (Qwen3.5 config.json lists `<|endoftext|>` while the tokenizer EOS is
    `<|im_end|>`). Merge tokenizer EOS, the model config list, and eot when present.
    """
    ids: list[int] = []
    seen: set[int] = set()

    def add(value) -> None:
        for token_id in _as_token_id_list(value):
            if token_id not in seen:
                seen.add(token_id)
                ids.append(token_id)

    add(getattr(tokenizer, "eos_token_id", None))
    generation_config = getattr(llm, "generation_config", None) if llm is not None else None
    add(getattr(generation_config, "eos_token_id", None))
    eot = getattr(tokenizer, "eot_token", None)
    if isinstance(eot, str) and hasattr(tokenizer, "convert_tokens_to_ids"):
        eot_id = tokenizer.convert_tokens_to_ids(eot)
        unk = getattr(tokenizer, "unk_token_id", None)
        if eot_id is not None and eot_id != unk:
            add(eot_id)
    # Qwen / GPT-OSS: pad is <|endoftext|>, a documented extra stop, not a pad-only id.
    pad_token = getattr(tokenizer, "pad_token", None)
    if pad_token in ("<|endoftext|>", "<|end|>"):
        add(getattr(tokenizer, "pad_token_id", None))
    return ids


class PositionalEncoder(nn.Module):
    def __init__(
        self,
        input_dims: int,
        num_freqs: int,
        include_input: bool = True,
        log_sampling: bool = False,
        polar_fourier: bool = False,
        l_min: float | None = None,
        l_max: float | None = None,
        more_axis: bool = False,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.include_input = False #True #include_input
        self.log_sampling = False #log_sampling
        self.more_axis = True #True #more_axis
        self.polar_fourier = False #polar_fourier

        # Use the *effective* (post-override) flags, and gate more_axis on input_dims==2
        # to match what forward() actually does.
        feature_input_dims = (
            input_dims * 2 if (self.more_axis and input_dims == 2) else input_dims
        )
        self.register_buffer("freq_bands", self._create_freq_bands(l_min, l_max))

        self.output_dims = 2 * num_freqs * feature_input_dims
        if self.include_input:
            self.output_dims += feature_input_dims
        if self.polar_fourier and input_dims == 2:
            self.output_dims += input_dims + 2 * num_freqs

    def _create_freq_bands(self, l_min: float | None, l_max: float | None) -> torch.Tensor:
        if self.log_sampling:
            if l_min is not None and l_max is not None:
                return (2.0 ** torch.linspace(l_min, l_max, self.num_freqs) * torch.pi).unsqueeze(1)
            return (2.0 ** torch.linspace(-10.0, self.num_freqs - 1, self.num_freqs) * torch.pi).unsqueeze(1)
        if l_min is not None and l_max is not None:
            return torch.linspace(l_min, l_max, self.num_freqs).unsqueeze(1)
        return torch.linspace(1.0, 6.0, self.num_freqs).unsqueeze(1)
        #return torch.linspace(.01, 3.0, self.num_freqs).unsqueeze(1)

    @staticmethod
    def more_axis_fn(xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, 0]
        y = xy[:, 1]
        x1 = (x + y) / (2**0.5)
        y1 = (y - x) / (2**0.5)
        return torch.stack([x1, y1, x, y], dim=-1)

    def polar_fourier_encoding(self, xy: torch.Tensor) -> torch.Tensor:
        x, y = xy[:, 0], xy[:, 1]
        r = torch.sqrt(x**2 + y**2)
        freqs = r.unsqueeze(1) * self.freq_bands.squeeze(1)
        return torch.cat([xy, torch.sin(freqs), torch.cos(freqs)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_x = x
        if self.more_axis and self.input_dims == 2:
            x = self.more_axis_fn(x)

        x_freq = x.unsqueeze(1) * self.freq_bands
        parts = [torch.sin(x_freq), torch.cos(x_freq)]
        if self.include_input:
            parts.insert(0, x.unsqueeze(1))
        out = torch.cat(parts, dim=1).reshape(x.shape[0], -1)

        if self.polar_fourier and self.input_dims == 2:
            out = torch.cat([out, self.polar_fourier_encoding(original_x)], dim=-1)
        return out


class GeoCLIP(nn.Module):
    def __init__(
        self,
        text_dim: int,
        hidden_dim: int = 4096,
        num_freqs: int = 36,
        log_sampling: bool = False,
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoder(
            input_dims=2,
            num_freqs=num_freqs,
            include_input=False,
            log_sampling=log_sampling,
            polar_fourier=False,
            more_axis=True,
        )
        self.geo_proj = nn.Sequential(
            nn.Linear(self.pos_encoder.output_dims, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, text_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(2.6593))

    def forward(self, coords: torch.Tensor, text_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        geo_feat = F.normalize(self.location_representation(coords), dim=-1)
        text_feat = F.normalize(text_emb, dim=-1)
        return geo_feat, text_feat

    def location_representation(self, coords: torch.Tensor) -> torch.Tensor:
        return self.geo_proj(self.pos_encoder(coords))


class GeoProjector(nn.Module):
    """Notebook-style MLP projector from GeoCLIP dimension into LLM hidden size."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Geo2LLMState:
    geo_clip: dict[str, torch.Tensor]
    geo_projector: dict[str, torch.Tensor]
    geo_token_id: int


class Geo2LLMLlama(nn.Module):
    """Inject learned location embeddings at `<GEO>` token positions."""

    def __init__(
        self,
        llm: nn.Module,
        tokenizer,
        geo_clip: GeoCLIP,
        geo_token_id: int,
        geo_out_dim: int,
        geo_projector_hidden_dim: int = 2048,
        spatial_representation: str = "LATENT",
        num_cells: int | None = None,
    ):
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.geo_clip = geo_clip
        self.geo_token_id = geo_token_id
        self.geo_out_dim = geo_out_dim
        self.spatial_representation = spatial_representation
        self.embed_tokens = self.llm.get_input_embeddings()
        self.llm_hidden = self._get_llm_hidden_size(self.llm)
        self.geo_projector = GeoProjector(geo_out_dim, geo_projector_hidden_dim, self.llm_hidden)
        self.embedding_layer = None

        if spatial_representation == "GRID":
            if num_cells is None:
                raise ValueError("num_cells is required when spatial_representation='GRID'.")
            self.num_cells = num_cells
            self.embedding_layer = nn.Embedding(num_cells, geo_out_dim)
        elif spatial_representation in ("TEXT", "HASH"):
            # Text-only location codes in the prompt (`(lat, lon)` or a Base32 geohash);
            # no encoder injection. `geo_feature` is empty.
            pass
        elif spatial_representation == "HYBRID":
            # Same model-side path as LATENT: the prompt spells unnormalized coordinates as text
            # before `<GEO>`, and `geo_feature` carries the normalized representation this encoder
            # consumes. Nothing extra is needed here.
            pass
        elif spatial_representation != "LATENT":
            raise ValueError(f"Unsupported spatial_representation={spatial_representation!r}")


    @staticmethod
    def _get_llm_hidden_size(llm: nn.Module) -> int:
        base = llm.base_model.model if hasattr(llm, "base_model") and hasattr(llm.base_model, "model") else llm
        cfg = base.config
        if hasattr(cfg, "text_config") and cfg.text_config is not None and hasattr(cfg.text_config, "hidden_size"):
            return cfg.text_config.hidden_size
        if hasattr(cfg, "hidden_size"):
            return cfg.hidden_size
        raise AttributeError("Could not find hidden_size on model config.")

    def _encode_geo(self, geo_tensor: torch.Tensor, geo_attention_mask: torch.Tensor) -> torch.Tensor:
        if self.spatial_representation == "GRID":
            cell_ids = geo_tensor.squeeze(-1).long()
            encoded = self.embedding_layer(cell_ids)
            return self.geo_projector(encoded)

        batch_size, geo_len, _ = geo_tensor.shape
        flat_geo = geo_tensor.reshape(-1, 2)
        flat_valid = geo_attention_mask.reshape(-1).bool()
        encoded = torch.zeros(
            flat_geo.shape[0],
            self.geo_out_dim,
            device=geo_tensor.device,
            dtype=self.geo_projector.net[0].weight.dtype,
        )
        if flat_valid.any():
            geo_valid = self.geo_clip.location_representation(flat_geo[flat_valid])
            encoded[flat_valid] = geo_valid.to(encoded.dtype)
        encoded = encoded.view(batch_size, geo_len, self.geo_out_dim)
        return self.geo_projector(encoded)

    def _inject_geo_embeddings(
        self,
        query_ids: torch.Tensor,
        query_embeds: torch.Tensor,
        geo_embeds: torch.Tensor,
        geo_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = query_embeds.clone()
        valid_counts = geo_attention_mask.sum(dim=-1)
        for batch_idx in range(query_ids.size(0)):
            token_positions = torch.nonzero(query_ids[batch_idx] == self.geo_token_id, as_tuple=False).squeeze(-1)
            geo_count = int(valid_counts[batch_idx].item())
            if token_positions.numel() < geo_count:
                raise ValueError(
                    f"Sample {batch_idx} has {token_positions.numel()} <GEO> tokens but {geo_count} geo rows."
                )
            if geo_count:
                out[batch_idx, token_positions[:geo_count]] = geo_embeds[batch_idx, :geo_count].to(out.dtype)
        return out

    def _prepare_query_geo_embeddings(
        self,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        geo_tensor: torch.Tensor,
        geo_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_embeds = self.embed_tokens(query_ids)
        if self.spatial_representation in ("TEXT", "HASH"):
            return query_embeds, query_mask
        geo_embeds = self._encode_geo(geo_tensor, geo_attention_mask)
        query_embeds = self._inject_geo_embeddings(query_ids, query_embeds, geo_embeds, geo_attention_mask)
        return query_embeds, query_mask

    def forward(
        self,
        query_ids: torch.Tensor,
        geo_tensor: torch.Tensor,
        labels: torch.Tensor,
        query_mask: torch.Tensor,
        label_mask: torch.Tensor,
        geo_attention_mask: torch.Tensor | None = None,
    ):
        device = query_ids.device
        geo_tensor = geo_tensor.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        query_mask = query_mask.to(device=device, non_blocking=True)
        label_mask = label_mask.to(device=device, non_blocking=True)
        if geo_attention_mask is None:
            geo_attention_mask = torch.ones(geo_tensor.shape[:2], dtype=torch.bool, device=device)
        else:
            geo_attention_mask = geo_attention_mask.to(device=device, non_blocking=True)

        query_embeds, query_mask = self._prepare_query_geo_embeddings(
            query_ids,
            query_mask,
            geo_tensor,
            geo_attention_mask,
        )
        label_embeds = self.embed_tokens(labels)

        inputs_embeds = torch.cat([query_embeds, label_embeds], dim=1)
        attention_mask = torch.cat([query_mask.long(), label_mask.long()], dim=1)

        labels_for_loss = labels.masked_fill(~label_mask.bool(), -100)
        pad_id = self.tokenizer.pad_token_id
        stop_ids = set(resolve_generation_stop_token_ids(self.tokenizer, self.llm))
        # Wipe pad copies that are not also a stop token. If pad_id == eos_id, the attention
        # mask already drops padding; masking by id would remove the EOS we just trained on.
        if pad_id is not None and pad_id not in stop_ids:
            labels_for_loss = labels_for_loss.masked_fill(labels_for_loss == pad_id, -100)
        prefix_ignore = labels_for_loss.new_full((labels.size(0), query_embeds.size(1)), -100)
        labels_full = torch.cat([prefix_ignore, labels_for_loss], dim=1)

        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels_full)

    @torch.no_grad()
    def generate(
        self,
        query_ids: torch.Tensor,
        geo_tensor: torch.Tensor,
        query_mask: torch.Tensor,
        geo_attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> torch.Tensor:
        device = query_ids.device
        geo_tensor = geo_tensor.to(device=device, non_blocking=True)
        query_mask = query_mask.to(device=device, non_blocking=True)
        if geo_attention_mask is None:
            geo_attention_mask = torch.ones(geo_tensor.shape[:2], dtype=torch.bool, device=device)
        else:
            geo_attention_mask = geo_attention_mask.to(device=device, non_blocking=True)

        query_embeds, query_mask = self._prepare_query_geo_embeddings(
            query_ids,
            query_mask,
            geo_tensor,
            geo_attention_mask,
        )
        eos_token_id = resolve_generation_stop_token_ids(self.tokenizer, self.llm)
        if not eos_token_id:
            eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = getattr(self.llm.generation_config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
        return self.llm.generate(
            inputs_embeds=query_embeds,
            attention_mask=query_mask.long(),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            use_cache=True,
        )
