from __future__ import annotations

import argparse
import pickle
from functools import partial
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from SpatialExplicitLM.build import load_checkpoint
from SpatialExplicitLM.config import load_config
from SpatialExplicitLM.evaluate import evaluate_generated_results, print_metrics
from SpatialExplicitLM.data import (
    EVALUATION_MODES,
    CoordinateGenerationDataset,
    collate_geo_chat_template,
    coordinate_pair_as_list,
    decode_generated_completions,
    extract_reasoning_answer,
    grid_bounds_from_records,
    load_all_records,
    load_coordinate_records,
    load_records_for_evaluation_mode,
    normalize_evaluation_mode,
    resolve_inference_chat_options,
    resolve_reasoning_config,
    resolve_system_prompt,
    unique_coordinate_tensors,
)
from SpatialExplicitLM.runtime import apply_runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate location descriptions for geospatial memory (training locations) "
            "or spatial generalization (held-out test locations)."
        )
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--mode",
        choices=EVALUATION_MODES,
        required=True,
        help=(
            "memory: generate at observed training locations; "
            "generalization: generate at spatially withheld test locations."
        ),
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip post-generation evaluation against ground-truth descriptions.",
    )
    parser.add_argument(
        "--coordinates-csv",
        type=str,
        default=None,
        help="Optional CSV override; if set, ignores --mode for which coordinates to load.",
    )
    return parser.parse_args()


def load_mode_dataframe(args: argparse.Namespace, cfg) -> pd.DataFrame:
    """Load rows for the selected evaluation mode, or a custom coordinates CSV."""
    if args.coordinates_csv:
        return load_coordinate_records(args.coordinates_csv, cfg.data)
    return load_records_for_evaluation_mode(cfg.data, args.mode)


def load_ground_truth_dataframe(args: argparse.Namespace, cfg) -> pd.DataFrame:
    df = load_mode_dataframe(args, cfg)
    cols = [cfg.data.latitude_col, cfg.data.longitude_col]
    if cfg.data.description_col in df.columns:
        cols.append(cfg.data.description_col)
    return df[cols].drop_duplicates()


def load_coordinates(args: argparse.Namespace, cfg) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Model-facing coordinates plus the real-world degrees for TEXT/HYBRID prompt text."""
    df = load_mode_dataframe(args, cfg)
    return unique_coordinate_tensors(df, cfg.data.latitude_col, cfg.data.longitude_col)


def extract_batch_coordinates(batch: dict) -> list[list[float]]:
    """Read per-sample lat/lon from a collated batch (never grid cell ids)."""
    if "coordinates" in batch:
        return [coordinate_pair_as_list(c[0], c[1]) for c in batch["coordinates"]]
    geo = batch["geo_feature"]
    if geo.ndim == 3 and geo.size(1) > 0 and geo.size(2) == 2:
        rows = geo[:, 0, :].cpu().numpy()
        return [coordinate_pair_as_list(row[0], row[1]) for row in rows]
    raise ValueError(
        "Batch has no usable lat/lon coordinates. "
        f"geo_feature shape={tuple(geo.shape)}; keys={list(batch.keys())}"
    )


def generate_descriptions(
    model,
    coordinates: torch.Tensor,
    cfg,
    grid_bounds: tuple[float, float, float, float] | None = None,
    text_coordinates: torch.Tensor | None = None,
) -> list[tuple[list[float], str]]:
    tokenizer = model.tokenizer
    dataset = CoordinateGenerationDataset(
        coordinates,
        spatial_representation=cfg.model.spatial_representation,
        grid_bounds=grid_bounds,
        grid_num_axis=cfg.model.grid_num_axis,
        geohash_length=cfg.model.geohash_length,
        text_coordinates=text_coordinates,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.generation.batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_geo_chat_template,
            tokenizer=tokenizer,
            system_prompt=resolve_system_prompt(cfg.reasoning),
            prompt_padding_side="left",
            **resolve_inference_chat_options(tokenizer, cfg.model.spatial_representation),
        ),
    )
    device = next(model.parameters()).device
    model.eval()
    results: list[tuple[list[float], str]] = []
    for _ in range(cfg.generation.generations_per_coordinate):
        for batch in tqdm(loader, desc="generating"):
            with torch.no_grad():
                outputs = model.generate(
                    query_ids=batch["prompt_input_ids"].to(device),
                    geo_tensor=batch["geo_feature"].to(device),
                    geo_attention_mask=batch["geo_attention_mask"].to(device),
                    query_mask=batch["prompt_attention_mask"].to(device),
                    max_new_tokens=cfg.generation.max_new_tokens,
                    temperature=cfg.generation.temperature,
                    top_k=cfg.generation.top_k,
                    top_p=cfg.generation.top_p,
                    do_sample=cfg.generation.do_sample,
                )
            generated = decode_generated_completions(tokenizer, outputs)
            if cfg.reasoning.enabled:
                generated = [
                    extract_reasoning_answer(
                        text,
                        close_tag=cfg.reasoning.thinking_close_tag,
                        answer_prefix=cfg.reasoning.answer_prefix,
                    )
                    for text in generated
                ]
            coords = extract_batch_coordinates(batch)
            for i in range(len(generated)):
                results.append((coords[i], generated[i]))
    return results


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    mode = normalize_evaluation_mode(args.mode)
    args.mode = mode
    args.checkpoint = cfg.generation.checkpoint

    output = Path(cfg.generation.output)
    output = output.with_name(f"{output.stem}_{mode}{output.suffix}")
    args.output = str(output)

    if args.coordinates_csv:
        source_desc = args.coordinates_csv
    else:
        source_desc = (
            f"{cfg.data.training_data} (memory / observed)"
            if mode == "memory"
            else f"{cfg.data.test_data} (generalization / withheld)"
        )
    print(f"mode={mode}; loading coordinates from {source_desc}")
    apply_runtime_config(cfg.runtime)
    model = load_checkpoint(cfg.model, args.checkpoint)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if cfg.reasoning.enabled:
        cfg.reasoning = resolve_reasoning_config(cfg.reasoning, model.tokenizer)
        print(
            "reasoning thinking tags resolved from tokenizer chat template: "
            f"open={cfg.reasoning.thinking_open_tag!r} close={cfg.reasoning.thinking_close_tag!r}"
        )
    coordinates, text_coordinates = load_coordinates(args, cfg)
    grid_bounds = None
    if cfg.model.spatial_representation == "GRID":
        grid_bounds = grid_bounds_from_records(
            load_all_records(cfg.data),
            cfg.data.latitude_col,
            cfg.data.longitude_col,
        )
    results = generate_descriptions(
        model,
        coordinates,
        cfg,
        grid_bounds=grid_bounds,
        text_coordinates=text_coordinates,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(results, f)
    print(f"Saved {len(results)} generated descriptions to {output}")

    run_eval = cfg.generation.evaluate_after_generation and not args.no_eval
    if run_eval:
        ground_truth_df = load_ground_truth_dataframe(args, cfg)
        if cfg.data.description_col not in ground_truth_df.columns:
            print(
                f"Skipping evaluation: ground truth has no {cfg.data.description_col!r} column "
                f"(columns: {list(ground_truth_df.columns)})."
            )
        else:
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            sentence_model = SentenceTransformer(cfg.generation.eval_sentence_model, device=device)
            metrics = evaluate_generated_results(
                results,
                ground_truth_df,
                sentence_model,
                latitude_col=cfg.data.latitude_col,
                longitude_col=cfg.data.longitude_col,
                description_col=cfg.data.description_col,
            )
            print_metrics(
                metrics,
                title=f"Post-generation evaluation [{mode}] (matched by coordinates):",
            )


if __name__ == "__main__":
    main()
