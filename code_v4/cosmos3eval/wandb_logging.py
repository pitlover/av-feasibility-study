"""Opt-in W&B logging of completed Cosmos3 RRE/RTE/ATE evaluations.

Each evaluation is a separate W&B run. The same project/group as training
allows comparing metrics while keeping training steps and eval aggregation
semantically separate. No import/network activity when disabled.
"""
from __future__ import annotations

import csv
from pathlib import Path


def _scalars(summary):
    scalars = {}
    for name in ('rre_deg_mean', 'rte_m_mean', 'ate_m_rmse',
                 'final_position_error_m_mean', 'n_clips', 'n_failed'):
        value = summary['overall'].get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            scalars[f'eval/overall/{name}'] = value
    for dataset, stats in summary.get('by_dataset', {}).items():
        for name in ('rre_deg_mean', 'rte_m_mean', 'ate_m_rmse', 'n_clips'):
            value = stats.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                scalars[f'eval/{dataset}/{name}'] = value
    return scalars


def log_evaluation(*, summary: dict, metrics_dir, mode: str,
                   project: str = 'cosmos3_av_id', group: str | None = None,
                   name: str | None = None, eval_dir=None, pred_dir=None):
    if mode == 'disabled':
        return None
    if mode not in ('online', 'offline'):
        raise ValueError(f'Unsupported W&B mode: {mode}')
    if summary.get('overall', {}).get('oracle'):
        raise ValueError('GT Oracle must not be uploaded as model evaluation')
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError('W&B is enabled. Install it with: pip install wandb') from exc

    metrics_dir = Path(metrics_dir).expanduser().resolve()
    metrics_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir = metrics_dir / 'wandb'
    wandb_dir.mkdir(parents=True, exist_ok=True)
    model = str(summary['overall']['model'])
    run_config = dict(
        model=model, eval_dir=str(Path(eval_dir).resolve()) if eval_dir else None,
        pred_dir=str(Path(pred_dir).resolve()) if pred_dir else None,
        metrics_definition=summary['overall'].get('metrics_definition'),
        oracle=False,
    )
    with wandb.init(project=project, group=group, name=name or f'eval_{model}',
                    job_type='evaluation', mode=mode, dir=str(wandb_dir),
                    config=run_config, reinit='create_new') as run:
        metrics = _scalars(summary)
        run.log(metrics)
        run.summary.update(metrics)
        per_clip = metrics_dir / 'per_clip.csv'
        if per_clip.is_file():
            with per_clip.open(newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames or []
                rows = list(reader)
            # W&B Tables allow filtering individual failure cases.
            if rows and len(rows) <= 5000:
                numeric_fields = {'rre_deg_mean','rte_m_mean','ate_m_rmse',
                                  'final_position_error_m','final_rotation_error_deg',
                                  'acquisition_duration_s'}
                data = [[float(r[c]) if c in numeric_fields and r[c] else r[c]
                         for c in columns] for r in rows]
                run.log({'eval/per_clip': wandb.Table(columns=columns, data=data)})
        # CSV and JSON remain inspectable in the original metrics folder,
        # and can be retrieved as a versioned evaluation artifact.
        artifact = wandb.Artifact(f'eval-{model}', type='evaluation')
        for filename in ('summary.json', 'per_clip.csv', 'per_frame.csv'):
            path = metrics_dir / filename
            if path.is_file():
                artifact.add_file(str(path))
        run.log_artifact(artifact)
        print(f'[WANDB EVAL] mode={mode} project={project} group={group or "(none)"} '
              f'model={model} metrics={len(metrics)}')
    return metrics
