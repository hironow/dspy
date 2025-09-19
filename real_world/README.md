# Real-World GEPA Examples (v3+)

This folder contains a minimal, practical example showing how to use `dspy.GEPA` in DSPy v3+. It is designed to be simple and easy to extend.

Files:
- `real_world/simple_gepa_basic.py` — a single-file, minimal GEPA workflow:
  - Defines a tiny DSPy program (Predict-based QA)
  - Implements a GEPA-friendly metric with feedback
  - Runs GEPA with a small train/val split
  - Prints pre/post optimization scores
  - Uses `loguru` for clear, step-by-step logging
  - Prints a compact ASCII table of GEPA candidates (Pareto summary)
  - Dataset (質問/回答)とプロンプトの説明は日本語（数学問題は避けています）

Prerequisites:
- DSPy installed: `pip install dspy`
- GEPA core library: `pip install gepa`
- A language model configured for DSPy (either a real LM or `DummyLM` for local dry runs)

How to run:
1) Option A — dry run with `DummyLM` (no external calls):
   - Run: `python real_world/simple_gepa_basic.py --dummy`
   - With detailed logs: `python real_world/simple_gepa_basic.py --dummy --log-level DEBUG`

2) Option B — real LM (replace with your provider/model):
   - Edit `simple_gepa_basic.py` near “LM configuration” and set something like:
     ```python
     task_lm = dspy.LM(model="gpt-4o-mini", temperature=0.0)
     reflection_lm = dspy.LM(model="gpt-4o", temperature=0.7)
     dspy.settings.configure(lm=task_lm)
     ```
   - Run: `python real_world/simple_gepa_basic.py`

Key GEPA compile arguments (required/important):
- `metric`: A function with 5 parameters `(gold, pred, trace, pred_name, pred_trace)` returning either a float score or `{"score": float, "feedback": str}`. Text feedback greatly improves sample efficiency.
- Exactly one of: `auto` | `max_metric_calls` | `max_full_evals`
  - Example: `auto="light"` or `max_metric_calls=1200`
- `reflection_lm` OR `instruction_proposer`
  - GEPA uses `reflection_lm` to analyze traces/feedback and propose improved instructions. Alternatively, provide a custom `instruction_proposer`.
- `trainset`: Non-empty list of `dspy.Example`
- `valset` (recommended): Small, representative set for Pareto tracking and selection
- Optional: `track_stats=True` to attach `optimized.detailed_results` for analysis

Notes:
- GEPA expects predictor-level names/traces to be available. The adapter in DSPy takes care of this.
- If you do not provide a `valset`, GEPA will use `trainset` for Pareto tracking (useful for inference-time search/overfitting to the batch).
- `teacher` is currently not supported by GEPA in DSPy v3.
- This example uses `loguru` for human-friendly logs. Pass `--log-level` (e.g., `DEBUG`, `INFO`) to control verbosity.
- After optimization, a compact ASCII table summarizes the top candidates (idx, score, coverage, discovery calls).
 - 本例の対象物（データセットの質問/回答、プロンプトの説明）は日本語です。英語に戻したい場合は `build_tiny_dataset()` と `with_instructions(...)` を編集してください。

Next steps / extensions:
- Swap the metric to a task-specific one (e.g., semantic similarity, schema validation, code execution success, etc.)
- Use `track_stats=True` and inspect `optimized.detailed_results` (Pareto frontier scores, candidates, per-instance best outputs)
- Try inference-time search by calling GEPA with `valset=trainset` and `track_best_outputs=True` (see docstring in `dspy.GEPA`)
