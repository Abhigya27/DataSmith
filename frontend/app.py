"""DSAgent: Streamlit front end. Run with ``dsagent ui`` (or ``streamlit run src/dsagent/ui/app.py``).

A thin layer: every action is a call the command line also makes (``run_pipeline``, ``predict``, ``summary``, ``compare``). Settings come from
``config.yaml`` plus what you choose on the page. Environment: ``DSAGENT_UI_OUT`` (where runs and uploads are stored, default ``runs``),
``DSAGENT_CONFIG`` (path of a config.yaml, default ``./config.yaml`` when it exists).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from dsagent import __version__, summary
from dsagent.config import load_settings
from dsagent.loading import Limits
from dsagent.predict import predict
from dsagent.privacy import data_egress_summary
from dsagent.ui import runner, service
from dsagent.utils import DsagentError

OUT = Path(os.environ.get("DSAGENT_UI_OUT", "runs"))
CONFIG = os.environ.get("DSAGENT_CONFIG") or ("config.yaml" if Path("config.yaml").is_file() else None)
PAGES = ["New run", "Runs", "Compare", "Predict", "About and privacy"]
OUTCOME = {"success": ("success", "Every stage completed."), "completed_with_warnings": ("warning", "Finished, but read the warnings before trusting the results."),
           "failed": ("error", "A required stage did not complete."), "incomplete": ("info", "Some stages have not run yet; resume to continue."),
           "interrupted": ("warning", "The run was interrupted; resume to continue."), "rejected": ("error", "The run was rejected before it started.")}
NO_TARGET = "(none: explore only)"


def banner(kind: str, text: str) -> None:
    {"success": st.success, "warning": st.warning, "error": st.error, "info": st.info}[kind](text)


def settings():
    try:
        return load_settings(CONFIG, {})
    except DsagentError as e:
        st.error(f"config.yaml problem: {e}")
        st.stop()


# ------------------------------------------------------------------ new run
def pick_source(cfg) -> Path | None:
    mode = st.radio("Data", ["Upload a file", "File path on this machine"], horizontal=True, key="src_mode")
    try:
        if mode == "Upload a file":
            up = st.file_uploader("CSV, Excel, Parquet, JSON or SQLite", type=[e.lstrip(".") for e in service.ALLOWED_EXT])
            return service.save_upload(up.name, up.getvalue(), OUT, Limits.from_settings(cfg.data)) if up else None
        text = st.text_input("Full path to the data file", key="src_path")
        return service.resolve_local_path(text) if text.strip() else None
    except DsagentError as e:
        st.error(str(e))
        return None


def new_run_page() -> None:
    cfg = settings()
    sess = runner.live_session()
    if sess is not None:
        st.info(f"A run is in progress: {Path(sess.run_dir).name}")
        progress(sess.run_dir)
        return
    src = pick_source(cfg)
    if src is None:
        st.caption("Choose a file to begin. Nothing is sent anywhere until you start a run, and by default no LLM is used.")
        return
    sheet = table = None
    try:
        ins = service.inspect_source(src, cfg)
        if ins.needs_choice == "sheet":
            sheet = st.selectbox("This workbook has several sheets. Which one holds the data?", ins.sheets)
            ins = service.inspect_source(src, cfg, sheet=sheet)
        elif ins.needs_choice == "table":
            table = st.selectbox("This database has several tables. Which one?", ins.tables)
            ins = service.inspect_source(src, cfg, table=table)
        else:
            sheet, table = ins.chosen.get("sheet"), ins.chosen.get("table")
    except DsagentError as e:
        st.error(str(e))
        return
    for w in ins.warnings:
        st.warning(w)
    st.subheader("Preview")
    st.caption(f"First {ins.n_preview_rows} rows of {len(ins.columns)} columns (a bounded read, not the whole file).")
    st.dataframe(ins.preview, use_container_width=True, height=220)

    st.subheader("Question")
    target = st.selectbox("Target: the column to predict", [NO_TARGET, *ins.columns], key="target")
    c1, c2 = st.columns(2)
    task = c1.selectbox("Task type", ["auto", "classification", "regression"], disabled=target == NO_TARGET, help="auto: decided from the target column") or "auto"
    goal = c2.text_input("Goal (optional, free text)", placeholder="e.g. predict churn; missing a churner is worse than a false alarm")
    split = st.radio("How should rows be split into train and test?", ["Random", "Chronological (time column)", "Group-aware (keep each entity together)"],
                     horizontal=True, disabled=target == NO_TARGET,
                     help="Group-aware: all rows of one customer / patient / device stay on the same side, so the test score is not inflated by near-duplicates.") or "Random"
    time_col = group_col = None
    others = [c for c in ins.columns if c != target]
    if split.startswith("Chron"):
        time_col = st.selectbox("Time column", ins.datetime_like + [c for c in others if c not in ins.datetime_like])
    elif split.startswith("Group"):
        group_col = st.selectbox("Group column", [c for c in others if c not in ins.datetime_like] or others,
                                 help="Not used as a model input. Rows with an empty value here make the run stop with a clear message.")

    with st.expander("Privacy and language model", expanded=False):
        use_llm = st.checkbox("Use the language model from config.yaml to propose plans", value=False,
                              help="Off: rule-based plans only, nothing leaves this machine. On: prompts contain column names, statistics and a few example values, never raw rows.")
        redact = st.selectbox("Redact personal data in prompts", ["auto", "on", "off"], help="auto: on unless the model server runs on this machine") or "auto"
        samples = st.slider("Example values per column placed in prompts", 0, 5, 5, help="0 = schema and statistics only")
        sens = st.multiselect("Columns whose example values are never shown to the model", ins.columns)
    with st.expander("Advanced"):
        tune = st.checkbox("Tune hyperparameters", value=True)
        r1, r2, r3 = st.columns(3)
        rv_c, rv_f, rv_t = r1.checkbox("Review cleaning plan", True), r2.checkbox("Review feature plan", False), r3.checkbox("Review tuning plan", False)
        orch_ok, mlflow_ok = _has("langgraph"), _has("mlflow")
        orch = st.selectbox("Orchestrator", ["native", "langgraph"] if orch_ok else ["native"], help="langgraph: the same stages run as a LangGraph graph") or "native"
        mlf = st.checkbox("Also log the run to MLflow", value=False, disabled=not mlflow_ok, help=None if mlflow_ok else "pip install 'dsagent[tracking]'")

    opts = service.RunOptions(source=str(src), target=None if target == NO_TARGET else target, task=task, goal=goal, time_col=time_col, group_col=group_col,
                              sheet=sheet, table=table, use_llm=use_llm, redact=redact, sample_values=samples, sensitive_columns=",".join(sens), tune=tune,
                              review_cleaning=rv_c, review_features=rv_f, review_tuning=rv_t, orchestrator=orch, mlflow=mlf)
    errors, warns = service.validate_options(opts, ins.columns)
    for msg in errors:
        st.error(msg)
    for w in warns:
        st.warning(w)
    if st.button("Start run", type="primary", disabled=bool(errors)):
        try:
            s = runner.launch(opts, OUT, CONFIG)
            st.session_state["active_run"] = s.run_dir
            st.rerun()
        except DsagentError as e:
            st.error(str(e))


def _has(mod: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(mod) is not None


@st.fragment(run_every=2)
def progress(run_dir: str) -> None:
    sess = runner.session_for(run_dir)
    snap = sess.snapshot() if sess else None
    try:
        st.dataframe(pd.DataFrame(service.stage_rows(run_dir)), hide_index=True, use_container_width=True)
    except Exception:  # noqa: BLE001 - the folder can be mid-write; try again on the next tick
        st.caption("Starting...")
    if sess is not None and snap and snap["pending"]:
        review_panel(sess, snap["pending"])
    if snap and snap["alive"]:
        if sess is not None and st.button("Stop after the current stage", key=f"stop-{run_dir}"):
            sess.request_stop()
        st.caption("A running stage is never cut off half-way; stopping takes effect when it ends, and the run can be resumed.")
    elif snap and snap["finished"]:
        if snap["error"]:
            st.error(snap["error"])
        st.session_state["view_run"] = run_dir
        st.session_state.pop("active_run", None)
        st.rerun(scope="app")


def review_panel(sess: runner.RunSession, req: runner.ReviewRequest) -> None:
    st.warning(f"Your decision is needed: review the {req.stage} plan. The run is paused until you answer.")
    for title, lines in req.sections.items():
        with st.expander(title, expanded=True):
            st.code("\n".join(lines) or "(none)", language=None)
    for n in req.notes:
        st.caption(f"note: {n}")
    a, b = st.columns(2)
    if a.button("Approve", type="primary", key=f"ok-{req.stage}"):
        sess.answer("approve")
    if b.button("Stop here", key=f"quit-{req.stage}"):
        sess.answer("quit")
    if req.can_revise:
        text = st.text_input("Or ask for changes", key=f"fb-{req.stage}", placeholder="e.g. do not drop the 'notes' column")
        if st.button("Request changes", key=f"rv-{req.stage}") and text.strip():
            sess.answer("feedback", text)


# ------------------------------------------------------------------ runs and results
def runs_page() -> None:
    live = runner.live_session()
    if live is not None:
        st.info(f"Running now: {Path(live.run_dir).name}")
        progress(live.run_dir)
    rows = service.list_runs(OUT)
    if not rows:
        st.caption("No runs yet. Start one from 'New run'.")
        return
    df = pd.DataFrame(rows).drop(columns=["path"])
    st.dataframe(df, hide_index=True, use_container_width=True)
    names = [r["run"] for r in rows]
    default = names.index(Path(st.session_state["view_run"]).name) if Path(st.session_state.get("view_run", "x")).name in names else 0
    chosen = st.selectbox("Open a run", names, index=default)
    show_run(next(r["path"] for r in rows if r["run"] == chosen))


def show_run(run_dir: str) -> None:
    try:
        s = summary.run_summary(run_dir)
    except Exception as exc:  # noqa: BLE001
        st.error(f"This run folder could not be read: {type(exc).__name__}: {exc}")
        return
    kind, text = OUTCOME.get(s["outcome"], ("info", s["outcome"]))
    banner(kind, f"{s['outcome'].replace('_', ' ').upper()}: {text}")
    if s["outcome"] in ("incomplete", "interrupted", "failed") and runner.live_session() is None and st.button("Resume this run", key=f"resume-{run_dir}"):
        try:
            runner.resume(run_dir)
            st.rerun()
        except DsagentError as e:
            st.error(str(e))
    for err in s["errors"]:
        st.error(f"{err['stage']}: {err['error']}")
    m = s.get("model")
    if m:
        c = st.columns(4)
        c[0].metric("Selected model", m["selected"] or "n/a")
        c[1].metric(f"Validation ({m['primary_metric']})", _fmt(m["validation_cv_score"]))
        c[2].metric("Final test", _fmt(m["final_test_score"]))
        c[3].metric("Baseline test", _fmt(m["baseline_test_score"]))
        for f in m.get("flags", []):
            st.warning(f)
    tabs = st.tabs(["Report", "Stages", "Data quality", "Models", "Charts", "Plans", "Files"])
    res = service.result_tables(run_dir)
    with tabs[0]:
        html = Path(run_dir, "report.html")
        if html.is_file():
            st.components.v1.html(html.read_text(encoding="utf-8"), height=800, scrolling=True)
        else:
            st.caption("No report yet.")
    with tabs[1]:
        st.dataframe(pd.DataFrame(service.stage_rows(run_dir)), hide_index=True, use_container_width=True)
        st.caption("'plan from': llm = proposed by the language model and validated; fallback = the rule-based plan ran instead; reused = a saved plan.")
    with tabs[2]:
        t = res.get("table")
        if t:
            st.caption(f"{t['n_rows']:,} rows x {t['n_cols']} columns; {t['duplicate_rows']} duplicate rows; {t['missing_pct']}% of cells missing.")
        if "findings" in res:
            st.dataframe(res["findings"], hide_index=True, use_container_width=True)
        if "cleaning" in res:
            c = res["cleaning"]
            st.write(f"Cleaning ({c['plan_source']}): rows {c['rows'][0]:,} to {c['rows'][1]:,}, columns {c['columns'][0]} to {c['columns'][1]}.")
            for line in c["log"] + c["notes"]:
                st.caption(str(line))
        if "columns" in res:
            st.dataframe(res["columns"], hide_index=True, use_container_width=True)
        if not res:
            st.caption("No profile yet.")
    with tabs[3]:
        if "leaderboard" in res:
            st.caption("Models compared by cross-validation on the training rows; the held-out test rows were used once, for the selected model only.")
            st.dataframe(res["leaderboard"], hide_index=True, use_container_width=True)
        if "importance" in res:
            st.bar_chart(res["importance"].head(15).set_index("feature")["importance"])
        if "leaderboard" not in res:
            st.caption("No model was trained in this run.")
    with tabs[4]:
        figs = service.figure_files(run_dir)
        for f in figs:
            st.image(str(f), caption=f.stem.replace("_", " "))
        if not figs:
            st.caption("No charts.")
    with tabs[5]:
        for p in sorted(Path(run_dir, "plans").glob("*.json")):
            with st.expander(p.stem):
                st.json(json.loads(p.read_text(encoding="utf-8")))
    with tabs[6]:
        for name, p in service.artifact_files(run_dir).items():
            st.download_button(f"Download {name}", p.read_bytes(), file_name=p.name, key=f"dl-{run_dir}-{name}")


def _fmt(v: object) -> str:
    return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


# ------------------------------------------------------------------ compare
def compare_page() -> None:
    rows = [r for r in service.list_runs(OUT) if not r["outcome"].startswith("unreadable")]
    if len(rows) < 2:
        st.caption("Compare needs at least two runs.")
        return
    names = [r["run"] for r in rows]
    a, b = st.selectbox("Run A", names, index=0), st.selectbox("Run B", names, index=1)
    if a == b:
        st.info("Choose two different runs.")
        return
    path = {r["run"]: r["path"] for r in rows}
    c = summary.compare_runs(path[a], path[b])
    comp = c["comparability"]
    kind = {"comparable": "success", "comparable_with_differences": "warning", "not_comparable": "error", "cannot_verify": "info"}[comp["verdict"]]
    banner(kind, summary._VERDICT_TEXT[comp["verdict"]])
    table = pd.DataFrame([{"check": x["label"], "A": _show(x, "a"), "B": _show(x, "b"), "result": x["status"], "matters": x["impact"]}
                          for x in comp["checks"] if x["status"] != "same" or x["impact"] == "blocking"])
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.download_button("Download the comparison (JSON)", json.dumps(c, indent=2, default=str), file_name="comparison.json")


def _show(x: dict, side: str) -> str:
    return "unknown" if not x[f"{side}_recorded"] else ("none" if x[side] is None else str(x[side])[:60])


# ------------------------------------------------------------------ predict
def predict_page() -> None:
    rows = [r for r in service.list_runs(OUT) if r["model"] and r["outcome"] in ("success", "completed_with_warnings")]
    if not rows:
        st.caption("Predictions need a finished run that trained a model.")
        return
    names = [r["run"] for r in rows]
    chosen = st.selectbox("Use the model from run", names)
    run_dir = next(r["path"] for r in rows if r["run"] == chosen)
    st.caption("The new file needs the same input columns the model was trained on. Extra columns are ignored.")
    cfg = settings()
    try:
        up = st.file_uploader("New data", type=[e.lstrip(".") for e in service.ALLOWED_EXT], key="pred_up")
        src = service.save_upload(up.name, up.getvalue(), OUT, Limits.from_settings(cfg.data)) if up else None
    except DsagentError as e:
        st.error(str(e))
        return
    if src and st.button("Score these rows", type="primary"):
        out = Path(run_dir) / "predictions" / f"{src.stem}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        try:
            path, n, notes = predict(run_dir, src, out)
        except DsagentError as e:
            st.error(str(e))
            return
        st.success(f"Scored {n:,} row(s).")
        for note in notes:
            st.caption(f"note: {note}")
        df = pd.read_csv(path)
        st.dataframe(df.head(100), use_container_width=True)
        st.download_button("Download predictions (CSV)", Path(path).read_bytes(), file_name=Path(path).name)


# ------------------------------------------------------------------ about
def about_page() -> None:
    cfg = settings()
    st.write(f"DSAgent {__version__} on Python {sys.version.split()[0]}. Runs and uploads are stored in `{OUT.resolve()}`.")
    eg = data_egress_summary(cfg)
    st.subheader("What can leave this machine")
    st.write("With the current config.yaml **and the language model switched on**: "
             + ("prompts are sent to an external service" if eg["leaves_machine"] else "prompts stay on this machine (local model server)")
             + f"; redaction of personal data is {'on' if eg['redaction'] else 'off'}; at most {eg['example_values_per_column']} example values per column are included. "
               "With the model off (the default on the New run page) nothing is sent anywhere.")
    st.caption("Redaction is pattern-based (emails, phone and card numbers, national IDs, IP addresses, credentials). It cannot recognise names or free text.")
    st.subheader("Optional components")
    for label, mod, extra in (("LangChain", "langchain_core", "langchain"), ("LangGraph", "langgraph", "langgraph"), ("MLflow", "mlflow", "tracking")):
        st.write(f"- {label}: {'installed' if _has(mod) else f'not installed (pip install dsagent[{extra}])'}")


def main() -> None:
    st.set_page_config(page_title="DSAgent", page_icon=None, layout="wide")
    st.title("DSAgent")
    page = st.sidebar.radio("Page", PAGES, key="page") or PAGES[0]
    st.sidebar.caption(f"v{__version__} | runs in {OUT}")
    {"New run": new_run_page, "Runs": runs_page, "Compare": compare_page, "Predict": predict_page, "About and privacy": about_page}[page]()


main()
