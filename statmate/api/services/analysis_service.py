"""Analysis service for running statistical analyses.

This service wraps the existing LangGraph workflow and manages analysis execution.
"""

import logging
import math
from datetime import datetime
from io import StringIO
from typing import Any

from sqlalchemy.orm import Session

from config.settings import settings
from database.models import Analysis, AnalysisStatus
from statmate.api.services.credential_service import CredentialService, QuotaExceededError
from statmate.api.services.dataset_service import DatasetService
from statmate.api.services.storage_service import StorageService
from statmate.api.services.visualization_service import VisualizationService
from statmate.api.services.workflow_graph_service import render_workflow_graph
from statmate.core.config import Config
from statmate.core.model_config import ModelProvider, ModelProviderConfig
from statmate.workflow.graph_builder import create_default_checkpointer
from statmate.workflow.graph_metadata import (
    map_step_to_node_id,
    workflow_payload,
)
from statmate.workflow.statmate_flow_refactored import StatMateWorkflow

logger = logging.getLogger(__name__)
WORKFLOW_STEP_TARGET = 10


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect provider rate limit errors without tight coupling to SDK types."""
    msg = str(exc).lower()
    return 'rate limit' in msg or 'rate_limit_exceeded' in msg


def _state_value(state: Any, key: str, default: Any) -> Any:
    """Safely fetch a key/attribute from a WorkflowState or dict."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _sanitize_for_json(value: Any) -> Any:
    """Recursively replace NaN/Inf floats with None for JSON compliance."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    return value


def _clean_probabilities(probabilities: Any) -> dict[str, float]:
    """Ensure probabilities dict only contains finite numeric values."""
    if not isinstance(probabilities, dict):
        return {}
    cleaned: dict[str, float] = {}
    for key, value in probabilities.items():
        if isinstance(value, (int, float)) and value is not None and math.isfinite(float(value)):
            cleaned[key] = float(value)
    return cleaned


def _build_test_hierarchy(
    decision_steps: list[dict[str, Any]] | None,
    assumption_log: list[dict[str, Any]] | None,
    reviewer_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a lightweight hierarchy summarizing tests and assumption failures."""
    attempted: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    assumption_by_test: dict[str, list[dict[str, Any]]] = {}

    for entry in assumption_log or []:
        test_key = str(entry.get('test_type') or entry.get('node') or 'unknown')
        assumption_by_test.setdefault(test_key, []).append(entry)
        for failure in entry.get('failures', []) or []:
            failures.append({'test': test_key, 'message': failure, 'timestamp': entry.get('timestamp')})

    chosen_test = None
    for step in reversed(decision_steps or []):
        step_name = step.get('step')
        if step_name and step_name.lower() not in ('summary', 'reviewer'):
            chosen_test = step_name
            break

    for step in decision_steps or []:
        name = step.get('step') or 'Step'
        attempted.append(
            {
                'name': name,
                'detail': step.get('detail'),
                'p_value': step.get('p_value'),
                'assumptions': assumption_by_test.get(name, []),
                'timestamp': step.get('timestamp'),
                'step_index': step.get('step_index'),
                'total_steps': step.get('total_steps'),
                'node': step.get('node'),
            }
        )

    return {
        'attempted': attempted,
        'failures': failures,
        'assumptions': assumption_by_test,
        'chosen_test': chosen_test,
        'reviewer': reviewer_report,
    }


class AnalysisService:
    """Service for managing statistical analysis execution."""

    WORKFLOW_STEP_TARGET = WORKFLOW_STEP_TARGET

    @staticmethod
    def sanitize_analysis(analysis: Analysis) -> Analysis:
        """Normalize non-finite numeric fields to keep API responses JSON-safe."""
        analysis.probabilities = _clean_probabilities(_sanitize_for_json(analysis.probabilities))
        analysis.decision_steps = _sanitize_for_json(analysis.decision_steps) or []
        analysis.assumption_log = _sanitize_for_json(analysis.assumption_log) or []
        return analysis

    @staticmethod
    def build_workflow_graph_state(
        decision_steps: list[dict[str, Any]] | None,
        test_hierarchy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Expose workflow metadata + progress for API consumers."""
        return workflow_payload(decision_steps or [], test_hierarchy)

    @staticmethod
    def workflow_graph_for_analysis(analysis: Analysis) -> dict[str, Any]:
        """Load graph metadata + state for a persisted analysis."""
        test_hierarchy: dict[str, Any] | None = None
        try:
            stored_results = StorageService.read_results(analysis.id)
            if stored_results:
                if stored_results.get('workflow_graph'):
                    return stored_results['workflow_graph']
                # results_detail may be nested
                detail = stored_results.get('results_detail') or stored_results
                test_hierarchy = detail.get('test_hierarchy') or stored_results.get('test_hierarchy')
        except Exception:
            test_hierarchy = None

        return AnalysisService.build_workflow_graph_state(analysis.decision_steps or [], test_hierarchy)

    @staticmethod
    def create_analysis(
        db: Session,
        dataset_id: str,
        *,
        selected_columns: list[str] | None = None,
        configuration: dict[str, Any] | None = None,
        route_override: str | None = None,
        model_name: str | None = None,
        provider: str | None = None,
        user_id: str | None = None,
        overwrite: bool = False,
    ) -> Analysis:
        """Create a new analysis record.

        Args:
            db: The database session.
            dataset_id: The UUID of the dataset to analyze.
            selected_columns: The columns to analyze. If None, all columns are used.
            configuration: Optional analysis configuration.
            route_override: Optional user override for routing choice.
            model_name: The AI model to use for the analysis.
            provider: The model provider (e.g., 'openai', 'ollama').

        Returns:
            The created Analysis model.
        """
        # Ownership check when applicable
        from config.settings import settings

        ds = DatasetService.get_dataset(db, dataset_id, user_id=user_id if settings.AUTH_REQUIRED else None)
        if settings.AUTH_REQUIRED and not ds:
            raise ValueError('Dataset not found or not owned by user')

        query = db.query(Analysis).filter(Analysis.dataset_id == dataset_id)
        if user_id:
            query = query.filter(Analysis.user_id == user_id)
        latest = query.order_by(Analysis.version.desc(), Analysis.start_time.desc().nullslast()).first()
        next_version = (latest.version if latest and latest.version else 0) + 1

        if overwrite and latest:
            latest.superseded_at = datetime.utcnow()
            db.commit()

        config_payload = dict(configuration or {})
        if route_override:
            config_payload['route_override'] = route_override

        analysis = Analysis(
            dataset_id=dataset_id,
            user_id=user_id,
            status=AnalysisStatus.PENDING,
            selected_columns=selected_columns,
            configuration=config_payload,
            model_name=model_name,
            provider=provider,
            version=next_version,
        )

        db.add(analysis)
        db.commit()
        db.refresh(analysis)

        logger.info('Created analysis: %s with model: %s', analysis.id, model_name or 'default')
        return analysis

    @staticmethod
    def get_analysis(db: Session, analysis_id: str, *, user_id: str | None = None) -> Analysis | None:
        """Get an analysis by its ID.

        Args:
            db: The database session.
            analysis_id: The analysis UUID.

        Returns:
            The Analysis model or None if not found.
        """
        query = db.query(Analysis).filter(Analysis.id == analysis_id)
        if user_id:
            query = query.filter(Analysis.user_id == user_id)
        return query.first()

    @staticmethod
    def list_analyses(
        db: Session, *, skip: int = 0, limit: int = 100, user_id: str | None = None, dataset_id: str | None = None
    ) -> list[Analysis]:
        """List all analyses with pagination.

        Args:
            db: The database session.
            skip: The number of records to skip.
            limit: The maximum number of records to return.

        Returns:
            A list of Analysis models.
        """
        query = db.query(Analysis)
        if user_id:
            query = query.filter(Analysis.user_id == user_id)
        if dataset_id:
            query = query.filter(Analysis.dataset_id == dataset_id)
        return query.order_by(Analysis.version.desc(), Analysis.start_time.desc()).offset(skip).limit(limit).all()

    @staticmethod
    def _update_analysis_status(
        db: Session,
        analysis: Analysis,
        *,
        status: AnalysisStatus,
        error_message: str | None = None,
    ) -> None:
        """Private helper to update the analysis status and commit to the database.

        Args:
            db: The database session.
            analysis: The analysis object to update.
            status: The new status to set.
            error_message: An optional error message if the status is FAILED.
        """
        analysis.status = status
        analysis.end_time = datetime.utcnow()
        if error_message:
            analysis.error_message = error_message
        db.commit()

    @staticmethod
    def _build_model_config_for_analysis(db: Session, analysis: Analysis) -> Config:
        """Create a Config with provider credentials scoped to the analysis owner."""
        multi_model_config = settings.create_multi_model_config()

        if analysis.provider:
            try:
                multi_model_config.default_provider = ModelProvider(analysis.provider)
            except ValueError:
                pass
        if analysis.model_name:
            multi_model_config.default_model_name = analysis.model_name

        if analysis.user_id:
            stored_keys = CredentialService.load_credentials(db, analysis.user_id)
            for provider_key, api_key in stored_keys.items():
                mapped_key = provider_key
                if provider_key == 'gemini':
                    mapped_key = 'google'
                try:
                    provider_enum = ModelProvider(mapped_key)
                except ValueError:
                    continue
                existing = multi_model_config.providers.get(provider_enum)
                if existing:
                    existing.api_key = api_key
                    existing.enabled = True
                else:
                    multi_model_config.providers[provider_enum] = ModelProviderConfig(
                        provider=provider_enum, api_key=api_key, enabled=True
                    )

        return Config(model=multi_model_config)

    @staticmethod
    def run_analysis(db: Session, analysis_id: str, *, user_id: str | None = None) -> Analysis:
        """Execute a statistical analysis using the refactored workflow.

        Args:
            db: The database session.
            analysis_id: The analysis UUID.

        Returns:
            The updated Analysis model.

        Raises:
            ValueError: If the analysis or its associated dataset is not found.
        """
        analysis = AnalysisService.get_analysis(db, analysis_id)
        if not analysis:
            raise ValueError(f'Analysis not found: {analysis_id}')
        if user_id and analysis.user_id and analysis.user_id != user_id:
            raise ValueError('Analysis does not belong to this user')
        if analysis.configuration is None:
            analysis.configuration = {}

        analysis.status = AnalysisStatus.RUNNING
        analysis.start_time = datetime.utcnow()
        analysis.decision_steps = []
        analysis.intermediate_log = ''
        analysis.assumption_log = []
        db.commit()
        workflow_config = AnalysisService._build_model_config_for_analysis(db, analysis)

        log_stream = StringIO()
        log_handler = logging.StreamHandler(log_stream)
        log_handler.setLevel(logging.INFO)

        log_file = settings.get_log_path(analysis_id)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        log_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        workflow_logger = logging.getLogger('statmate')
        workflow_logger.addHandler(log_handler)
        workflow_logger.addHandler(file_handler)

        # Persist log path immediately so clients can poll the live log
        analysis.log_path = str(log_file)
        db.commit()

        seen_step_ids: set[str] = set()
        seen_assumption_ids: set[str] = set()

        def _persist_decision_steps(new_steps: list[dict[str, Any]], *, node: str | None = None) -> None:
            """Append streamed decision steps and flush to the database."""
            if not new_steps:
                return
            stored = analysis.decision_steps or []
            for step in new_steps:
                if not isinstance(step, dict):
                    continue
                entry = dict(step)
                if node and 'node' not in entry:
                    entry['node'] = node
                node_id = entry.get('node_id') or map_step_to_node_id(entry.get('node') or entry.get('step'))
                if node_id:
                    entry['node_id'] = node_id
                step_id = str(entry.get('timestamp') or entry.get('step') or f'{node}-{len(stored)}')
                if step_id in seen_step_ids:
                    continue
                seen_step_ids.add(step_id)
                entry.setdefault('step_index', len(stored) + 1)
                entry.setdefault('total_steps', WORKFLOW_STEP_TARGET)
                progress = min(1.0, entry['step_index'] / WORKFLOW_STEP_TARGET)
                entry.setdefault('progress_pct', round(progress * 100, 2))
                stored.append(entry)
            analysis.decision_steps = stored

        def _persist_assumption_log(entries: list[dict[str, Any]]) -> None:
            """Persist assumption diagnostics without duplicating entries."""
            if not entries:
                return
            stored = analysis.assumption_log or []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                entry_id = str(entry.get('timestamp') or entry.get('test_type') or len(stored))
                if entry_id in seen_assumption_ids:
                    continue
                seen_assumption_ids.add(entry_id)
                stored.append(entry)
            analysis.assumption_log = stored

        def _persist_intermediate_log() -> None:
            """Write the rolling in-memory log to the analysis row."""
            analysis.intermediate_log = log_stream.getvalue()

        def _on_state_update(state_update: Any) -> None:
            """Handle LangGraph stream updates as they arrive."""
            if not isinstance(state_update, dict) or not state_update:
                return
            node, state_val = next(iter(state_update.items()))
            trace_entries = _state_value(state_val, 'execution_trace', []) or []
            if isinstance(trace_entries, list):
                _persist_decision_steps(trace_entries, node=str(node))
            assumption_entries = _state_value(state_val, 'assumption_log', []) or []
            if isinstance(assumption_entries, list):
                _persist_assumption_log(assumption_entries)
            _persist_intermediate_log()
            db.commit()
            try:
                file_handler.flush()
                log_handler.flush()
            except Exception:
                pass

        def _record_system_step(step: str, detail: str) -> None:
            """Persist non-agent steps (fallbacks, retries) for clients."""
            payload = {'step': step, 'detail': detail, 'timestamp': datetime.utcnow().isoformat()}
            _persist_decision_steps([payload])
            _persist_intermediate_log()
            db.commit()

        try:
            df = DatasetService.load_dataset_dataframe(db, analysis.dataset_id, user_id=user_id or analysis.user_id)
            if df is None:
                raise ValueError(f'Dataset not found for analysis: {analysis.dataset_id}')

            logger.info('=' * 80)
            logger.info('🚀 STARTING ANALYSIS: %s', analysis_id)
            logger.info('   Model: %s', analysis.model_name or 'default')
            logger.info('   Dataset ID: %s', analysis.dataset_id)
            logger.info('   Rows: %d, Columns: %d', len(df), len(df.columns))
            logger.info('=' * 80)

            provider_for_quota = analysis.provider or (
                workflow_config.model.default_provider.value if getattr(workflow_config, 'model', None) else None
            )
            if analysis.user_id and provider_for_quota:
                try:
                    CredentialService.enforce_quota(db, analysis.user_id, str(provider_for_quota))
                except QuotaExceededError as exc:
                    _record_system_step('Quota exceeded', str(exc))
                    raise

            fallback_attempted = False

            def _run_workflow(model_override: str | None = None) -> Any:
                if model_override and workflow_config.model:
                    workflow_config.model.default_model_name = model_override
                workflow = StatMateWorkflow(
                    config=workflow_config,
                    checkpointer=create_default_checkpointer(),
                )
                return workflow.run(
                    data=df,
                    target_columns=analysis.selected_columns,
                    paired=analysis.configuration.get('paired', False),
                    do_association=analysis.configuration.get('do_association', False),
                    model_name=model_override or analysis.model_name,
                    provider=analysis.provider,
                    route_override=analysis.configuration.get('route_override'),
                    on_update=_on_state_update,
                    thread_id=analysis_id,
                )

            try:
                # First attempt with requested/default model
                result_state = _run_workflow()

            except Exception as e:
                if (
                    _is_rate_limit_error(e)
                    and not fallback_attempted
                    and analysis.provider in (None, '', 'openai')
                    and settings.OPENAI_FALLBACK_ENABLED
                ):
                    fallback_attempted = True
                    fallback_model = settings.OPENAI_FALLBACK_MODEL
                    log_stream.write(
                        f'\nRate limit hit for {analysis.model_name}, retrying with fallback {fallback_model}\n'
                    )
                    logger.warning(
                        'Rate limit for %s, retrying analysis %s with fallback model %s',
                        analysis.model_name,
                        analysis_id,
                        fallback_model,
                    )
                    # Switch to fallback model for this run and future retrievals
                    analysis.model_name = fallback_model
                    db.commit()
                    _record_system_step(
                        'Model fallback',
                        f'Rate limit detected for {analysis.model_name or "default"}, retrying with {fallback_model}',
                    )
                    result_state = _run_workflow(model_override=fallback_model)
                else:
                    raise

            messages = _state_value(result_state, 'results', [])
            probabilities = _clean_probabilities(_sanitize_for_json(_state_value(result_state, 'probabilities', {})))
            execution_trace = _sanitize_for_json(_state_value(result_state, 'execution_trace', [])) or []
            reviewer_report = _sanitize_for_json(_state_value(result_state, 'reviewer_report', None))

            def _normalize_steps(steps: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
                """Ensure step metadata (ordering/progress) is populated."""
                normalized: list[dict[str, Any]] = []
                total = AnalysisService.WORKFLOW_STEP_TARGET
                for idx, raw in enumerate(steps or [], start=1):
                    if not isinstance(raw, dict):
                        continue
                    entry = dict(raw)
                    node_id = entry.get('node_id') or map_step_to_node_id(entry.get('node') or entry.get('step'))
                    if node_id:
                        entry['node_id'] = node_id
                    entry.setdefault('step_index', idx)
                    entry.setdefault('total_steps', total)
                    progress = min(1.0, entry['step_index'] / total)
                    entry.setdefault('progress_pct', round(progress * 100, 2))
                    normalized.append(entry)
                return normalized

            # Prefer the final execution_trace from workflow state to guarantee all steps are present.
            if execution_trace:
                analysis.decision_steps = _normalize_steps(execution_trace)
            elif analysis.decision_steps:
                analysis.decision_steps = _normalize_steps(analysis.decision_steps)
            analysis.decision_steps = _sanitize_for_json(analysis.decision_steps) or []

            assumption_log = _sanitize_for_json(
                _state_value(result_state, 'assumption_log', []) or analysis.assumption_log or []
            )
            analysis.assumption_log = assumption_log or []
            test_hierarchy_state = _sanitize_for_json(_state_value(result_state, 'test_hierarchy', None))
            test_hierarchy = test_hierarchy_state or _build_test_hierarchy(
                analysis.decision_steps or execution_trace, assumption_log, reviewer_report
            )
            _persist_intermediate_log()
            summary = None
            if reviewer_report and isinstance(reviewer_report, dict):
                summary = reviewer_report.get('adjusted_summary') or reviewer_report.get('summary')
            if not summary:
                summary = messages[-1].content if messages else 'Analysis completed without a summary.'
            full_output = '\n\n'.join(msg.content for msg in messages)
            viz_payload = VisualizationService.generate_visualizations(
                df,
                selected_columns=analysis.selected_columns,
                limit=6,
            )
            plots = viz_payload.get('plots', [])
            effect_sizes = viz_payload.get('effect_sizes', {})
            workflow_graph_state = AnalysisService.build_workflow_graph_state(analysis.decision_steps, test_hierarchy)
            workflow_graph_payload = {
                **workflow_graph_state,
                'assets': render_workflow_graph(workflow_graph_state),
            }

            results_data = {
                'analysis_id': analysis_id,
                'dataset_id': analysis.dataset_id,
                'selected_columns': analysis.selected_columns,
                'version': analysis.version,
                'comment': analysis.comment,
                'full_output': full_output,
                'messages': [msg.content for msg in messages],
                'probabilities': probabilities,
                'summary': summary,
                'execution_trace': execution_trace,
                'decision_steps': analysis.decision_steps,
                'intermediate_log': analysis.intermediate_log,
                'assumption_log': assumption_log,
                'test_hierarchy': test_hierarchy,
                'reviewer_report': reviewer_report,
                'plots': plots,
                'effect_sizes': effect_sizes,
                'workflow_graph': workflow_graph_payload,
                'timestamp': datetime.utcnow().isoformat(),
            }

            results_data = _sanitize_for_json(results_data)
            analysis.result_path = str(StorageService.save_results(analysis_id, results_data))
            analysis.summary = summary[:1000]
            analysis.probabilities = probabilities
            AnalysisService._update_analysis_status(db, analysis, status=AnalysisStatus.COMPLETED)

            logger.info('=' * 80)
            logger.info('✅ ANALYSIS COMPLETED: %s', analysis_id)
            if analysis.start_time:
                duration = (datetime.utcnow() - analysis.start_time).total_seconds()
                logger.info('   Duration: %.2fs', duration)
            logger.info('   Tests performed: %d', len(probabilities))
            logger.info('=' * 80)

        except QuotaExceededError as e:
            logger.error('Quota exceeded for analysis %s: %s', analysis_id, e)
            AnalysisService._update_analysis_status(db, analysis, status=AnalysisStatus.FAILED, error_message=str(e))
            log_stream.write(f'\n\nQuota exceeded: {e}')
            raise
        except Exception as e:
            logger.error('Analysis failed: %s - %s', analysis_id, e, exc_info=True)
            AnalysisService._update_analysis_status(db, analysis, status=AnalysisStatus.FAILED, error_message=str(e))
            log_stream.write(f'\n\nERROR: {e}')
            # We re-raise the exception to ensure the caller knows the operation failed.
            raise
        finally:
            log_content = log_stream.getvalue()
            analysis.intermediate_log = log_content
            analysis.log_path = str(StorageService.save_log(analysis_id, log_content))
            db.commit()
            workflow_logger.removeHandler(log_handler)
            workflow_logger.removeHandler(file_handler)
            log_handler.close()
            file_handler.close()

        return analysis

    @staticmethod
    def get_analysis_results(db: Session, analysis_id: str, *, user_id: str | None = None) -> dict[str, Any] | None:
        """Get detailed analysis results.

        Args:
            db: The database session.
            analysis_id: The analysis UUID.

        Returns:
            A dictionary with the results or None if not found.
        """
        analysis = AnalysisService.get_analysis(db, analysis_id, user_id=user_id)
        if not analysis or analysis.status != AnalysisStatus.COMPLETED:
            return None

        results_data = _sanitize_for_json(StorageService.read_results(analysis_id))
        if not results_data:
            return None

        dataset = DatasetService.get_dataset(db, str(analysis.dataset_id))
        duration = None
        if analysis.start_time and analysis.end_time:
            duration = (analysis.end_time - analysis.start_time).total_seconds()

        workflow_graph_state = results_data.get('workflow_graph') or AnalysisService.build_workflow_graph_state(
            analysis.decision_steps, results_data.get('test_hierarchy')
        )
        if isinstance(workflow_graph_state, dict) and 'assets' not in workflow_graph_state:
            workflow_graph_state = {**workflow_graph_state, 'assets': render_workflow_graph(workflow_graph_state)}

        return {
            'id': analysis.id,
            'status': analysis.status.value,
            'dataset_id': analysis.dataset_id,
            'dataset_name': dataset.original_filename if dataset else 'Unknown',
            'user_id': analysis.user_id,
            'model_name': analysis.model_name,
            'provider': analysis.provider,
            'start_time': analysis.start_time,
            'end_time': analysis.end_time,
            'duration_seconds': duration,
            'summary': analysis.summary,
            'comment': analysis.comment,
            'probabilities': _clean_probabilities(_sanitize_for_json(analysis.probabilities)),
            'results_detail': results_data,
            'execution_trace': results_data.get('execution_trace'),
            'decision_steps': analysis.decision_steps or results_data.get('decision_steps'),
            'intermediate_log': analysis.intermediate_log,
            'assumption_log': analysis.assumption_log or results_data.get('assumption_log'),
            'test_hierarchy': results_data.get('test_hierarchy'),
            'reviewer_report': results_data.get('reviewer_report'),
            'plots': results_data.get('plots'),
            'effect_sizes': results_data.get('effect_sizes'),
            'log_available': bool(analysis.log_path),
            'version': analysis.version,
            'superseded_at': analysis.superseded_at,
            'workflow_graph': workflow_graph_state,
        }

    @staticmethod
    def get_analysis_log(db: Session, analysis_id: str, *, user_id: str | None = None) -> str | None:
        """Get the analysis execution log.

        Args:
            db: The database session.
            analysis_id: The analysis UUID.

        Returns:
            The log content as a string or None if not found.
        """
        analysis = AnalysisService.get_analysis(db, analysis_id, user_id=user_id)
        if not analysis or not analysis.log_path:
            return None

        return StorageService.read_log(analysis_id)

    @staticmethod
    def delete_analysis(db: Session, analysis_id: str, *, user_id: str | None = None) -> bool:
        """Delete an analysis record and its artifacts."""
        analysis = AnalysisService.get_analysis(db, analysis_id, user_id=user_id)
        if not analysis:
            return False

        StorageService.delete_results(analysis_id)
        StorageService.delete_log(analysis_id)
        db.delete(analysis)
        db.commit()
        return True

    @staticmethod
    def update_comment(
        db: Session, analysis_id: str, *, comment: str | None, user_id: str | None = None
    ) -> Analysis | None:
        """Update the comment for an analysis."""
        analysis = AnalysisService.get_analysis(db, analysis_id, user_id=user_id)
        if not analysis:
            return None
        analysis.comment = comment or ''
        # Clean any legacy NaN/Inf fields to avoid JSON serialization errors on return.
        analysis = AnalysisService.sanitize_analysis(analysis)
        db.commit()
        db.refresh(analysis)
        return AnalysisService.sanitize_analysis(analysis)
