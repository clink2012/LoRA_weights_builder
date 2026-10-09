"""Server-owned experimental proposals and atomic saved composition history."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import gentle_balance_policy
import role_start_policy

from fastapi import APIRouter, Body, HTTPException

from analysis_job_service import JobError
from composition_versions import PreparationChangedError
from computed_baselines import resolve_baseline
from experiment_versions import experiment_plan_digest, get_experiment, save_experiment
from gentle_balance_policy import POLICY_VERSION as GENTLE_VERSION, PolicyError, propose_gentle_balance
from role_start_policy import POLICY_VERSION as ROLE_VERSION, propose_role_start
from profile_versions import ProfileNotFoundError, ProfileValidationError, get_version

POLICY_SOURCES = {module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                  for module in (gentle_balance_policy, role_start_policy)}


def build_experiment_plan(conn, service, *, job_id, priorities, expected_preparation_digest, policy_kind='gentle', force_recompute=False, baseline_metadata=None):
    if policy_kind not in ('gentle', 'role_start'):
        raise PolicyError('Choose a supported starting policy')
    if POLICY_SOURCES != {module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                          for module in (gentle_balance_policy, role_start_policy)}:
        raise PreparationChangedError('The calculation source changed; restart the app before computing or reusing values.')
    validated = service.revalidate(job_id)
    job, prepared = validated['job'], validated['preparation']
    if prepared['preparation_digest'] != expected_preparation_digest:
        raise PreparationChangedError('The displayed selection changed. Prepare and analyse it again.')
    nodes = prepared['node_payloads']
    if (not isinstance(priorities, dict)
            or (set(priorities) != {node['stable_id'] for node in nodes} if policy_kind == 'gentle'
                else bool(set(priorities) - {node['stable_id'] for node in nodes}))
            or any(type(priority) is not int or priority not in (0, 1, 2) for priority in priorities.values())):
        raise ProfileValidationError('Choose Flexible, Normal or Protect for every selected LoRA.')
    parents = [get_version(conn, node['stable_id'], node['profile_version_id']) for node in nodes]
    for node, parent in zip(nodes, parents):
        if (parent['values'] != node['loader_export']['architecture_slot_values']
                or parent['binding'] != node['profile_default_binding']):
            raise PreparationChangedError('The saved selection no longer matches its fresh preparation.')
    policy_entries = [{'stable_id': parent['stable_id'], 'profile_version_id': parent['version_id'],
                       'values': parent['values'], 'strength_model': parent['settings']['strength_model'],
                       'role': parent['settings']['role'],
                       'priority': priorities.get(parent['stable_id'], 1)} for parent in parents]
    metrics = job['metrics']
    # The measurement timestamp/job ID is not a calculation input. Source,
    # runtime/worker pins, ordered saved values, role choices and all numerical
    # measurements are inputs and must invalidate reuse when changed.
    context = {'policy_version': ROLE_VERSION if policy_kind == 'role_start' else GENTLE_VERSION,
               'policy_sources': POLICY_SOURCES,
               'worker_binding': job.get('worker_binding'),
               'target_contract_id': prepared['target_contract_id'], 'entries': policy_entries,
               'bindings': [parent['binding'] for parent in parents], 'overrides': priorities,
               'metrics': {key: value for key, value in metrics.items() if key not in ('created_at', 'work')}}
    baseline = resolve_baseline(conn, context, lambda: (
        propose_role_start(metrics, policy_entries, priorities) if policy_kind == 'role_start'
        else propose_gentle_balance(metrics, policy_entries)), force=force_recompute)
    preview = baseline['preview']
    if baseline_metadata is not None:
        baseline_metadata.update({key: baseline[key] for key in ('baseline_id', 'context_key', 'created_at', 'reused')})
    plan = {'job_id': job_id, 'engine_version': metrics['engine_version'],
            'policy_version': preview['policy_version'], 'target_contract_id': prepared['target_contract_id'],
            'input_preparation_digest': expected_preparation_digest, 'policy_preview': preview,
            'metrics_receipt': metrics,
            'metrics_receipt_sha256': hashlib.sha256(json.dumps(metrics, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
            'entries': []}
    for parent, proposal in zip(parents, preview['entries']):
        changed = parent['values'] != proposal['values']
        plan['entries'].append({'stable_id': parent['stable_id'], 'parent_version_id': parent['version_id'],
                                'values': proposal['values'], 'settings': parent['settings'],
                                'ab': {} if changed else parent['ab'], 'priority': proposal['priority']})
    plan['proposal_digest'] = experiment_plan_digest(plan)
    return plan


def create_experiment_router(connection_factory, analysis_service, preparation_resolver):
    router = APIRouter(prefix='/api/experiments', tags=['experimental balance'])

    def fields(body, required, optional=()):
        if not isinstance(body, dict) or set(body) - set(required) - set(optional) or not set(required).issubset(body):
            raise HTTPException(422, 'Missing or unexpected experiment fields. Saved values and provenance are calculated by the server.')

    def run(operation):
        try:
            with closing(connection_factory()) as conn:
                return operation(conn)
        except JobError as exc:
            raise HTTPException(exc.status, {'reason_code': exc.code, 'reason': str(exc)}) from exc
        except PreparationChangedError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ProfileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except (ProfileValidationError, PolicyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post('/preview')
    def preview(body: dict = Body(...)):
        fields(body, ('job_id', 'priorities', 'expected_preparation_digest'), ('policy_kind', 'force_recompute'))
        def operation(conn):
            metadata = {}
            plan = build_experiment_plan(conn, analysis_service, baseline_metadata=metadata, **body)
            return {'job_id': plan['job_id'], 'proposal_digest': plan['proposal_digest'],
                    'input_preparation_digest': plan['input_preparation_digest'],
                    'policy_preview': plan['policy_preview'], 'computed_baseline': metadata,
                    'can_save': bool(plan['policy_preview']['changes']),
                    'ab_handling': 'Changed variants use the displayed per-slot policy trial ranges. Earlier A/B settings remain in their parent versions; they are not copied into these new variants.'}
        return run(operation)

    @router.post('/save')
    def save(body: dict = Body(...)):
        fields(body, ('job_id', 'priorities', 'expected_preparation_digest', 'expected_proposal_digest', 'name', 'idempotency_key'), ('parent_version_id', 'policy_kind'))
        def operation(conn):
            def resolve(connection, job_id):
                return build_experiment_plan(connection, analysis_service, job_id=job_id,
                                             priorities=body['priorities'], expected_preparation_digest=body['expected_preparation_digest'],
                                             policy_kind=body.get('policy_kind', 'gentle'))
            return save_experiment(conn, job_id=body['job_id'], expected_proposal_digest=body['expected_proposal_digest'],
                                   name=body['name'], idempotency_key=body['idempotency_key'],
                                   parent_version_id=body.get('parent_version_id'), experiment_resolver=resolve,
                                   preparation_resolver=preparation_resolver)
        return run(operation)

    @router.get('/{experiment_id}')
    def get(experiment_id: str):
        return run(lambda conn: get_experiment(conn, experiment_id))

    return router
