"""Role-informed starting experiment, with measured rather than semantic bands.

Role defaults are declared preferences, not model facts. Reuse the signed
effective-update policy without changing its threshold to manufacture changes.
"""
from copy import deepcopy
import math

from gentle_balance_policy import PolicyError, _finite, propose_gentle_balance

POLICY_VERSION = 'role_measured_start_v1'
ROLE_RULES = {
    'character': (2, 'Preserve the intended identity initially'),
    'clothing': (2, 'Preserve the requested garment or footwear initially'),
    'pose': (2, 'Preserve the requested pose initially'),
    'style': (0, 'Allow the general look to yield to specific requested effects'),
    'lighting': (0, 'Allow lighting to yield to specific requested effects'),
    'environment': (1, 'Keep scene context at normal priority'),
    'utility': (1, 'Keep the helper at normal priority until its purpose is known'),
    'other': (1, 'Unknown intent needs user judgement'),
}
ROLE_ALIASES = {'person': 'character', 'people': 'character', 'action': 'pose',
                'coat': 'clothing', 'jacket': 'clothing', 'shoes': 'clothing',
                'footwear': 'clothing'}


def role_rule(role):
    if not isinstance(role, str) or not role.strip():
        raise PolicyError('Every saved profile needs an explicit role')
    normalized = role.strip().lower()
    normalized = ROLE_ALIASES.get(normalized, normalized)
    priority, reason = ROLE_RULES.get(normalized, ROLE_RULES['other'])
    return {'role': role, 'normalized_role': normalized if normalized in ROLE_RULES else 'other',
            'default_priority': priority, 'reason': reason}


def propose_role_start(metrics, entries, overrides=None):
    """Return independent multipliers and graph-ready measured guidance.

Inputs come from revalidated source receipts and immutable saved profiles.
No normalization below is fed back into editable multipliers or loader output.
"""
    if (not isinstance(entries, list) or not entries
            or any(not isinstance(e, dict) or not isinstance(e.get('stable_id'), str)
                   or not e['stable_id'].strip() for e in entries)):
        raise PolicyError('Ordered saved profile entries are required')
    overrides = {} if overrides is None else overrides
    ids = {e.get('stable_id') for e in entries if isinstance(e.get('stable_id'), str)}
    if (not isinstance(overrides, dict) or set(overrides) - ids
            or any(type(p) is not int or p not in (0, 1, 2) for p in overrides.values())):
        raise PolicyError('Priority overrides must name selected LoRAs and use 0, 1 or 2')
    inputs, rules = deepcopy(entries), []
    for entry in inputs:
        rule = role_rule(entry.get('role'))
        entry['priority'] = overrides.get(entry['stable_id'], rule['default_priority'])
        rules.append({**rule, 'stable_id': entry['stable_id'], 'priority': entry['priority'],
                      'priority_basis': 'owner_override' if entry['stable_id'] in overrides else 'role_starting_rule'})
    preview = propose_gentle_balance(metrics, inputs)
    preview['policy_version'] = POLICY_VERSION
    preview['role_rules'] = rules
    changed = {(c['stable_id'], c['slot_index']): c for c in preview['changes']}
    graph = []
    for index, (entry, result) in enumerate(zip(inputs, preview['entries'])):
        norms = [math.sqrt(value) for value in metrics['sources'][index]['block_squared_norms']]
        peak = max(norms)
        before = [_finite(abs(float(entry['strength_model']) * float(v)) * n)
                  for v, n in zip(entry['values'], norms)]
        after = [_finite(abs(float(entry['strength_model']) * float(v)) * n)
                 for v, n in zip(result['values'], norms)]
        guide = []
        for slot, (label, norm, old, value) in enumerate(zip(metrics['slot_labels'], norms, entry['values'], result['values'])):
            change = changed.get((entry['stable_id'], slot))
            if norm == 0:
                state, reason = 'inactive', 'No measured update in this block; a multiplier cannot add missing content'
            elif slot == 0 or entry['priority'] == 2:
                state, reason = 'protected', 'BASE or the chosen Protect priority leaves this block unchanged'
            elif change:
                state, reason = 'suggested', change['reason']
            else:
                state, reason = 'review', 'No automatic adjustment qualifies; measured size is not a semantic safety score'
            guide.append({'slot_index': slot, 'slot_label': label, 'state': state, 'reason': reason,
                          'original_norm': norm, 'relative_original_norm': norm / peak if peak else 0,
                          'before_multiplier': old, 'proposed_multiplier': value,
                          'before_norm': before[slot], 'proposed_norm': after[slot],
                          'trial_interval': {'min': change['min'], 'max': change['max'], 'basis': change['basis']} if change else None})
        graph.append({'stable_id': entry['stable_id'], 'profile_version_id': entry['profile_version_id'],
                      'original_norms': norms, 'before_norms': before, 'proposed_norms': after,
                      'fixed_plot_peak': max(norms + before + after), 'guidance': guide})
    preview['contribution_graphs'] = graph
    preview['limitations'] += [
        'Role defaults are editable starting preferences, not trained or calibrated block semantics.',
        'Identity and clothing can both be protected; an unresolved clash may correctly produce no changes.',
        'Original update norms, current multipliers and adjusted norms are different quantities; graph scaling is display-only.',
        'No averaged profile, rank-based face map, global normalization or role energy quota is used.',
    ]
    return preview
