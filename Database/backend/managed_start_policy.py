"""Declared role priors followed by measured overlap checks, not semantic maps."""
from copy import deepcopy
import math

from gentle_balance_policy import _finite, propose_gentle_balance
from role_start_policy import role_rule

POLICY_VERSION = 'managed_role_start_v1'
# Initial preferences, derived from the project's advisory role strengths.
# Pose is kept nearer full strength; unknown intent receives no guessed cut.
ROLE_STARTS = {'character': (.90, 2), 'clothing': (.65, 2), 'pose': (.75, 2),
               'style': (.60, 0), 'lighting': (.60, 0), 'environment': (.45, 1),
               'utility': (.35, 1), 'other': (1.0, 1)}


def propose_managed_start(metrics, entries):
    inputs, rules = deepcopy(entries), []
    for entry in inputs:
        rule = role_rule(entry.get('role'))
        factor, priority = ROLE_STARTS[rule['normalized_role']]
        entry['priority'] = priority
        rules.append({**rule, 'stable_id': entry['stable_id'], 'priority': priority,
                      'starting_factor': factor, 'priority_basis': 'declared_role_start',
                      'reason': 'Experimental role starting level; not a measured semantic block assignment'})
    # Validate the full original signed Gram matrix and input contract first.
    original = propose_gentle_balance(metrics, inputs)
    for entry, rule, source in zip(inputs, rules, metrics['sources']):
        entry['values'] = [value if slot == 0 or source['block_squared_norms'][slot] == 0
                           else _finite(value * rule['starting_factor'])
                           for slot, value in enumerate(entry['values'])]
    preview = propose_gentle_balance(metrics, inputs)
    preview.update(policy_version=POLICY_VERSION, role_rules=rules, changes=[])
    preview['constants']['role_starting_factors'] = {role: spec[0] for role, spec in ROLE_STARTS.items()}
    # Different role cuts can weaken cancellation. Keep the original block if
    # the full signed stack energy would rise. This is a numerical guard only.
    for slot, (before, after) in enumerate(zip(original['blocks'], preview['blocks'])):
        tolerance = 1e-10 * max(before['energy_before'], after['energy_after'], 1e-20)
        reverted = after['energy_after'] > before['energy_before'] + tolerance
        after.update(energy_before=before['energy_before'], starting_energy=after['energy_before'],
                     reverted=after['reverted'] or reverted)
        if reverted:
            after.update(energy_after=before['energy_before'], reason='Role starting levels would increase signed stack energy; original block retained')
            for result, entry in zip(preview['entries'], entries):
                result['values'][slot] = entry['values'][slot]
    graphs = []
    for index, (entry, result, rule) in enumerate(zip(entries, preview['entries'], rules)):
        result['before_values'] = deepcopy(entry['values'])
        norms = [math.sqrt(value) for value in metrics['sources'][index]['block_squared_norms']]
        before_norms = [_finite(abs(entry['strength_model'] * v) * n) for v, n in zip(entry['values'], norms)]
        after_norms = [_finite(abs(entry['strength_model'] * v) * n) for v, n in zip(result['values'], norms)]
        guidance = []
        for slot, (old, value, norm, label) in enumerate(zip(entry['values'], result['values'], norms, metrics['slot_labels'])):
            changed = value != old
            overlap_factor = value / (old * rule['starting_factor']) if changed and old and rule['starting_factor'] else 1
            reason = (f"Declared role start x{rule['starting_factor']}; measured overlap adjustment x{overlap_factor:.6g}"
                      if changed else preview['blocks'][slot]['reason'] or 'No additional starting adjustment')
            interval = {'min': min(value, old), 'max': max(value, old),
                        'basis': 'Experimental starting interval, not a proven safe range'} if changed else None
            if changed:
                preview['changes'].append({'stable_id': entry['stable_id'], 'slot_index': slot, 'slot_label': label,
                                           'before': old, 'value': value, **interval, 'reason': reason})
            guidance.append({'slot_index': slot, 'slot_label': label,
                             'state': 'inactive' if norm == 0 else 'suggested' if changed else 'protected' if slot == 0 else 'review',
                             'reason': 'No measured update; changing the multiplier cannot add missing content' if norm == 0 else reason,
                             'original_norm': norm, 'relative_original_norm': norm / max(norms) if max(norms) else 0,
                             'before_multiplier': old, 'proposed_multiplier': value,
                             'before_norm': before_norms[slot], 'proposed_norm': after_norms[slot], 'trial_interval': interval})
        graphs.append({'stable_id': entry['stable_id'], 'profile_version_id': entry['profile_version_id'],
                       'original_norms': norms, 'before_norms': before_norms, 'proposed_norms': after_norms,
                       'fixed_plot_peak': max(norms + before_norms + after_norms), 'guidance': guidance})
    preview['contribution_graphs'] = graphs
    preview['limitations'] = [
        'Role starting factors are declared experimental preferences, not learned or image-tested recommendations.',
        'Each LoRA retains its own vector; actual update norms and signed inner products determine overlap adjustments.',
        'BASE, zeros and unmeasured blocks are retained. Unknown roles receive no guessed role reduction.',
        'No block is assumed to control a face, coat or shoe. No normalized norm or averaged profile becomes a multiplier.',
        'Signed stack energy is a numerical guard, not a guarantee of structural or visual quality.',
    ]
    return preview
