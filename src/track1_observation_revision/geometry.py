"""Nested geometry budgets with a certificate-preserving feasible incumbent."""
import numpy as np
from track1_belief.geometry import fit_adaptive_geometry, geometry_curves, _source_rings
from track1_lab.ring_topology import predicate


def fit_budget_path(rings, roles, parents, budgets=(16, 24, 32), **kwargs):
    """Optimize within each budget, retaining the best feasible earlier fit.

    Every fit sees exactly the same source. A larger budget permits fewer
    controls; it does not force them. Certificates stay bound to their original
    coefficients. No averaging, padding, or covariance transfer is performed.
    """
    if not budgets or any(not isinstance(b, int) or b <= 0 for b in budgets):
        raise ValueError('Positive integer budgets required')
    if list(budgets) != sorted(set(budgets)):
        raise ValueError('Budgets must be strictly increasing')
    penalty = kwargs.get('rate_penalty', 0.)
    source = _source_rings(rings, roles, parents)
    options = kwargs.get('control_options', (8, 16, 24, 32))
    if not options or any(not isinstance(m, int) or m < 4 or m % 2 for m in options):
        raise ValueError('Control choices must be even integers >=4')
    incumbent = None
    results = []
    for budget in budgets:
        if min(options) * len(source) > budget:
            results.append(dict(supported=False, certified=False, budget=budget,
                                allocation=[], source_completeness=False,
                                carried_incumbent=False, fresh_candidate_supported=False,
                                fresh_candidate_failure='all_source_rings_exceed_control_budget',
                                failure_code='all_source_rings_exceed_control_budget'))
            continue
        candidate = fit_adaptive_geometry(rings, roles, parents,
                                          max_total_controls=budget, **kwargs)
        chosen = candidate
        if candidate['supported']:
            geometry_curves(candidate, samples=16)  # Validate certificate binding.
        def objective(fit):
            return fit['retained_normalized_mse'] + penalty * fit['coefficient_bytes']
        if incumbent is not None and (not candidate['supported'] or
                                      objective(incumbent) <= objective(candidate)):
            chosen = incumbent
        if chosen['supported']:
            incumbent = chosen
        results.append(dict(chosen, budget=budget,
                            carried_incumbent=chosen is not candidate,
                            fresh_candidate_supported=candidate['supported'],
                            fresh_candidate_failure=candidate['failure_code']))
    return results


def split_boundary_vertices(rings, roles, parents):
    """Reject a holdout that changes source topology or has too few targets.

    This is a source interpolation diagnostic, not independent annotation.
    Topology is checked before and after thinning with identical orientation.
    """
    source = _source_rings(rings, roles, parents)
    if min(map(len, source)) < 16:
        raise ValueError('At least four withheld and twelve retained vertices per ring required')
    retained = [p[np.arange(len(p)) % 4 != 0] for p in source]
    targets = [p[np.arange(len(p)) % 4 == 0] for p in source]
    if not all(predicate(p, roles, parents)['ring_topology_valid']
               for p in (source, retained)):
        raise ValueError('Full or retained source topology invalid')
    return retained, targets
