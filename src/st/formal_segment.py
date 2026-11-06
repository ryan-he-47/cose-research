"""2026-10-19：Slurm时间分段只接续未耗尽的现有预算。"""
def segment_action(latest, stop, coverage):
    target = latest.get('target_inputs')
    if target is not None:
        complete = latest['successful_inputs'] >= target
        expected_reason = 'input_budget'
    else:
        complete = coverage['progress'] >= 1
        expected_reason = 'data_coverage'
    if complete and stop['reason'] == expected_reason:
        return 'complete'
    if not complete and stop['reason'] in ('max_seconds', 'SIGUSR1'):
        return 'continue'
    return 'stop'
