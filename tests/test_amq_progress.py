"""Progress must describe measured calibration scores without changing search."""
import json

from hds_quant.amq_progress import progress_snapshot, write_progress


def test_history_tracks_anchor_wins_refinement_and_overspend(tmp_path):
    archive = [('initial', .2, 3.20), ('over_cap', .001, 3.751)]
    history = []

    def record(stage, new_from, iteration=0):
        history.append(progress_snapshot(archive, [3, 3.5], 128, history,
            stage=stage, iteration=iteration, new_from=new_from, elapsed_sec=iteration+1))

    record('initial_design', 0)
    assert history[-1]['budgets']['3.00']['best_archive_index'] == 0
    archive += [('anchor3', .1, 3.25), ('anchor35', .04, 3.75)]
    record('budget_anchors', 2)
    assert history[-1]['budgets']['3.50']['winner_origin'] == 'budget_anchors'
    archive += [('worse', .5, 2.75)]
    record('refinement', 4, 1)
    assert not history[-1]['budgets']['3.00']['improved_vs_previous']
    assert history[-1]['budgets']['3.00']['winner_origin'] == 'budget_anchors'
    archive += [('better', .03, 3.15)]
    record('refinement', 5, 2)
    assert all(row['improved_vs_previous'] for row in history[-1]['budgets'].values())
    assert history[-1]['budgets']['3.50']['best_archive_index'] == 5
    assert history[-1]['new_candidate_payload_min'] == 2.9
    write_progress(tmp_path, history)
    saved = json.loads((tmp_path/'search_progress.json').read_text())
    assert saved['history'] == history
    import csv
    with (tmp_path/'search_progress.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8
    assert rows[-1]['best_archive_index'] == '5'


def test_empty_feasible_set_and_no_new_candidates_are_explicit():
    row = progress_snapshot([('too_large', .01, 4.25)], [3], 128, [],
        stage='initial_design', iteration=0, new_from=1, elapsed_sec=1)
    assert row['budgets']['3.00']['best_calibration_jsd'] is None
    assert row['budgets']['3.00']['winner_origin'] is None
    assert row['new_candidate_payload_min'] is None
    assert row['new_configurations'] == 0
