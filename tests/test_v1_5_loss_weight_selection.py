from __future__ import annotations

import hashlib, json, math
from pathlib import Path

import pytest

from src.mepi_v1.loss_weight_sensitivity_v1_5 import (
    PRIMARY_SELECTION_METRIC, build_grid_plan, compute_val_task_score,
    rank_completed_configurations,
)

ROOT=Path(__file__).resolve().parents[1]
RULE=ROOT/'MEPI-V1.5-LOSS-WEIGHT-SELECTION-RULE.md'
RULE_HASH='8c4eda9035e7e5e4f75a4b0ccde643c8946fd3b990575ec0323977cb8497a5be'

def test_selection_rule_is_frozen_and_byte_identical():
 docs=ROOT/'docs/MEPI_V1_5_LOSS_WEIGHT_SELECTION_RULE.md'
 assert RULE.read_bytes()==docs.read_bytes()
 assert hashlib.sha256(RULE.read_bytes()).hexdigest()==RULE_HASH
 text=RULE.read_text()
 assert 'VAL_TASK_SCORE = validation_L_electrical + validation_L_LSP' in text
 assert 'weighted_validation_total_loss' in text and 'prohibited' in text

def test_primary_score_is_sum_and_rejects_nonfinite_inputs():
 assert PRIMARY_SELECTION_METRIC=='VAL_TASK_SCORE'
 assert compute_val_task_score(0.2,0.3)==pytest.approx(0.5)
 with pytest.raises(ValueError):compute_val_task_score(float('nan'),0.3)
 with pytest.raises(ValueError):compute_val_task_score(0.2,float('inf'))

def test_ranking_uses_task_score_and_excludes_numerical_failures():
 rows=[
  {'configuration_id':'weighted-total-would-win','VAL_TASK_SCORE':0.4,'weighted_validation_total_loss':0.1,'numerical_failure':False,'eligible_for_selection':True},
  {'configuration_id':'task-score-wins','VAL_TASK_SCORE':0.2,'weighted_validation_total_loss':9.0,'numerical_failure':False,'eligible_for_selection':True},
  {'configuration_id':'numerical-failure','VAL_TASK_SCORE':0.01,'weighted_validation_total_loss':0.01,'numerical_failure':True,'eligible_for_selection':False},
 ]
 result=rank_completed_configurations(rows)
 assert result['winner_declared'] is True
 assert result['selected_configuration_id']=='task-score-wins'

def test_exact_task_score_tie_does_not_invent_tiebreaker():
 rows=[{'configuration_id':name,'VAL_TASK_SCORE':0.2,'weighted_validation_total_loss':weighted,'numerical_failure':False,'eligible_for_selection':True} for name,weighted in [('a',0.1),('b',9.0)]]
 result=rank_completed_configurations(rows)
 assert result['winner_declared'] is False and result['selected_configuration_id'] is None
 assert result['tied_configuration_ids']==['a','b']

def test_baseline_reuse_audit_passes_and_leaves_11_new_runs():
 audit=json.loads((ROOT/'reports/MEPI_V1_5_BASELINE_GRID_REUSE_AUDIT.json').read_text())
 assert audit['status']=='PASS' and audit['baseline_reuse_valid'] is True
 assert all(audit['proof'].values())
 assert audit['scientific_training_run'] is False and audit['test_accessed'] is False
 result=json.loads((ROOT/'experiments/finetune_v1_5_xlstm_depth8/loss_weight_sensitivity/l3_0p3_l4_0p20/completed.json').read_text())
 assert result['reused_completed_v1_5_baseline'] is True
 assert result['numerical_failure'] is False and result['eligible_for_selection'] is True
 for field in ('Efficiency_MAE','Efficiency_RMSE','Efficiency_R2','P_loss_MAE','P_loss_RMSE','P_loss_R2','LSP_raw_MAE','LSP_raw_RMSE','LSP_raw_R2','LSP_raw_MAPE_PERCENT','validation_L_electrical','validation_L_LSP','validation_L_physics','validation_L_UQ','weighted_validation_total_loss','best_epoch','stop_epoch','epochs_completed'): assert field in result
 assert math.isclose(result['VAL_TASK_SCORE'],result['validation_L_electrical']+result['validation_L_LSP'])
 plan=build_grid_plan(ROOT/'configs/loss_weight_sensitivity_v1_5.yaml')
 assert len(plan)==12 and sum(row['requires_new_run'] for row in plan)==11

def test_notebook_33_records_completed_grid_and_keeps_test_locked():
 nb=json.loads((ROOT/'notebooks/33_xlstm_v1_5_loss_weight_sensitivity.ipynb').read_text())
 source='\n'.join(''.join(cell['source']) for cell in nb['cells'])
 assert 'RUN_GRID = True' in source
 assert 'PRIMARY_SELECTION_METRIC = VAL_TASK_SCORE' in source
 assert 'NEW_RUNS_REQUIRED =", NEW_RUNS_REQUIRED' in source
 assert 'TEST_ACCESSED = FALSE' in source
 code=[cell for cell in nb['cells'] if cell['cell_type']=='code']
 assert all(cell.get('execution_count') is not None for cell in code)
 assert not [output for cell in code for output in cell.get('outputs',[]) if output.get('output_type')=='error']
