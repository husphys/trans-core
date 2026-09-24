from __future__ import annotations

import ast, hashlib, json
from pathlib import Path

import torch, yaml

from src.mepi_v1.finetune_v1_4 import StrictEarlyStopping
from src.mepi_v1.finetune_v1_5 import (
    MAX_EPOCHS, PARENT_STATE_SHA256, PROTOCOL_SHA256, RESUME_NEXT_EPOCH,
    audit_exact_parent_state, audit_v1_5_frozen_evidence, validate_v1_5_config,
)
from src.mepi_v1.loss_weight_sensitivity_v1_5 import build_grid_plan, grid_pairs, load_sensitivity_config

ROOT=Path(__file__).resolve().parents[1]
V14_RUN=ROOT/'experiments/finetune_v1_4_xlstm_depth8/baseline_l3_0p3_l4_0p20'
V15_RUN=ROOT/'experiments/finetune_v1_5_xlstm_depth8/baseline_l3_0p3_l4_0p20'
V14_HASHES={
 'MEPI-FROZEN-PROTOCOL v1.4.md':'685f79b6b700ae131ff2a19af524417ead46ee9c073f1c2ddf9d96191f59b487',
 'docs/MEPI_FROZEN_PROTOCOL_v1.4.md':'685f79b6b700ae131ff2a19af524417ead46ee9c073f1c2ddf9d96191f59b487',
 'configs/finetune_v1_4.yaml':'cc26c0f3ccf6857e4b4f2fd87829f09192a89f1e3f1f46adcae2d6a92830a305',
 'src/mepi_v1/finetune_v1_4.py':'5e4fd8177c571d89c7fc687926e6a5f2864a2dc8a3362b6cc4c0e3fad514e254',
 'notebooks/32_xlstm_v1_4_finetune.ipynb':'51e9ec68d74e1567361c7f4b34b62359f097c91b27b80504c229818458a168da',
 'reports/MEPI_V1_4_BASELINE_VALIDATION_ANALYSIS.md':'570bbe6f25616c4c13bbccb555f457f732b9d1adc10737a971438099e690f1ee',
 'reports/MEPI_V1_4_BASELINE_VALIDATION_METRICS.json':'0c78748cd507b59770da1e814bc3959b2b919cdafaabfaeba08793d81ed135b3',
 'experiments/finetune_v1_4_xlstm_depth8/baseline_l3_0p3_l4_0p20/best_checkpoint.pt':'22aca18c37b4526c46e3044eeeda4e36a6a0364891e767cace755961eef44138',
 'experiments/finetune_v1_4_xlstm_depth8/baseline_l3_0p3_l4_0p20/last_checkpoint.pt':'22aca18c37b4526c46e3044eeeda4e36a6a0364891e767cace755961eef44138',
 'experiments/finetune_v1_4_xlstm_depth8/baseline_l3_0p3_l4_0p20/history.json':'c3f33c9ccb2526efda37b26d4429e26fa3075b8ffa3f306a38edd0884f62a90b',
 'experiments/finetune_v1_4_xlstm_depth8/baseline_l3_0p3_l4_0p20/completed.json':'0b029376b0d152b301f3a55941a350065a6d6d500bed2d55d6ea9d8fc99aac52',
}
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def notebook_source(name):
 nb=json.loads((ROOT/'notebooks'/name).read_text());return nb,'\n'.join(''.join(c['source']) for c in nb['cells'])

def test_v1_4_protected_evidence_is_byte_unchanged():
 included = {relative: expected for relative, expected in V14_HASHES.items() if (ROOT / relative).is_file()}
 assert included
 for relative,expected in included.items(): assert sha(ROOT/relative)==expected
 data_notes=(ROOT/'DATA.md').read_text()
 for omitted in (
  'notebooks/32_xlstm_v1_4_finetune.ipynb',
  'experiments/finetune_v1_4_xlstm_depth8/',
 ):
  assert omitted in data_notes

def test_v1_5_protocol_is_byte_identical_and_only_epoch_budget_changes():
 a=ROOT/'MEPI-FROZEN-PROTOCOL v1.5.md';b=ROOT/'docs/MEPI_FROZEN_PROTOCOL_v1.5.md'
 assert a.read_bytes()==b.read_bytes();assert sha(a)==PROTOCOL_SHA256
 assert 'maximum downstream training epochs: 50 -> 100' in a.read_text()
 c4=yaml.safe_load((ROOT/'configs/finetune_v1_4.yaml').read_text());c5=yaml.safe_load((ROOT/'configs/finetune_v1_5.yaml').read_text())
 assert c5['training']['max_epochs']==100 and c5['training']['early_stopping_patience']==10
 assert c5['training']['scheduler'] is None and c5['training']['min_delta']==0.0 and c5['training']['strict_improvement'] is True
 normalized=dict(c5);normalized['protocol_version']=c4['protocol_version'];normalized['protocol_sha256']=c4['protocol_sha256'];normalized['training']=dict(c5['training']);normalized['training']['max_epochs']=50;normalized['training']['command']=c4['training']['command'];normalized['resume']=c4['resume']
 assert normalized==c4

def test_v1_5_runtime_contract_and_strict_patience():
 c=load_config=__import__('src.mepi_v1.config',fromlist=['load_config']).load_config(ROOT/'configs/finetune_v1_5.yaml');validate_v1_5_config(c)
 assert MAX_EPOCHS==100;assert audit_v1_5_frozen_evidence(ROOT,ROOT/'configs/finetune_v1_5.yaml')['scheduler'] is None
 stop=StrictEarlyStopping(patience=10,min_delta=0.0);assert stop.update(1.0,0)==(True,False)
 for i in range(1,10):assert stop.update(1.0,i)==(False,False)
 assert stop.update(1.0,10)==(False,True)
 source=(ROOT/'src/mepi_v1/finetune_v1_5.py').read_text();assert 'scheduler.step(' not in source;assert 'weights-only or altered parent continuation is forbidden' in source

def test_exact_parent_resume_provenance_is_preserved_after_completion():
 provenance=json.loads((V15_RUN/'continuation_provenance.json').read_text())
 assert provenance['parent_checkpoint_sha256']==PARENT_STATE_SHA256
 assert provenance['imported_last_checkpoint_sha256']==PARENT_STATE_SHA256
 assert provenance['optimizer_state_preserved'] and provenance['amp_scaler_state_preserved'] and provenance['rng_state_preserved']
 completed=json.loads((V15_RUN/'completed.json').read_text())
 assert completed['status']=='TRAINING_COMPLETE_VALIDATION_ONLY' and completed['resume_source']=='v1.4_exact_parent_state'
 assert completed['resume_next_epoch']==50 and completed['test_accessed'] is False
 assert not (V14_RUN/'last_checkpoint.pt').exists()
 assert 'experiments/finetune_v1_4_xlstm_depth8/' in (ROOT/'DATA.md').read_text()

def test_notebook_32_defaults_safe_and_has_exact_resume_provenance():
 nb,source=notebook_source('32_xlstm_v1_5_finetune.ipynb')
 assert 'RUN_TRAINING = False' in source and 'FORCE_RETRAIN = False' in source
 for required in ('PROTOCOL_V1_5_SHA256','PARENT_V1_4_STATE_SHA256','RESUME_NEXT_EPOCH','MAX_EPOCHS','EARLY_STOPPING_PATIENCE','TEST_ACCESSED = FALSE'): assert required in source
 assert 'FrozenV14Dataset(CONFIG_PATH, "test")' not in source and 'build_test' not in source
 completed=json.loads((V15_RUN/'completed.json').read_text());assert completed['test_accessed'] is False

def test_sensitivity_grid_has_12_common_initializations_and_frozen_rule():
 cfg=load_sensitivity_config(ROOT/'configs/loss_weight_sensitivity_v1_5.yaml');plan=build_grid_plan(ROOT/'configs/loss_weight_sensitivity_v1_5.yaml')
 assert len(grid_pairs())==len(plan)==12;assert len({(x['lambda3'],x['lambda4']) for x in plan})==12
 assert len({x['initialization_checkpoint_sha256'] for x in plan})==1
 assert all(x['seed']==42 and x['max_epochs']==100 and x['patience']==10 and x['scheduler'] is None and x['test_access'] is False for x in plan)
 assert cfg['selection_rule_ready'] is True and cfg['automatic_winner_declaration'] is True
 assert cfg['selection_rule']['primary_metric']=='VAL_TASK_SCORE'
 assert cfg['selection_rule']['formula']=='validation_L_electrical + validation_L_LSP'
 assert cfg['selection_rule']['weighted_validation_total_loss_cross_configuration_ranking'] is False
 assert sum(x['requires_new_run'] for x in plan)==11 and sum(x['reuse_completed_baseline'] for x in plan)==1
 nb,source=notebook_source('33_xlstm_v1_5_loss_weight_sensitivity.ipynb');assert 'RUN_GRID = True' in source and 'LOSS_WEIGHT_SELECTION_RULE_READY = TRUE' in source
 code=[c for c in nb['cells'] if c['cell_type']=='code']
 assert all(c.get('execution_count') is not None for c in code) and not [o for c in code for o in c.get('outputs',[]) if o.get('output_type')=='error']

def test_notebook_34_records_completed_exactly_once_final_test():
 nb,source=notebook_source('34_xlstm_v1_5_final_training_and_test.ipynb')
 assert 'FINAL_CONFIG_FROZEN = True' in source and 'RUN_FINAL_TEST = True' in source
 assert "audit_final_preparation(PROJECT_ROOT, FINAL_CONFIG_PATH)" in source
 assert "run_final_test(PROJECT_ROOT, FINAL_CONFIG_PATH, authorize=True)" in source
 assert "preparation_audit['training_performed'] is False" in source
 assert "preparation_audit['test_dataset_created'] is False" in source
 assert "preparation_audit['test_loader_created'] is False" in source
 assert 'FINAL_TEST_READY = TRUE' in source and 'TEST_ACCESSED = FALSE' in source
 code=[c for c in nb['cells'] if c['cell_type']=='code']
 assert [c.get('execution_count') for c in code]==[1,2,3]
 assert not [o for c in code for o in c.get('outputs',[]) if o.get('output_type')=='error']
 output='\n'.join(''.join(o.get('text',[])) for c in code for o in c.get('outputs',[]) if o.get('output_type')=='stream')
 for required in ('FINAL_CONFIG_FROZEN = TRUE','RUN_FINAL_TEST = TRUE','FINAL_TEST_COMPLETE = TRUE','TEST_ACCESSED = TRUE','TEST_EVALUATION_COUNT = 1'):
  assert required in output

def test_notebook_35_requires_final_model_and_36_marks_missing():
 _,s35=notebook_source('35_xlstm_v1_5_frequency_screening.ipynb');_,s36=notebook_source('36_xlstm_v1_5_tables_and_figures.ipynb')
 assert 'RUN_FREQUENCY_SCREENING and not (FINAL_MODEL_READY and FREQUENCY_SCREENING_INPUT_READY)' in s35
 assert 'run_frequency_screening(PROJECT_ROOT, authorize=True)' in s35
 assert 'RUN_FREQUENCY_SCREENING = False' in s35
 assert 'load_json_or_missing' in s36 and '"status": "MISSING"' in s36 and 'final_test_metrics' in s36

def test_experiment_manifest_locks_test_and_records_dependencies():
 m=json.loads((ROOT/'experiments/MEPI_V1_5_EXPERIMENT_MANIFEST.json').read_text())
 assert m['test_lock']['status']=='LOCKED' and m['test_lock']['expected_test_rows']==90 and m['test_lock']['exactly_once'] is True
 assert m['edges']==[['32_xlstm_v1_5_finetune','33_xlstm_v1_5_loss_weight_sensitivity'],['33_xlstm_v1_5_loss_weight_sensitivity','34_xlstm_v1_5_final_training_and_test'],['34_xlstm_v1_5_final_training_and_test','35_xlstm_v1_5_frequency_screening']]
 assert m['scientific_training_run'] is False and m['test_accessed'] is False

def test_new_sources_and_notebooks_parse_without_execution():
 for path in [ROOT/'src/mepi_v1/finetune_v1_5.py',ROOT/'src/mepi_v1/loss_weight_sensitivity_v1_5.py']:
  ast.parse(path.read_text())
 for name in ['32_xlstm_v1_5_finetune.ipynb','33_xlstm_v1_5_loss_weight_sensitivity.ipynb','34_xlstm_v1_5_final_training_and_test.ipynb','35_xlstm_v1_5_frequency_screening.ipynb','36_xlstm_v1_5_tables_and_figures.ipynb']:
  nb=json.loads((ROOT/'notebooks'/name).read_text())
  for cell in nb['cells']:
   if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
