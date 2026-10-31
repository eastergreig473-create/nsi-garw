import importlib.util
from pathlib import Path
import sys
import numpy as np

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts";sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location("association_audit",SCRIPTS/"reanalyze_associations.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_exact_small_sample_permutation_and_interval_withholding():
    result=module.association(np.arange(5),np.arange(5),replicates=1000)
    assert np.isclose(result["spearman_rho"],1,atol=1e-12)
    assert result["permutation_p_value"]==2/120
    assert result["confidence_interval"] is None
    assert result["permutation_replicates"]==120


def test_constant_observation_is_undefined_not_zero():
    result=module.association(np.zeros(5),np.arange(5),replicates=1000)
    assert result["spearman_rho"] is None
    assert result["permutation_p_value"] is None


def test_degenerate_cluster_draws_are_counted():
    result=module.association([0,0,0,0,1],[1,2,3,4,5],replicates=1000)
    assert result["bootstrap_undefined_count"]>200
    assert result["bootstrap_valid_count"]+result["bootstrap_undefined_count"]==1000
    assert result["confidence_interval"] is None
