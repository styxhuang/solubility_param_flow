"""Tests for HSP calculator."""

import pytest
from solubility_param_flow.core.hsp_calculator import HSPCalculator, HSPParameters


class TestHSPCalculator:
    """Test HSP calculator."""
    
    def test_calculate_from_smiles(self):
        """Test HSP calculation from SMILES."""
        calc = HSPCalculator()
        hsp = calc.calculate_from_smiles("CCO", "ethanol")
        
        assert hsp.delta_d > 0
        assert hsp.delta_p > 0
        assert hsp.delta_h > 0
        assert hsp.delta_total > 0
    
    def test_hsp_distance(self):
        """Test HSP distance calculation."""
        hsp1 = HSPParameters(delta_d=15.0, delta_p=5.0, delta_h=10.0)
        hsp2 = HSPParameters(delta_d=16.0, delta_p=6.0, delta_h=11.0)
        
        distance = hsp1.distance_to(hsp2)
        assert distance > 0
    
    def test_predict_solubility(self):
        """Test solubility prediction."""
        calc = HSPCalculator()
        solute = HSPParameters(delta_d=15.0, delta_p=5.0, delta_h=10.0)
        solvent = HSPParameters(delta_d=15.0, delta_p=5.0, delta_h=10.0)
        
        solubility = calc.predict_solubility(solute, solvent)
        assert 0 <= solubility <= 1
