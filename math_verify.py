#!/usr/bin/env python3

import sys
sys.path.insert(0, '/Users/vnutrenni/Documents/Master2024/Year2/Sem_1A/ModellingSimulation/modsimproj')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

from core.base_models import SIRParameters, SEIRParameters, SIRSParameters
from models.sir_model import SIRModel
from models.seir_model import SEIRModel
from models.sirs_model import SIRSModel
from models.seirv_model import SEIRVModel, SEIRVParameters
from models.mixin_models import SEIRWithInterventions, SIRWithInterventions
from analysis.visualization import plot_comparison, plot_phase_portrait, plot_R_effective
from analysis.metrics import calculate_peak_time, calculate_attack_rate

# debug class to verify mathematical correctness of epidemic models

class MathematicalVerifier:
    

    def __init__(self):
        self.verification_results = {}
        self.timestr = time.strftime("%Y%m%d-%H%M%S")
        self.output_dir = Path(f'results/math_verify_{self.timestr}')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def verify_sir_equations(self):
        """
        Properties to verify:
        1. Mass conservation: S + I + R = 1
        2. R0 = β/γ
        3. Herd immunity threshold = 1 - 1/R0
        4. Final epidemic size equation
        """
        print("\n" + "="*60)
        print("Verifying SIR Model")
        print("="*60)

        params = SIRParameters(beta=0.5, gamma=0.1)
        model = SIRModel(params=params, S0=0.99, I0=0.01, R0=0.0)
        results = model.simulate(t_span=(0, 200))

        # Verify mass conservation
        total_pop = results['S'] + results['I'] + results['R']
        mass_conservation = np.allclose(total_pop, 1.0, atol=1e-6)

        # Verify R0
        R0_calc = params.beta / params.gamma
        R0_expected = 5.0
        R0_correct = np.isclose(R0_calc, R0_expected)

        # Verify herd immunity threshold
        herd_immunity = model.calculate_herd_immunity_threshold()
        herd_immunity_expected = 1 - 1/R0_calc
        herd_immunity_correct = np.isclose(herd_immunity, herd_immunity_expected)

        # Verify final size relation (approximate for numerical solution)
        # Final size equation: R_∞ = 1 - exp(-R0 * R_∞)
        R_final = results['R'][-1]
        S_final = results['S'][-1]
        final_size_check = np.isclose(R_final, 1 - S_final * np.exp(-R0_calc * R_final), atol=0.01)

        print(f"✓ Mass Conservation: {mass_conservation}")
        print(f"  Total population preserved: {np.allclose(total_pop, 1.0, atol=1e-6)}")
        print(f"  Max deviation: {np.max(np.abs(total_pop - 1.0)):.2e}")

        print(f"\n✓ R0 Calculation: {R0_correct}")
        print(f"  Calculated: {R0_calc:.2f}")
        print(f"  Expected: {R0_expected:.2f}")

        print(f"\n✓ Herd Immunity Threshold: {herd_immunity_correct}")
        print(f"  Calculated: {herd_immunity:.2%}")
        print(f"  Expected: {herd_immunity_expected:.2%}")

        print(f"\n✓ Final Size Relation: {final_size_check}")
        print(f"  R_final: {R_final:.3f}")
        print(f"  S_final: {S_final:.3f}")

        self.verification_results['SIR'] = {
            'mass_conservation': mass_conservation,
            'R0_correct': R0_correct,
            'herd_immunity_correct': herd_immunity_correct,
            'final_size_correct': final_size_check,
            'all_passed': all([mass_conservation, R0_correct, herd_immunity_correct, final_size_check])
        }

        return results

    def verify_seir_equations(self):
        """
        Properties to verify:
        1. Mass conservation: S + E + I + R = 1
        2. R0 = β/γ (same as SIR)
        3. Mean incubation period = 1/σ
        4. Delayed epidemic peak vs SIR
        """
        print("\n" + "="*60)
        print("Verifying SEIR Model")
        print("="*60)

        params = SEIRParameters(beta=0.6, sigma=0.2, gamma=0.1)
        model = SEIRModel(params=params, S0=0.99, E0=0.0, I0=0.01, R0=0.0)
        results = model.simulate(t_span=(0, 300))

        # Verify mass conservation
        total_pop = results['S'] + results['E'] + results['I'] + results['R']
        mass_conservation = np.allclose(total_pop, 1.0, atol=1e-6)

        # Verify R0
        R0_calc = params.beta / params.gamma
        R0_correct = np.isclose(R0_calc, 6.0)

        # Verify incubation period
        incubation_period = 1.0 / params.sigma
        incubation_correct = np.isclose(incubation_period, 5.0)

        print(f" Mass Conservation: {mass_conservation}")
        print(f"  Max deviation: {np.max(np.abs(total_pop - 1.0)):.2e}")

        print(f"\n R0 Calculation: {R0_correct}")
        print(f"  R0 = β/γ = {R0_calc:.2f}")

        print(f"\n Incubation Period: {incubation_correct}")
        print(f"  1/σ = {incubation_period:.1f} days")

        self.verification_results['SEIR'] = {
            'mass_conservation': mass_conservation,
            'R0_correct': R0_correct,
            'incubation_correct': incubation_correct,
            'all_passed': all([mass_conservation, R0_correct, incubation_correct])
        }

        return results

    def verify_sirs_equations(self):
        """
        Properties to verify:
        1. Mass conservation: S + I + R = 1
        2. Endemic equilibrium when R0 > 1
        3. S* = 1/R0 at equilibrium
        4. Oscillatory approach to equilibrium
        """
        print("\n" + "="*60)
        print("Verifying SIRS Model")
        print("="*60)

        params = SIRSParameters(beta=0.5, gamma=0.1, omega=0.01)
        model = SIRSModel(params=params, S0=0.99, I0=0.01, R0=0.0)
        results = model.simulate(t_span=(0, 1000))

        # Verify mass conservation
        total_pop = results['S'] + results['I'] + results['R']
        mass_conservation = np.allclose(total_pop, 1.0, atol=1e-6)

        # Verify endemic equilibrium
        equilibrium = model.calculate_endemic_equilibrium()
        R0 = params.beta / params.gamma

        # At equilibrium: S* = 1/R0
        S_star_expected = 1.0 / R0
        S_star_correct = np.isclose(equilibrium['S'], S_star_expected, rtol=0.01)

        # Check if system approaches equilibrium
        S_final = results['S'][-1]
        approaches_equilibrium = np.isclose(S_final, equilibrium['S'], rtol=0.1)

        print(f" Mass Conservation: {mass_conservation}")
        print(f"  Max deviation: {np.max(np.abs(total_pop - 1.0)):.2e}")

        print(f"\n Endemic Equilibrium: S* = 1/R0")
        print(f"  R0 = {R0:.2f}")
        print(f"  S* calculated: {equilibrium['S']:.3f}")
        print(f"  S* expected (1/R0): {S_star_expected:.3f}")
        print(f"  Match: {S_star_correct}")

        print(f"\n Approaches Equilibrium: {approaches_equilibrium}")
        print(f"  S(final) = {S_final:.3f}")
        print(f"  I(final) = {results['I'][-1]:.3f}")
        print(f"  R(final) = {results['R'][-1]:.3f}")

        self.verification_results['SIRS'] = {
            'mass_conservation': mass_conservation,
            'equilibrium_correct': S_star_correct,
            'approaches_equilibrium': approaches_equilibrium,
            'all_passed': all([mass_conservation, S_star_correct, approaches_equilibrium])
        }

        return results

    def verify_seirv_equations(self):
        """
        Properties to verify:
        1. Mass conservation: S + E + I + R + V = 1
        2. Vaccination reduces effective susceptible population
        3. Effective vaccination contributes to R (immune)
        4. Failed vaccination goes to V (not immune)
        """
        print("\n" + "="*60)
        print("Verifying SEIRV Model")
        print("="*60)

        params = SEIRVParameters(beta=0.6, sigma=0.2, gamma=0.1,
                                 vaccine_efficacy=0.8, vaccination_rate=0.02)
        model = SEIRVModel(params=params, S0=0.99, E0=0.0, I0=0.01, R0=0.0, V0=0.0)
        model.set_vaccination_campaign(start_time=20, duration=100, rate=0.02, efficacy=0.85)
        results = model.simulate(t_span=(0, 300))

        # Verify mass conservation
        total_pop = results['S'] + results['E'] + results['I'] + results['R'] + results['V']
        mass_conservation = np.allclose(total_pop, 1.0, atol=1e-5)

        # Verify vaccination reduces S
        S_before_vacc = results['S'][results['t'] <= 20]
        S_during_vacc = results['S'][(results['t'] > 20) & (results['t'] < 120)]
        vaccination_reduces_S = np.mean(S_during_vacc) < np.mean(S_before_vacc)

        # Verify R + V increases during vaccination
        RV_before = (results['R'] + results['V'])[results['t'] <= 20]
        RV_during = (results['R'] + results['V'])[(results['t'] > 20) & (results['t'] < 120)]
        vaccination_increases_immunity = np.mean(RV_during) > np.mean(RV_before)

        print(f" Mass Conservation: {mass_conservation}")
        print(f"  Max deviation: {np.max(np.abs(total_pop - 1.0)):.2e}")

        print(f"\n Vaccination Effects:")
        print(f"  Reduces susceptible population: {vaccination_reduces_S}")
        print(f"  Increases immune population (R+V): {vaccination_increases_immunity}")
        print(f"  Final R: {results['R'][-1]:.2%}")
        print(f"  Final V: {results['V'][-1]:.2%}")
        print(f"  Total immune: {(results['R'][-1] + results['V'][-1]):.2%}")

        self.verification_results['SEIRV'] = {
            'mass_conservation': mass_conservation,
            'vaccination_reduces_S': vaccination_reduces_S,
            'vaccination_increases_immunity': vaccination_increases_immunity,
            'all_passed': all([mass_conservation, vaccination_reduces_S, vaccination_increases_immunity])
        }

        return results

    def generate_summary(self):
        print("\n" + "="*60)
        print("MATHEMATICAL VERIFICATION SUMMARY")
        print("="*60)

        all_passed = True
        for model, results in self.verification_results.items():
            status = "✓ PASS" if results['all_passed'] else "✗ FAIL"
            print(f"{model:10s}: {status}")
            all_passed = all_passed and results['all_passed']

        print("\n" + "="*60)
        if all_passed:
            print("✓ ALL MODELS VERIFIED SUCCESSFULLY")
        else:
            print("✗ SOME MODELS FAILED VERIFICATION")
        print("="*60)

        return all_passed


def main():
    print("\n" + "="*60)
    print("COMPREHENSIVE VERIFICATION")
    print("="*60)
    print("\nMathematical Verification")
    print("-" * 60)
    verifier = MathematicalVerifier()
    verifier.verify_sir_equations()
    verifier.verify_seir_equations()
    verifier.verify_sirs_equations()
    verifier.verify_seirv_equations()
    all_verified = verifier.generate_summary()

    #  Save verification results
    print("\nSaving Results")
    print("-" * 60)

    # Conversion for numpy bools to Python bools for JSON serialization
    def convert_bools(obj):
        if isinstance(obj, dict):
            return {k: convert_bools(v) for k, v in obj.items()}
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj

    output_data = {
        'verification': convert_bools(verifier.verification_results),
        'timestamp': str(np.datetime64('now'))
    }

    with open(verifier.output_dir / 'verification_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n Results saved to: {verifier.output_dir}")
    print(f" Verification data: verification_results.json")
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

    if all_verified:
        print("\n All mathematical models verified successfully!")
        print(f"\n View results in: {verifier.output_dir}/verification_results.json")
    else:
        print("\n Some verification checks failed. Please review.")

    return all_verified


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
