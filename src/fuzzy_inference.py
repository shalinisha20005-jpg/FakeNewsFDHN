
"""
Fuzzy inference component for the Fake News Detection System.

Inputs:
    confidence: Highest probability from the LSTM (0 to 1)
    uncertainty: Normalized entropy of the class probabilities (0 to 1)

Output:
    reliability: Fuzzy decision-support score (0 to 100)

This score is NOT the probability that a news statement is true.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# --------------------------------------------------
# 1. Define fuzzy variables
# --------------------------------------------------
confidence = ctrl.Antecedent(
    np.arange(0, 1.01, 0.01), "confidence"
)

uncertainty = ctrl.Antecedent(
    np.arange(0, 1.01, 0.01), "uncertainty"
)

reliability = ctrl.Consequent(
    np.arange(0, 101, 1), "reliability"
)

# Use centroid defuzzification for the output.
reliability.defuzzify_method = "centroid"


# --------------------------------------------------
# 2. Define membership functions
# --------------------------------------------------
confidence["low"] = fuzz.trapmf(
    confidence.universe, [0.0, 0.0, 0.25, 0.45]
)
confidence["medium"] = fuzz.trimf(
    confidence.universe, [0.25, 0.50, 0.75]
)
confidence["high"] = fuzz.trapmf(
    confidence.universe, [0.55, 0.75, 1.0, 1.0]
)

uncertainty["low"] = fuzz.trapmf(
    uncertainty.universe, [0.0, 0.0, 0.20, 0.40]
)
uncertainty["medium"] = fuzz.trimf(
    uncertainty.universe, [0.20, 0.50, 0.80]
)
uncertainty["high"] = fuzz.trapmf(
    uncertainty.universe, [0.60, 0.80, 1.0, 1.0]
)

reliability["low"] = fuzz.trapmf(
    reliability.universe, [0, 0, 20, 40]
)
reliability["medium"] = fuzz.trimf(
    reliability.universe, [25, 50, 75]
)
reliability["high"] = fuzz.trapmf(
    reliability.universe, [60, 80, 100, 100]
)


# --------------------------------------------------
# 3. Define fuzzy inference rules
# --------------------------------------------------
rules = [
    ctrl.Rule(
        confidence["high"] & uncertainty["low"],
        reliability["high"]
    ),
    ctrl.Rule(
        confidence["high"] & uncertainty["medium"],
        reliability["medium"]
    ),
    ctrl.Rule(
        confidence["high"] & uncertainty["high"],
        reliability["low"]
    ),
    ctrl.Rule(
        confidence["medium"] & uncertainty["low"],
        reliability["medium"]
    ),
    ctrl.Rule(
        confidence["medium"] & uncertainty["medium"],
        reliability["medium"]
    ),
    ctrl.Rule(
        confidence["medium"] & uncertainty["high"],
        reliability["low"]
    ),
    ctrl.Rule(
        confidence["low"],
        reliability["low"]
    ),
]

fuzzy_system = ctrl.ControlSystem(rules)


# --------------------------------------------------
# 4. Calculate uncertainty using normalized entropy
# --------------------------------------------------
def calculate_uncertainty(probabilities):
    """
    Return normalized entropy of a probability distribution.

    0 means highly concentrated predictions.
    1 means a uniform distribution over the classes.
    """
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 1 or len(probabilities) < 2:
        raise ValueError("Provide a 1D array with at least two classes.")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must be finite.")

    if np.any(probabilities < 0):
        raise ValueError("Probabilities cannot be negative.")

    total = probabilities.sum()

    if not np.isclose(total, 1.0, atol=1e-5):
        raise ValueError("Probabilities must sum to 1.")

    positive = probabilities[probabilities > 0]
    entropy = -np.sum(positive * np.log(positive))
    maximum_entropy = np.log(len(probabilities))

    return float(entropy / maximum_entropy)


# --------------------------------------------------
# 5. Apply fuzzy inference
# --------------------------------------------------
def evaluate_prediction(probabilities):
    """
    Evaluate model confidence and uncertainty.

    Returns:
        predicted_class_index
        confidence
        uncertainty
        fuzzy_score
        review recommendation
    """
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 1 or len(probabilities) < 2:
        raise ValueError("Provide probabilities for at least two classes.")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must be finite.")

    if np.any(probabilities < 0):
        raise ValueError("Probabilities cannot be negative.")

    if not np.isclose(probabilities.sum(), 1.0, atol=1e-5):
        raise ValueError("Probabilities must sum to 1.")

    predicted_index = int(np.argmax(probabilities))
    max_confidence = float(probabilities[predicted_index])
    uncertainty_value = calculate_uncertainty(probabilities)

    simulator = ctrl.ControlSystemSimulation(fuzzy_system)

    simulator.input["confidence"] = max_confidence
    simulator.input["uncertainty"] = uncertainty_value
    simulator.compute()

    score = float(simulator.output["reliability"])

    # Initial conservative rule for review.
    # Thresholds must be evaluated on validation data.
    needs_review = (
        max_confidence < 0.50 or uncertainty_value > 0.75
    )

    return {
        "predicted_class_index": predicted_index,
        "confidence": max_confidence,
        "uncertainty": uncertainty_value,
        "fuzzy_score": score,
        "needs_human_review": needs_review,
    }


# --------------------------------------------------
# 6. Run a simple demonstration
# --------------------------------------------------
if __name__ == "__main__":
    # Example probabilities for six classes.
    # These are demonstration values, NOT model predictions.
    example_probabilities = [
        0.10, 0.35, 0.20, 0.15, 0.05, 0.15
    ]

    result = evaluate_prediction(example_probabilities)

    print("\nFuzzy inference demonstration")
    print("-" * 35)
    print("Predicted class index:", result["predicted_class_index"])
    print(f"Model confidence: {result['confidence']:.2%}")
    print(f"Normalized uncertainty: {result['uncertainty']:.3f}")
    print(f"Fuzzy decision-support score: {result['fuzzy_score']:.2f}/100")
    print(
        "Recommendation:",
        "Needs Human Review"
        if result["needs_human_review"]
        else "No review flag"
    )
