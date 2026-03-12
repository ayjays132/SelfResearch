import numpy as np
import logging
import json
from typing import List, Dict, Union, Any
from tools.base_tool import BaseTool

log = logging.getLogger(__name__)

class EntropyVerifier(BaseTool):
    """
    Analyzes the convergence of curves (e.g., loss, entropy, error) 
    from numeric data or dictionary-based plot inputs.
    """
    name = "entropy_verifier"
    description = "Analyzes curve convergence (monotonically decreasing, flat, or diverging). Returns structured verdict and confidence score."
    parameters = {
        "type": "object",
        "properties": {
            "data": {"type": "object", "description": "Numeric data to analyze (list of floats or dictionary with list values)."},
            "key": {"type": "string", "description": "If data is a dictionary, which key contains the target metric to analyze."}
        },
        "required": ["data"]
    }

    def execute(self, data: Union[List[float], Dict[str, List[float]]], key: str = None) -> Dict[str, Any]:
        """
        Computes the convergence verdict for the given data.
        """
        try:
            # 1. Normalize data to a list of floats
            if isinstance(data, dict):
                if not key:
                    # Default to the first key found if not specified
                    key = next(iter(data))
                values = data.get(key, [])
            else:
                values = data

            if len(values) < 3:
                return {
                    "verdict": "stalling",
                    "confidence": 0.0,
                    "reason": "Insufficient data points (minimum 3 required)."
                }

            # Ensure numeric
            y = np.array(values, dtype=float)
            x = np.arange(len(y))

            # 2. Linear Regression (Trend)
            slope, intercept = np.polyfit(x, y, 1)
            
            # 3. Monotonic Check
            diffs = np.diff(y)
            is_mono_dec = np.all(diffs <= 1e-6) # Allowing for minor noise
            is_mono_inc = np.all(diffs >= -1e-6)

            # 4. Variance Check (Flatness)
            std_dev = np.std(y)
            mean_val = np.abs(np.mean(y))
            rel_std = std_dev / mean_val if mean_val > 1e-9 else std_dev

            # 5. Logic for Verdict
            # Converging: Strong downward trend (negative slope)
            if slope < -1e-4:
                verdict = "converging"
                confidence = min(0.95, abs(slope) * 10) # Simple scaling
            # Diverging: Strong upward trend
            elif slope > 1e-4:
                verdict = "diverging"
                confidence = min(0.95, slope * 10)
            # Stalling: Little change (slope near zero) or high noise
            else:
                if rel_std < 0.01:
                    verdict = "stalling" # Flat
                else:
                    verdict = "stalling" # Noisy but no clear trend
                confidence = 0.5 + (1.0 - min(1.0, rel_std)) * 0.4

            # Final adjustment for Monotonicity
            if verdict == "converging" and is_mono_dec:
                confidence = min(1.0, confidence + 0.1)
            if verdict == "diverging" and is_mono_inc:
                confidence = min(1.0, confidence + 0.1)

            result = {
                "verdict": verdict,
                "confidence": round(float(confidence), 2),
                "slope": round(float(slope), 6),
                "is_monotonic": bool(is_mono_dec or is_mono_inc),
                "std_dev": round(float(std_dev), 6),
                "reason": f"Slope detected at {slope:.6f} with relative std_dev of {rel_std:.4f}."
            }
            log.info(f"Entropy Verifier Result: {verdict} (conf: {confidence})")
            return result

        except Exception as e:
            log.error(f"Entropy Verifier Error: {e}")
            return {"verdict": "error", "confidence": 0.0, "reason": str(e)}

if __name__ == "__main__":
    ev = EntropyVerifier()
    # Test Converging
    print(ev.execute([10.0, 8.0, 7.0, 6.5, 6.0]))
    # Test Diverging
    print(ev.execute([1.0, 2.0, 4.0, 7.0, 11.0]))
    # Test Flat/Stalling
    print(ev.execute([5.0, 5.01, 4.99, 5.0, 5.0]))
