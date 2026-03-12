from tools.base_tool import BaseTool
import re

class ScientificUnitConverterTool(BaseTool):
    name = "unit_converter"
    description = "Converts between scientific units (e.g., Celsius to Kelvin, Light-years to Parsecs, Joules to Electronvolts). Supports prefixes like kilo, mega, milli, micro."
    parameters = {
        "type": "object",
        "properties": {
            "value": {"type": "number", "description": "The numeric value to convert."},
            "from_unit": {"type": "string", "description": "Current unit (e.g., 'Celsius', 'kpc', 'Joules', 'eV')."},
            "to_unit": {"type": "string", "description": "Target unit (e.g., 'Kelvin', 'pc', 'eV', 'J')."}
        },
        "required": ["value", "from_unit", "to_unit"]
    }

    def execute(self, value: float, from_unit: str, to_unit: str, **kwargs) -> str:
        f = from_unit.lower().strip()
        t = to_unit.lower().strip()
        
        # Prefix handling
        prefixes = {
            'peta': 1e15, 'tera': 1e12, 'giga': 1e9, 'mega': 1e6, 'kilo': 1e3,
            'milli': 1e-3, 'micro': 1e-6, 'nano': 1e-9, 'pico': 1e-12,
            'p': 1e15, 't': 1e12, 'g': 1e9, 'm': 1e6, 'k': 1e3,
            'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12
        }

        def normalize(unit_str):
            # Check for temperature first (no prefixes usually used like 'kiloCelsius')
            if unit_str in ['c', 'celsius']: return 'c', 1.0
            if unit_str in ['k', 'kelvin', 'kelvins']: return 'k', 1.0
            
            # Check energy
            if unit_str in ['j', 'joule', 'joules']: return 'j', 1.0
            if unit_str in ['ev', 'electronvolt', 'electronvolts']: return 'ev', 1.0
            
            # Check distance
            if unit_str in ['ly', 'lightyear', 'light-year', 'lightyears', 'light-years']: return 'ly', 1.0
            if unit_str in ['pc', 'parsec', 'parsecs']: return 'pc', 1.0

            # Try prefix + unit
            for p, mult in prefixes.items():
                if unit_str.startswith(p):
                    base = unit_str[len(p):]
                    if base in ['j', 'joule', 'joules']: return 'j', mult
                    if base in ['ev', 'electronvolt', 'electronvolts']: return 'ev', mult
                    if base in ['pc', 'parsec', 'parsecs']: return 'pc', mult
                    if base in ['ly', 'lightyear', 'lightyears']: return 'ly', mult
            
            return unit_str, 1.0

        f_base, f_mult = normalize(f)
        t_base, t_mult = normalize(t)
        
        val_in_base = value * f_mult

        # Temperature
        if f_base == 'c' and t_base == 'k': return f"{(val_in_base + 273.15) / t_mult} {t.upper()}"
        if f_base == 'k' and t_base == 'c': return f"{(val_in_base - 273.15) / t_mult} {t.upper()}"
        
        # Distance
        if f_base == 'ly' and t_base == 'pc': return f"{(val_in_base / 3.262) / t_mult:.6f} {t.upper()}"
        if f_base == 'pc' and t_base == 'ly': return f"{(val_in_base * 3.262) / t_mult:.6f} {t.upper()}"
        
        # Energy
        if f_base == 'j' and t_base == 'ev': return f"{(val_in_base * 6.242e18) / t_mult:.4e} {t.upper()}"
        if f_base == 'ev' and t_base == 'j': return f"{(val_in_base / 6.242e18) / t_mult:.4e} {t.upper()}"
        
        if f_base == t_base:
            return f"{(val_in_base) / t_mult} {t.upper()}"

        return f"Error: Conversion from '{from_unit}' to '{to_unit}' not supported. Supported: Temp (C, K), Dist (LY, PC), Energy (J, eV) + Prefixes."
