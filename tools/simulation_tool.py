import logging
import json
from typing import List, Dict, Union, Any
from tools.base_tool import BaseTool
from simulation_lab.experiment_simulator import ExperimentSimulator

log = logging.getLogger(__name__)

class SimulationTool(BaseTool):
    """
    Empowers the research agent to run virtual experiments (physics, biology)
    to empirically validate generated hypotheses.
    """
    
    @property
    def name(self) -> str:
        return "simulation_lab"
        
    @property
    def description(self) -> str:
        return "Runs physical or biological simulations to validate hypotheses empirically. " \
               "Args: [type ('physics' or 'biology'), param1, param2, ...]. " \
               "Physics: ['physics', initial_pos, initial_vel, time_steps, dt]. " \
               "Biology: ['biology', init_pop_A, init_pop_B, growth_A, death_B, interaction, time_steps, dt]."

    def execute(self, args: List[Any], **kwargs) -> Dict[str, Any]:
        if not args or len(args) < 5:
            return {"error": "Insufficient arguments for simulation. See tool description."}
            
        sim_type = args[0]
        device = kwargs.get("device", "cpu")
        
        try:
            simulator = ExperimentSimulator(device=device)
            
            if sim_type == "physics":
                _, p0, v0, steps, dt = args
                results = simulator.run_physics_simulation(float(p0), float(v0), int(steps), float(dt))
                # Just return summary stats to not overwhelm LLM context
                return {
                    "simulation_type": "1D Physics (Gravity)",
                    "initial_state": {"position": p0, "velocity": v0},
                    "final_state": {"position": float(results[-1]), "velocity": float((results[-1]-results[-2])/dt) if steps > 1 else 0},
                    "trajectory_summary": f"Simulated {steps} steps. Min pos: {float(results.min())}, Max pos: {float(results.max())}."
                }
            elif sim_type == "biology":
                if len(args) < 8: return {"error": "Biology simulation requires 8 arguments."}
                _, pA, pB, gA, dB, inter, steps, dt = args
                results = simulator.run_biological_simulation(float(pA), float(pB), float(gA), float(dB), float(inter), int(steps), float(dt))
                return {
                    "simulation_type": "Lotka-Volterra Predator-Prey",
                    "final_population_A": float(results[-1, 0]),
                    "final_population_B": float(results[-1, 1]),
                    "status": "Simulation completed successfully."
                }
            else:
                return {"error": f"Unknown simulation type: {sim_type}"}
                
        except Exception as e:
            log.error(f"Simulation failed: {e}")
            return {"error": str(e)}
