import pandas as pd
import numpy as np
import random
import warnings

from vbree.providers.base import BaseProvider
from vbree.prompts.vbree_mcq import build_prompt
from vbree.utils.parse import extract_json, clamp_score, validate_letter

RESULT_COLS = [
    "id", "iteration", "model",
    "question", "choices", "domain",
    "previous_answer", "updated_answer", "selected_choice",
    "score", "score_moving_avg", "score_moving_variance",
    "chosen_response", "confidence_score"
]


class Ensemble:


    def __init__(self, providers: dict[str, BaseProvider], verbose: bool = False):

        self.variance_threshold = 5.0
        self.providers = providers
        self.verbose = verbose
        self.variance_scaling_factor = 1.1
        self.variance_confidence_coefficient = 1.0
        self.mean_confidence_coefficient = 1.0
        self.n_confidence_coefficient = 1.0
        self.results = pd.DataFrame(columns=RESULT_COLS)
        self.models: list[str] = []

    def add_model(self, model_name: str):
        if model_name not in self.providers:
            raise ValueError(f"Model '{model_name}' not found in providers dictionary.")
        self.models.append(model_name)

    def get_response(self, model_name: str, prompt: str, **kwargs) -> str:
        if model_name not in self.providers:
            raise ValueError(f"Model '{model_name}' not found in providers dictionary.")
        provider = self.providers[model_name]
        response_format = self.build_response_format()
        raw =  provider.generate(prompt, response_format=response_format, **kwargs)
        
        parsed = extract_json(raw, verbose=self.verbose)
        parsed["score"] = clamp_score(parsed["score"])
        parsed["letter"] = str(parsed["letter"]).strip().upper()
        parsed["response"] = str(parsed["response"]).strip()
        
        return parsed
    
    def _calculate_confidence_score(self, mean: float, variance: float, n: int) -> float:
        pv = variance * self.variance_confidence_coefficient
        pm = mean * self.mean_confidence_coefficient
        pn = n * self.n_confidence_coefficient
        # small constant avoids division by 0
        return pm / ((pv * np.sqrt(pn)) + ((100 - pm) * self.mean_confidence_coefficient) + 1e-6)
    
    def _scale_variance(self, threshold: float) -> float:
        return threshold ** self.variance_scaling_factor
    
    def build_response_format(self):
        return {
            "type": "json_schema",        
            "json_schema": {
                "name": "ResponseExtraction",
                "strict": True,          
                "schema": {
                    "type": "object",    
                    "properties": {
                        "score":    {"type": "number"},   
                        "response": {"type": "string"},  
                        "letter":   {"type": "string"}    
                    },
                    "required": ["score", "response", "letter"],  
                    "additionalProperties": False  
                }
            }
        }


    def run (self,
            data: pd.DataFrame,
            choices_col: str = "choices",
            id_col: str = "id",
            question_col: str = "question",
            domain_col: str = "domain",
            model_algorithm: str = "random_start",
            iter_max: int=99,
            verbose: bool = False,
            **kwargs
            ):
        # validate choices column type
        if not data[choices_col].apply(lambda x: isinstance(x, list)).all():
            raise ValueError(f"All entries in '{choices_col}' must be lists of choice texts.")
    
        if len(self.models) == 0:
            raise ValueError("Add at least one model using add_model().")
    
        N = len(self.models)
    
        for row in data.itertuples():
            qid = getattr(row, id_col)
            question = getattr(row, question_col)
            choices = getattr(row, choices_col)
            domain = getattr(row, domain_col)

            # choose starting model index
            if model_algorithm == "order_added":
                model_index = 0
            elif model_algorithm == "random_start":
                model_index = random.randint(0, N - 1) if N > 1 else 0
            else:
                raise ValueError("model_algorithm must be 'order_added' or 'random_start'")
            
            local_threshold = self.variance_threshold
            last_variance = float("inf")

            if self.verbose:
                print(f"\nProcessing question ID: {qid}")


            last_variance = float("inf")
            iter_index = 0
            prev_answer = ""
            row_results = []
            best_window_variance = float("inf")
            best_window_end_iter = None

            while iter_index < iter_max and(iter_index < N or last_variance >= local_threshold):
                model_name= self.models[model_index]
                prompt = build_prompt(question = question, 
                                      previous_answer= prev_answer, 
                                      choices = choices)
                
                out = self.get_response(model_name, prompt, **kwargs)

                score = out["score"]
                updated_answer = out["response"]
                letter = validate_letter(out["letter"], n_choices=len(choices))
               
                record = {
                    "id": qid,
                    "iteration": iter_index,
                    "model": model_name,
                    "question": question,
                    "choices": choices,
                    "domain": domain,
                    "previous_answer": prev_answer,
                    "updated_answer": updated_answer,
                    "selected_choice": letter,
                    "score": score,
                    "score_moving_avg": None,
                    "score_moving_variance": None,
                    "chosen_response": False,
                    "confidence_score": None,
                }

                row_results.append(record)
                   
                prev_answer = updated_answer

                if iter_index >= (N-1):
                    window_scores = [r["score"] for r in row_results[-N:]]
                    moving_avg = float(np.mean(window_scores))
                    moving_var = float(np.var(window_scores))

                    last_variance = moving_var
                    row_results[-1]["score_moving_avg"] = moving_avg
                    row_results[-1]["score_moving_variance"] = moving_var
                    row_results[-1]["confidence_score"] = self._calculate_confidence_score(moving_avg, moving_var, len(row_results))

                    # historical recovery: track lowest-variance window
                    if moving_var <= best_window_variance:
                        best_window_variance = moving_var
                        best_window_end_iter = iter_index
                    
                # scale threshold every full cycle through models (same spirit as your friend)
                if (iter_index + 1) % N == 0 and (iter_index + 1) > N:
                    local_threshold = self._scale_variance(local_threshold)

                # move to next model
                if N > 1:
                    model_index = (model_index + 1) % N
                else:
                    warnings.warn("Single-model mode: no real collaboration.")

                iter_index += 1

                 # choose final answer: last answer of lowest-variance window
            if best_window_end_iter is None:
                chosen_idx = len(row_results) - 1
            else:
                chosen_idx = best_window_end_iter

            row_results[chosen_idx]["chosen_response"] = True

            self.results = pd.concat([self.results, pd.DataFrame(row_results)], ignore_index=True)

        return self.results
                

            

