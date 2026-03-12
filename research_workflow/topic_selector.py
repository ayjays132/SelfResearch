from __future__ import annotations

"""Topic selection and validation utility using centralized LLM resources and RAG Skills."""

import logging
from typing import Optional, Any, List
import re

import torch
from models.model_wrapper import LanguageModelWrapper, DEFAULT_GENERATOR, DEFAULT_SENTIMENT
from skills.skill_manager import SkillManager

log = logging.getLogger(__name__)

class TopicSelector:
    """
    A robust and advanced class to select and validate research topics and questions.
    It leverages the LanguageModelWrapper, EMOTIONVERSE-2, and SkillManager.
    """

    def __init__(self, device: Optional[str] = None):
        log.info("Initializing TopicSelector substrate...")
        self.device_str = device
        self.generator_model_name = DEFAULT_GENERATOR
        self.validator_model_name = DEFAULT_GENERATOR
        self.sentiment_model_name = DEFAULT_SENTIMENT

        try:
            self.generator = LanguageModelWrapper(model_name=self.generator_model_name, device=self.device_str)
            self.validator = LanguageModelWrapper(model_name=self.validator_model_name, device=self.device_str)
            self.sentiment_analyzer = LanguageModelWrapper(model_name=self.sentiment_model_name, device=self.device_str)
            self.skill_manager = SkillManager(device=self.device_str)
            log.info("TopicSelector models and SkillManager calibrated.")
        except Exception as e:
            log.critical(f"Failed to initialize TopicSelector: {e}")
            raise RuntimeError(f"TopicSelector initialization failed: {e}")

    def suggest_topic(self, research_area: str, num_suggestions: int = 1, **kwargs: Any) -> str | List[str]:
        log.info(f"Requesting {num_suggestions} research topic(s) for area: '{research_area}'...")
        
        # Dynamically retrieve related skills via RAG
        skills_context = self.skill_manager.retrieve_relevant_skills(f"Generate research topics about {research_area}")
        
        prompt = (f"{skills_context}"
                  f"Generate a highly specific, novel, and academically suitable research topic "
                  f"in the field of {research_area}. Avoid common knowledge topics. "
                  f"Focus on emerging trends or interdisciplinary approaches. "
                  f"The topic should be concise and impactful. Do not include any explanations, preambles, or additional text. Just output the topic title itself.\n"
                  f"Topic:")
        
        try:
            generated_texts = self.generator.generate(
                prompt,
                num_return_sequences=num_suggestions,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                **kwargs
            )

            if isinstance(generated_texts, str):
                generated_texts = [generated_texts]

            suggested_topics = []
            for text in generated_texts:
                # Remove the prompt if the model repeated it
                topic_raw = text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
                
                # Take just the first line (since we asked for only the title)
                cleaned_topic = topic_raw.split('\n')[0].strip()
                
                # Strip markdown bolding if present
                cleaned_topic = cleaned_topic.strip('*')
                
                # Clean up prefixes
                cleaned_topic = re.sub(r"^(Here is|Suggested Topic:|Topic:|Title:)\s*", "", cleaned_topic, flags=re.IGNORECASE).strip()
                if cleaned_topic.endswith('.'):
                    cleaned_topic = cleaned_topic[:-1]
                
                suggested_topics.append(cleaned_topic)
                log.info(f"  Generated Topic: {cleaned_topic}")

            if num_suggestions == 1:
                return suggested_topics[0] if suggested_topics else ""
            return suggested_topics if suggested_topics else [""] * num_suggestions

        except Exception as e:
            log.error(f"Failed to suggest topic for '{research_area}': {e}")
            raise RuntimeError(f"Topic suggestion failed: {e}")

    def suggest_radical_topic(self, research_area: str, **kwargs: Any) -> str:
        log.info(f"Requesting RADICAL hypothesis for area: '{research_area}'...")
        
        prompt = (
            f"You are an eccentric, visionary scientist capable of paradigm-shifting leaps (like PhillVision). "
            f"Do NOT just combine two existing fields. Do NOT just apply existing method X to dataset Y. "
            f"Invent a genuinely novel, radically unintuitive, and deeply surprising research hypothesis in the field of {research_area} "
            f"that distinguishes causal mechanisms from mere correlation. "
            f"Output ONLY the hypothesis title, no extra text.\n"
            f"Radical Hypothesis:"
        )
        
        try:
            generated_text = self.generator.generate(
                prompt,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.95,  # Higher temp for more radical ideas
                top_p=0.95,
                **kwargs
            )
            
            cleaned_topic = generated_text.strip().split('\n')[0].strip('* ')
            cleaned_topic = re.sub(r"^(Here is|Suggested Topic:|Topic:|Title:|Radical Hypothesis:)\s*", "", cleaned_topic, flags=re.IGNORECASE).strip()
            if cleaned_topic.endswith('.'):
                cleaned_topic = cleaned_topic[:-1]
                
            log.info(f"  Generated Radical Topic: {cleaned_topic}")
            return cleaned_topic
            
        except Exception as e:
            log.error(f"Failed to suggest radical topic for '{research_area}': {e}")
            return self.suggest_topic(research_area) # fallback

    def validate_question(self, question: str, **kwargs: Any) -> bool:
        log.info(f"Validating research question: '{question}'...")

        skills_context = self.skill_manager.retrieve_relevant_skills(f"Evaluate and validate academic research question: {question}")

        validation_prompt = (
            f"{skills_context}"
            f"Evaluate the following research question for clarity, focus, feasibility, and academic suitability. "
            f"Provide a brief feedback and then state 'VALID: YES' or 'VALID: NO'.\n\n"
            f"Research Question: \"{question}\"\n\n"
            f"Evaluation:"
        )

        try:
            validation_response = self.validator.generate(
                validation_prompt,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.1,
                num_return_sequences=1,
                **kwargs
            )

            is_valid_llm = False
            feedback = "No specific feedback from LLM."
            
            feedback_match = re.search(r"Evaluation:\s*(.*?)(VALID: (?:YES|NO))", validation_response, re.DOTALL)
            if feedback_match:
                feedback = feedback_match.group(1).strip()
                if feedback.endswith('.'):
                    feedback = feedback[:-1]

            valid_tag_match = re.search(r"VALID: (YES|NO)", validation_response)
            if valid_tag_match:
                if valid_tag_match.group(1) == 'YES':
                    is_valid_llm = True
            
            log.info(f"LLM Feedback: {feedback}")
            log.info(f"LLM Valid Status: {is_valid_llm}")

            # EMOTIONVERSE-2 based Validation
            emotions = self.sentiment_analyzer.get_emotions(question)
            suitability_score = emotions.get('trust', 0) + emotions.get('joy', 0)
            is_emotionally_suitable = suitability_score > 0.1 
            
            log.info(f"EMOTIONVERSE-2 Results: {emotions}")
            log.info(f"Suitability Score (Trust+Joy): {suitability_score:.2f}")

            final_is_valid = is_valid_llm and is_emotionally_suitable and ('?' in question and len(question.split()) > 5)

            if final_is_valid:
                log.info(f"Question '{question}' is VALID.")
            else:
                log.info(f"Question '{question}' is INVALID.")

            return final_is_valid

        except Exception as e:
            log.error(f"Failed to validate question '{question}': {e}")
            raise RuntimeError(f"Question validation failed: {e}")

if __name__ == "__main__":
    try:
        selector = TopicSelector()
        area = "renewable energy"
        topic = selector.suggest_topic(area)
        print(topic)
    except Exception as e:
        print(f"Error: {e}")
