
import torch
from transformers import pipeline

class TopicSelector:
    """
    A class to select and validate research topics and questions.
    This simulates an intelligent suggestion/validation system using a pre-trained LLM.
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initializes the TopicSelector with a specified device and loads a pre-trained LLM.
        Args:
            device (str): The device to use for computations ('cuda' or 'cpu').
        """
        self.device = torch.device(device)
        print(f"TopicSelector initialized on device: {self.device}")

        # Load a small, efficient pre-trained language model for text generation
        # In a real scenario, this would be a more powerful LLM (e.g., GPT-3, Llama)
        self.generator = pipeline("text-generation", model="distilgpt2", device=0 if self.device.type == 'cuda' else -1)

    def suggest_topic(self, research_area: str) -> str:
        """
        Suggests a research topic based on the input research area using a generative LLM.
        Args:
            research_area (str): The broad area of research.
        Returns:
            str: A suggested research topic.
        """
        prompt = f"Generate a highly specific and novel research topic in the field of {research_area}. The topic should be suitable for academic research and avoid common knowledge. Research Topic:"
        
        # Generate text using the pre-trained model
        generated_text = self.generator(
            prompt,
            max_new_tokens=50,
            num_return_sequences=1,
            truncation=True
        )[0]['generated_text']

        # Extract the generated topic by removing the prompt
        suggested_topic = generated_text[len(prompt):].strip()
        if suggested_topic.endswith('.'):
            suggested_topic = suggested_topic[:-1]

        # Simulate some PyTorch computation with the generated text length
        dummy_tensor = torch.tensor([len(suggested_topic)], dtype=torch.float32, device=self.device)
        _ = torch.log(dummy_tensor + 1) # Perform a dummy computation

        return suggested_topic

    def validate_question(self, question: str) -> bool:
        """
        Validates a research question for clarity, focus, and feasibility using a simple NLP model.
        In a real-world scenario, this would involve more sophisticated NLP models or a dedicated LLM call.
        Args:
            question (str): The research question to validate.
        Returns:
            bool: True if the question is considered valid, False otherwise.
        """
        # Using a simple sentiment analysis model as a proxy for question quality/coherence
        # A positive sentiment might indicate a well-formed, clear question.
        validator_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if self.device.type == 'cuda' else -1)
        validation_result = validator_pipeline(question)

        is_valid = False
        if validation_result and validation_result[0]['label'] == 'POSITIVE' and validation_result[0]['score'] > 0.8:
            # Further checks for question marks and length
            if '?' in question and len(question.split()) > 5:
                is_valid = True

        # Simulate some PyTorch computation
        dummy_tensor = torch.tensor([len(question)], dtype=torch.float32, device=self.device)
        _ = torch.sqrt(dummy_tensor) # Perform a dummy computation

        return is_valid

if __name__ == "__main__":
    # Example Usage
    if torch.cuda.is_available():
        device = 'cuda'
        print("CUDA is available! Using GPU.")
    else:
        device = 'cpu'
        print("CUDA not available. Using CPU.")

    selector = TopicSelector(device=device)

    area = "renewable energy"
    suggested_topic = selector.suggest_topic(area)
    print(f"\nResearch Area: {area}")
    print(f"Suggested Topic: {suggested_topic}")

    question1 = "How can solar panel efficiency be improved using novel nanomaterials?"
    is_valid1 = selector.validate_question(question1)
    print(f"Question: '{question1}' is valid: {is_valid1}")

    question2 = "Solar panels good?"
    is_valid2 = selector.validate_question(question2)
    print(f"Question: '{question2}' is valid: {is_valid2}")

    area_medicine = "personalized medicine"
    suggested_topic_medicine = selector.suggest_topic(area_medicine)
    print(f"\nResearch Area: {area_medicine}")
    print(f"Suggested Topic: {suggested_topic_medicine}")

    question3 = "What are the ethical implications of AI-driven personalized medicine?"
    is_valid3 = selector.validate_question(question3)
    print(f"Question: '{question3}' is valid: {is_valid3}")


