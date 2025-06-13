
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, pipeline

class RubricGrader:
    """
    An automated rubric-based grader for research proposals/reports.
    Uses NLP and text similarity (e.g., transformers or sentence embeddings).
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initializes the RubricGrader with a specified device.
        Args:
            device (str): The device to use for computations ('cuda' or 'cpu').
        """
        self.device = torch.device(device)
        print(f"RubricGrader initialized on device: {self.device}")

        # Load a pre-trained model for sentence embeddings
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(self.device)

        # Initialize a text generation pipeline for feedback
        self.feedback_generator_pipeline = pipeline("text-generation", model="distilgpt2", device=0 if self.device.type == 'cuda' else -1)

    def _get_sentence_embedding(self, text: str) -> torch.Tensor:
        """
        Generates a sentence embedding for the given text.
        Args:
            text (str): The input text.
        Returns:
            torch.Tensor: The sentence embedding.
        """
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        # Mean pooling to get sentence embedding
        sentence_embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embedding

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def grade_submission(self, submission_text: str, rubric: dict) -> dict:
        """
        Grades a submission based on a provided rubric.
        Args:
            submission_text (str): The text of the student's submission.
            rubric (dict): A dictionary representing the rubric, e.g.,
                           {
                               "criterion1": {"expected_content": "...", "max_score": 10},
                               "criterion2": {"expected_content": "...", "max_score": 15}
                           }
        Returns:
            dict: A dictionary with scores and feedback for each criterion.
        """
        results = {}
        submission_embedding = self._get_sentence_embedding(submission_text)

        for criterion, details in rubric.items():
            expected_content = details["expected_content"]
            max_score = details["max_score"]

            expected_embedding = self._get_sentence_embedding(expected_content)

            # Calculate similarity using PyTorch
            similarity = cosine_similarity(submission_embedding.cpu().numpy(), expected_embedding.cpu().numpy())[0][0]

            # Scale score based on similarity
            score = float(similarity * max_score)
            score = max(0.0, min(max_score, score)) # Ensure score is within bounds

            feedback = self.generate_feedback(submission_text, expected_content, score, max_score)

            results[criterion] = {
                "score": score,
                "max_score": max_score,
                "similarity": similarity,
                "feedback": feedback
            }
        return results

    def generate_feedback(self, submission_text: str, expected_content: str, score: float, max_score: float) -> str:
        """
        Generates actionable and empathetic feedback based on the score.
        Args:
            submission_text (str): The student's submission text.
            expected_content (str): The expected content for the criterion.
            score (float): The calculated score for the criterion.
            max_score (float): The maximum possible score for the criterion.
        Returns:
            str: Generated feedback.
        """
        # More nuanced prompt for feedback generation
        if score / max_score > 0.8:
            feedback_prompt = f"The submission demonstrates strong understanding of the topic. Specifically, regarding '{expected_content[:50]}...', the submission was well-articulated. Provide further suggestions for excellence. Feedback:"
        elif score / max_score > 0.5:
            feedback_prompt = f"The submission shows a good grasp of the topic, but there are areas for improvement. For '{expected_content[:50]}...', consider elaborating on the following. Feedback:"
        else:
            feedback_prompt = f"The submission needs significant improvement in understanding '{expected_content[:50]}...'. Please review the core concepts related to this criterion. Feedback:"

        # Generate feedback using the pre-trained model
        generated_feedback = self.feedback_generator_pipeline(
            feedback_prompt,
            max_new_tokens=100,
            num_return_sequences=1,
            truncation=True
        )[0]['generated_text']

        # Post-process to remove the prompt itself from the generated text
        if generated_feedback.startswith(feedback_prompt):
            generated_feedback = generated_feedback[len(feedback_prompt):].strip()

        return generated_feedback

if __name__ == "__main__":
    # Example Usage
    if torch.cuda.is_available():
        device = 'cuda'
        print("CUDA is available! Using GPU.")
    else:
        device = 'cpu'
        print("CUDA not available. Using CPU.")

    grader = RubricGrader(device=device)

    rubric_example = {
        "Introduction Clarity": {"expected_content": "The introduction should clearly state the research question, its significance, and the paper's structure.", "max_score": 10},
        "Methodology Detail": {"expected_content": "The methodology section must provide sufficient detail for replication, including data sources, experimental design, and analytical techniques.", "max_score": 15}
    }

    submission1 = "This paper investigates the effects of climate change on polar bear populations. It will discuss data from satellite imagery and population surveys. The methods section outlines our approach."
    grades1 = grader.grade_submission(submission1, rubric_example)
    print("\n--- Grading Submission 1 ---")
    for criterion, result in grades1.items():
        print(f"Criterion: {criterion}")
        print(f"  Score: {result['score']:.2f}/{result['max_score']}")
        print(f"  Similarity: {result['similarity']:.2f}")
        print(f"  Feedback: {result['feedback']}")

    submission2 = "My research is about bears. I looked at some pictures. It was hard."
    grades2 = grader.grade_submission(submission2, rubric_example)
    print("\n--- Grading Submission 2 ---")
    for criterion, result in grades2.items():
        print(f"Criterion: {criterion}")
        print(f"  Score: {result['score']:.2f}/{result['max_score']}")
        print(f"  Similarity: {result['similarity']:.2f}")
        print(f"  Feedback: {result['feedback']}")


