import sys
import math
from pydantic import BaseModel, Field
from litellm import completion
from dotenv import load_dotenv #The `dotenv` module is used to load environment variables from a `.env` file. The `load_dotenv` function is called with `override=True`, which means that any existing environment variables will be overridden by the values in the `.env` file. This is often used to manage configuration settings, such as API keys or database credentials, without hardcoding them into the codebase.

from evaluation.test import TestQuestion, load_tests
from implementation.answer import answer_question, fetch_context #these functions are imported from the `answer` module in the `implementation` package. The `answer_question` function is likely responsible for generating an answer to a given question using the RAG system, while the `fetch_context` function is probably used to retrieve relevant documents or context based on the question. These functions will be used in the evaluation process to assess both the retrieval performance and the quality of the generated answers.


load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
db_name = "vector_db"   #the name of the vector database that will be used for retrieval. This variable is likely used in the `fetch_context` function to specify which vector database to query when retrieving relevant documents based on the test questions. The actual implementation of how this variable is used would depend on the details of the `fetch_context` function and how it interacts with the vector database.


class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance."""

    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")     #this class is used to represent the evaluation metrics for retrieval performance. It includes fields for Mean Reciprocal Rank (MRR), Normalized Discounted Cumulative Gain (nDCG), the number of keywords found in the top-k results, the total number of keywords that were supposed to be found, and the percentage of keywords that were successfully found in the retrieved context. These metrics will be calculated and used to assess how well the retrieval component of the RAG system is performing in terms of finding relevant information based on the test questions.


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

    feedback: str = Field(
        description="Concise feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context"
    )
    accuracy: float = Field(
        description="How factually correct is the answer compared to the reference answer? 1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). An acceptable answer would score 3."
    )
    completeness: float = Field(
        description="How complete is the answer in addressing all aspects of the question? 1 (very poor - missing key information) to 5 (ideal - all the information from the reference answer is provided completely). Only answer 5 if ALL information from the reference answer is included."
    )
    relevance: float = Field(
        description="How relevant is the answer to the specific question asked? 1 (very poor - off-topic) to 5 (ideal - directly addresses question and gives no additional information). Only answer 5 if the answer is completely relevant to the question and gives no additional information."
    )   #this class is used to represent the evaluation of answer quality using an LLM as a judge. It includes fields for feedback, which is a concise evaluation of the answer quality comparing it to the reference answer and the retrieved context, as well as numerical scores for accuracy, completeness, and relevance. The accuracy score assesses how factually correct the answer is compared to the reference answer, with a scale from 1 (wrong) to 5 (ideal). The completeness score evaluates how thoroughly the answer addresses all aspects of the question, also on a scale from 1 to 5. The relevance score measures how well the answer directly addresses the specific question without providing additional information, again on a scale from 1 to 5. These metrics will be used to assess the quality of the generated answers in comparison to the reference answers and the retrieved context.


def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    """Calculate reciprocal rank for a single keyword (case-insensitive)."""
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank
    return 0.0  #this function calculates the Mean Reciprocal Rank (MRR) for a single keyword based on the retrieved documents. It iterates through the list of retrieved documents, checking if the keyword (in a case-insensitive manner) is present in the content of each document. If the keyword is found, it returns the reciprocal of the rank (1 divided by the position of the document in the list). If the keyword is not found in any of the retrieved documents, it returns 0.0, indicating that the keyword was not successfully retrieved. This function will be used to evaluate how well the retrieval system is performing in terms of finding relevant information based on the test questions.


def calculate_dcg(relevances: list[int], k: int) -> float:
    """Calculate Discounted Cumulative Gain."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)  # i+2 because rank starts at 1
    return dcg  #this function calculates the Discounted Cumulative Gain (DCG) based on a list of relevance scores for the retrieved documents. The DCG is calculated by iterating through the relevance scores up to a specified rank `k` and applying a logarithmic discount to the relevance score based on its position in the list. The relevance score of each document is divided by the logarithm of its rank (starting from 1), which gives more weight to higher-ranked documents. The resulting DCG value is returned, which can be used in conjunction with the ideal DCG (IDCG) to calculate the Normalized Discounted Cumulative Gain (nDCG) for evaluating retrieval performance.


def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 10) -> float:
    """Calculate nDCG for a single keyword (binary relevance, case-insensitive)."""
    keyword_lower = keyword.lower()

    # Binary relevance: 1 if keyword found, 0 otherwise
    relevances = [
        1 if keyword_lower in doc.page_content.lower() else 0 for doc in retrieved_docs[:k]
    ]

    # DCG
    dcg = calculate_dcg(relevances, k)

    # Ideal DCG (best case: keyword in first position)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0  #this function calculates the Normalized Discounted Cumulative Gain (nDCG) for a single keyword based on the retrieved documents. It first creates a list of binary relevance scores, where each score is 1 if the keyword is found in the document (case-insensitive) and 0 otherwise, for the top `k` retrieved documents. It then calculates the DCG using the `calculate_dcg` function. To calculate the ideal DCG (IDCG), it sorts the relevance scores in descending order (best case scenario where the keyword is found in the highest-ranked document) and calculates the IDCG using the same `calculate_dcg` function. Finally, it returns the nDCG value by dividing the DCG by the IDCG, ensuring that if IDCG is 0 (which would indicate that there are no relevant documents), it returns 0.0 to avoid division by zero. This function will be used to evaluate how well the retrieval system ranks relevant documents based on the presence of keywords in relation to their position in the retrieved list.


def evaluate_retrieval(test: TestQuestion, k: int = 10) -> RetrievalEval:
    """
    Evaluate retrieval performance for a test question.

    Args:
        test: TestQuestion object containing question and keywords
        k: Number of top documents to retrieve (default 10)

    Returns:
        RetrievalEval object with MRR, nDCG, and keyword coverage metrics
    """
    # Retrieve documents using shared answer module
    retrieved_docs = fetch_context(test.question)

    # Calculate MRR (average across all keywords)
    mrr_scores = [calculate_mrr(keyword, retrieved_docs) for keyword in test.keywords]
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    # Calculate nDCG (average across all keywords)
    ndcg_scores = [calculate_ndcg(keyword, retrieved_docs, k) for keyword in test.keywords]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    # Calculate keyword coverage
    keywords_found = sum(1 for score in mrr_scores if score > 0)
    total_keywords = len(test.keywords)
    keyword_coverage = (keywords_found / total_keywords * 100) if total_keywords > 0 else 0.0

    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )   #this function evaluates the retrieval performance for a given test question by calculating the Mean Reciprocal Rank (MRR), Normalized Discounted Cumulative Gain (nDCG), and keyword coverage metrics. It retrieves relevant documents using the `fetch_context` function, then calculates the MRR and nDCG for each keyword in the test question using the previously defined functions. The average MRR and nDCG scores are computed across all keywords. Additionally, it counts how many keywords were found in the retrieved documents to calculate the keyword coverage percentage. Finally, it returns a `RetrievalEval` object containing all these metrics, which can be used to assess the effectiveness of the retrieval component of the RAG system for that specific test question.


def evaluate_answer(test: TestQuestion) -> tuple[AnswerEval, str, list]:
    """
    Evaluate answer quality using LLM-as-a-judge (async).

    Args:
        test: TestQuestion object containing question and reference answer

    Returns:
        Tuple of (AnswerEval object, generated_answer string, retrieved_docs list)
    """
    # Get RAG response using shared answer module
    generated_answer, retrieved_docs = answer_question(test.question)

    # LLM judge prompt
    judge_messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator assessing the quality of answers. Evaluate the generated answer by comparing it to the reference answer. Only give 5/5 scores for perfect answers.",
        },
        {
            "role": "user",
            "content": f"""Question:    #This entire function evaluates the quality of the generated answer for a given test question using an LLM as a judge. It first generates an answer using the `answer_question` function, which also retrieves relevant documents. Then, it constructs a prompt for the LLM to evaluate the generated answer by comparing it to the reference answer provided in the test question. The prompt instructs the LLM to provide feedback and scores for accuracy, completeness, and relevance based on specific criteria. The response from the LLM is expected to be in a structured format that can be parsed into an `AnswerEval` object, which contains the evaluation results. The function returns a tuple containing the `AnswerEval` object, the generated answer string, and the list of retrieved documents for further analysis if needed.
{test.question} 

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
2. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1.""",
        },
    ]

    # Call LLM judge with structured outputs (async)
    judge_response = completion(model=MODEL, messages=judge_messages, response_format=AnswerEval)

    answer_eval = AnswerEval.model_validate_json(judge_response.choices[0].message.content)

    return answer_eval, generated_answer, retrieved_docs


def evaluate_all_retrieval():
    """Evaluate all retrieval tests."""
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_retrieval(test)
        progress = (index + 1) / total_tests
        yield test, result, progress    #this function evaluates the retrieval performance for all test questions by loading the tests and iterating through them. For each test, it calls the `evaluate_retrieval` function to get the evaluation results, and it calculates the progress as a percentage of tests completed. The function yields a tuple containing the test question, the retrieval evaluation result, and the progress, allowing for asynchronous processing or real-time updates in a user interface while the evaluations are being performed.


def evaluate_all_answers():
    """Evaluate all answers to tests using batched async execution."""
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_answer(test)[0]
        progress = (index + 1) / total_tests
        yield test, result, progress    #this function evaluates the quality of generated answers for all test questions by loading the tests and iterating through them. For each test, it calls the `evaluate_answer` function to get the evaluation results (specifically the `AnswerEval` object), and it calculates the progress as a percentage of tests completed. The function yields a tuple containing the test question, the answer evaluation result, and the progress, allowing for asynchronous processing or real-time updates in a user interface while the evaluations are being performed.


def run_cli_evaluation(test_number: int):
    """Run evaluation for a specific test (async helper for CLI)."""
    # Load tests
    tests = load_tests("tests.jsonl")

    if test_number < 0 or test_number >= len(tests):
        print(f"Error: test_row_number must be between 0 and {len(tests) - 1}")
        sys.exit(1)

    # Get the test
    test = tests[test_number]

    # Print test info
    print(f"\n{'=' * 80}")
    print(f"Test #{test_number}")
    print(f"{'=' * 80}")
    print(f"Question: {test.question}")
    print(f"Keywords: {test.keywords}")
    print(f"Category: {test.category}")
    print(f"Reference Answer: {test.reference_answer}")

    # Retrieval Evaluation
    print(f"\n{'=' * 80}")
    print("Retrieval Evaluation")
    print(f"{'=' * 80}")

    retrieval_result = evaluate_retrieval(test)

    print(f"MRR: {retrieval_result.mrr:.4f}")
    print(f"nDCG: {retrieval_result.ndcg:.4f}")
    print(f"Keywords Found: {retrieval_result.keywords_found}/{retrieval_result.total_keywords}")
    print(f"Keyword Coverage: {retrieval_result.keyword_coverage:.1f}%")

    # Answer Evaluation
    print(f"\n{'=' * 80}")
    print("Answer Evaluation")
    print(f"{'=' * 80}")

    answer_result, generated_answer, retrieved_docs = evaluate_answer(test)

    print(f"\nGenerated Answer:\n{generated_answer}")
    print(f"\nFeedback:\n{answer_result.feedback}")
    print("\nScores:")
    print(f"  Accuracy: {answer_result.accuracy:.2f}/5")
    print(f"  Completeness: {answer_result.completeness:.2f}/5")
    print(f"  Relevance: {answer_result.relevance:.2f}/5")
    print(f"\n{'=' * 80}\n")    #this function is a command-line interface (CLI) helper that allows the user to run an evaluation for a specific test question by providing its row number as an argument. It loads the tests, checks if the provided test number is valid, and then retrieves the corresponding test question. The function prints the details of the test question, including the question itself, keywords, category, and reference answer. It then performs both retrieval and answer evaluations using the previously defined functions and prints the results in a structured format for easy reading. This allows users to quickly assess the performance of the RAG system on individual test questions directly from the command line.


def main():
    """CLI to evaluate a specific test by row number."""
    if len(sys.argv) != 2:
        print("Usage: uv run eval.py <test_row_number>")
        sys.exit(1)

    try:
        test_number = int(sys.argv[1])
    except ValueError:
        print("Error: test_row_number must be an integer")
        sys.exit(1)

    run_cli_evaluation(test_number) #this function serves as the main entry point for the command-line interface (CLI) to evaluate a specific test question by its row number. It checks if the correct number of arguments is provided, attempts to parse the test row number as an integer, and handles any errors that may arise from invalid input. If the input is valid, it calls the `run_cli_evaluation` function with the specified test number to perform the evaluation and display the results. This allows users to easily run evaluations for individual test questions directly from the command line by specifying their row numbers.


if __name__ == "__main__":
    main()
