import json
from pathlib import Path    #The `json` module is used for parsing JSON data, and `Path` from the `pathlib` module is used for handling file paths in a more convenient way.
from pydantic import BaseModel, Field #Pydantic is a data validation and settings management library that uses Python type annotations. `BaseModel` is the base class for creating data models, and `Field` is used to provide additional metadata and validation for model fields.

TEST_FILE = str(Path(__file__).parent / "tests.jsonl")  #here we assume the test questions are stored in a JSONL file named "tests.jsonl" in the same directory as this script.


class TestQuestion(BaseModel):
    """A test question with expected keywords and reference answer."""

    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(description="Keywords that must appear in retrieved context")
    reference_answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category (e.g., direct_fact, spanning, temporal)") #this class is used to represent a test question, which includes the question itself, a list of keywords that should be present in the retrieved context, a reference answer for evaluation, and a category to classify the type of question (e.g., direct fact, spanning multiple documents, temporal reasoning, etc.).


def load_tests() -> list[TestQuestion]: #this function is responsible for loading the test questions from the specified JSONL file. It reads each line of the file, parses it as JSON, and creates a `TestQuestion` object using the parsed data. The resulting list of `TestQuestion` objects is returned for use in testing the RAG system.
    """Load test questions from JSONL file."""
    tests = [] #it is an empty list that will be populated with `TestQuestion` objects created from the data in the JSONL file. Each line in the file is expected to contain a JSON object that can be parsed into a `TestQuestion` instance.
    with open(TEST_FILE, "r", encoding="utf-8") as f: #the file is opened in read mode with UTF-8 encoding to ensure that it can handle a wide range of characters. The `with` statement is used to ensure that the file is properly closed after its suite finishes, even if an error occurs.
        for line in f: #the file is read line by line, and for each line, the `json.loads` function is used to parse the JSON data into a Python dictionary. The resulting dictionary is then unpacked using `**data` to create a `TestQuestion` object, which is appended to the `tests` list.
            data = json.loads(line.strip()) #the `strip()` method is used to remove any leading or trailing whitespace from the line before parsing it as JSON. This ensures that the JSON data is clean and can be parsed correctly without any issues caused by extra whitespace.
            tests.append(TestQuestion(**data))  #the `**data` syntax is used to unpack the dictionary returned by `json.loads` and pass its keys as keyword arguments to the `TestQuestion` constructor. This allows for easy creation of `TestQuestion` instances from the JSON data. The function returns a list of `TestQuestion` objects that can be used for testing the RAG system.
    return tests #when the TestQuestion objects are created, the `**data` syntax is used to unpack the dictionary returned by `json.loads` and pass its keys as keyword arguments to the `TestQuestion` constructor. This allows for easy creation of `TestQuestion` instances from the JSON data. The function returns a list of `TestQuestion` objects that can be used for testing the RAG system.

