"""
Util Module

Holds utility classes and functions.

Example:

    TextManipulator holds methods for spellchecking and indexing text
    InputChecker holds methods for verifying inputs.
    Both classes are used as a collection of methods, rather than relying on
    any instance specific data

        text_util = TextManipulator()
        text_util.correct_spacing('s ome miss pelled te xt')

        input_checker = InputChecker()
        input_checker.can_be_int('1')

"""

import datetime as dt
import re
from pathlib import Path
import logging
import enchant


def set_up_logging(logging_dir: Path):
    """Set up logging for run of main script

    Args:
        logging_dir (Path): path to save logging file in
    """
    logging_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        encoding="utf-8",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(logging_dir / "run.log"),
            logging.StreamHandler(),
        ],
    )

    st_logger = logging.getLogger("sentence_transformers")
    st_logger.setLevel(logging.WARNING)


class TextManipulator:
    """Utility class for text manipulation

    eg spell checking and indexing text
    """

    def __init__(self):
        """Set dictionary and sentence split regex pattern"""
        self.english_dict = enchant.Dict("en_US")
        self.sentence_split_pattern = r"(?<=[.?!])\s+"

    @staticmethod
    def find_all_matches(text: str, terms: list) -> list:
        """Returns a list of indexes for all occurrences of terms in text

        Matches are made non case sensitive

        Args:
            text (str): a longer text to search within
            terms (list): a list of terms to search for

        Returns:
            list: all start indices of terms in text
        """
        all_positions = []
        for term in terms:
            matches = list(re.finditer(term, text, re.IGNORECASE))
            positions = [match.start() for match in matches]
            all_positions += positions
        return all_positions

    def split_into_sentences(self, text: str):
        """Splits text document into sentences using regular expression

        Args:
            text (str): text document to be split

        Returns:
            list: list of str (individual sentences)
        """
        return re.split(self.sentence_split_pattern, text)

    @staticmethod
    def find_sentence_starts(text: str, sentences: list):
        """Find the indices of the starts of sentences in a document

        Args:
            text (str): document as a long string
            sentences (_type_): document split into sentences

        Returns:
            list: list of sentence start indices
        """
        start_indices = []
        current_pos = 0
        for sentence in sentences:
            # Find the start index of the sentence in the original text
            start_index = text.find(sentence, current_pos)
            if start_index != -1:
                start_indices.append(start_index)
                # Update current_pos to the end of the current sentence
                current_pos = start_index + len(sentence)

        return start_indices

    @staticmethod
    def find_closest_index(start_indices, chosen_value, direction="forward"):
        """finds closest index to chosen value

        takes a list of indices (start_indices), finds the closest index either
            before (when direction = "backward) or after (if direction =
            "forward") a chosen value


        Args:
            start_indices (list): list of index
            chosen_value (int): a single value
            direction (str, optional): Whether to look forward or backward
                when searching for the closest index. Defaults to "forward".

        Returns:
            closest_index: index in start indices closest to chosen values,
                depending on direction
        """
        closest_index = None
        for index in start_indices:
            if index > chosen_value:
                next_index = index
                prev_index = start_indices[start_indices.index(index) - 1]
                break
            # if at final index
            elif start_indices.index(index) == len(start_indices) - 1:
                next_index = index
                prev_index = index
                break

        if direction == "forward":
            closest_index = next_index
        else:
            closest_index = prev_index
        return closest_index

    def get_section(
        self, text: str, sentences: list, chosen_idx: int, n_sentence: int
    ) -> list:
        """From an index in a text document, get a section of n sentences long

        Args:
            text (str): Full document as a single string
            sentences (list): document split into a list of sentences
            chosen_idx (int): index to find section around
            n_sentence (int): length of section to return

        Returns:
            list: list containing section of text
        """
        # find idxs of sentences in main text
        sent_start_idxs = self.find_sentence_starts(text, sentences)
        # find sentence that includes chosen idx
        start_idx = self.find_closest_index(
            sent_start_idxs, chosen_idx, direction="backwards"
        )
        # check you can take n sentences from chosen idx without running out of text
        if sent_start_idxs.index(start_idx) + n_sentence < len(sentences):
            end_idx = sent_start_idxs[sent_start_idxs.index(start_idx) + n_sentence]
        else:
            # if not take as many sentences as possible
            n = 0
            end_idx = len(text)

            while sent_start_idxs.index(start_idx) + n_sentence < len(sentences):
                end_idx = sent_start_idxs[sent_start_idxs.index(start_idx) + n]
                n += 1
        section = text[start_idx:end_idx]

        return section

    def correct_spacing(self, text):
        """Remove spaces inserted into words

        Insertion of random spaces happens when reading text from PDFs
        If joining words either side of a space makes a valid word,
            then words will be combined

        Args:
            text (str): Document (ie text loaded from PDF) to be corrected

        Returns:
            str: corrected text
        """

        # Split the text into words and punctuation
        words = re.findall(r"\b\w+\b|\S", text)
        corrected_words = []
        i = 0
        while i < len(words):
            word = words[i]
            if i + 1 < len(words):
                combined_word = word + words[i + 1]
                if self.english_dict.check(combined_word):
                    corrected_words.append(combined_word)
                    i += 1  # Skip the next word
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)

            i += 1
        return " ".join(corrected_words)


class InputChecker:
    """Utility class holding input checking methods"""

    def __init__(self):
        """"""

    @staticmethod
    def check_search_years(start_year, end_year):
        """Check that year inputs are valid numbers in acceptable range"""

        start_year = InputChecker.can_be_int(
            start_year, "Year inputs must be convertible to an integer"
        )
        end_year = InputChecker.can_be_int(
            end_year, "Year inputs must be convertible to an integer"
        )

        InputChecker.validate_input(
            start_year, int, lambda x: x >= 1975, "Start year must be 1975 or later"
        )

        InputChecker.validate_input(
            end_year,
            int,
            lambda x: x <= dt.datetime.now().year,
            f"End year must be {dt.datetime.now().year} or earlier",
        )

    @staticmethod
    def can_be_int(input_value, error_message):
        """Check that input can be typecast to int"""
        try:
            input_value = int(input_value)
        except ValueError:
            raise ValueError(error_message)
        return input_value

    @staticmethod
    def validate_input(
        input_value,
        expected_type: type,
        condition=lambda x: True,
        error_message="Invalid input",
    ):
        """Validate that input is of correct type and fufills condition, otherwise
            raise error message

        Args:
            input_value: input to be validated
            expected_type (type): type input should be.
            condition (optional): condition that to be fulfilled. eg lambda x: x > 10
            error_message (str, optional): Message to display in case of error. Defaults
                to "Invalid input".

        Raises:
            TypeError: raised if input is not expected type
            ValueError: raised if input do not fulfill condition
        """
        if not isinstance(input_value, expected_type):
            raise TypeError(
                f"Expected {expected_type}, got {type(input_value)} instead"
            )
        if not condition(input_value):
            raise ValueError(error_message)

    @staticmethod
    def check_file_path(path: Path, stage: str):
        """Check file path exists and remind to run stage of pipeline if not

        Args:
            path (Path): file path to verify
            stage (str): Name to stage that should produce file

        Raises:
            FileNotFoundError: If file not found
        """
        stage = stage.replace("_", " ").title()
        if not Path(path).exists():
            raise FileNotFoundError(f'No file found. Ensure you have run "{stage}"')

    @staticmethod
    def check_search_terms(input_value, input_name: str):
        """Check output input is str or list of str

        if input is single str, convert to list of str

        Args:
            input_value (): search_terms inputted
            input_name (str): name of search_term variable being checked

        Raises:
            TypeError:

        Returns:
            list: input_value as a list
        """
        if isinstance(input_value, str):
            output = [input_value]
        elif isinstance(input_value, list):
            if all(isinstance(element, str) for element in input_value):
                output = input_value
            else:
                raise TypeError(
                    f"{input_name} must be a single string or list of strings"
                )
        return output

    @staticmethod
    def check_query_settings(query_settings):
        """Check each key in the query settings dict passed to ClauseExtractor

        Args:
            query_settings (dict): dict holding query settings

        Returns:
            query_settings: input dict with search terms converted to list
        """
        InputChecker.check_search_terms(
            query_settings["match_sentences"], "match_sentences"
        )
        InputChecker.is_str(query_settings["clause_query"], "clause_query")
        InputChecker.is_str(query_settings["date_query"], "date_query")
        query_settings["n_sentences"] = InputChecker.can_be_int(
            query_settings["n_sentences"],
            "Number of sentences must be convertible to int",
        )
        InputChecker.is_float(
            query_settings["similarity_threshold"], "similarity_threshold"
        )
        return query_settings

    @staticmethod
    def is_str(input_value, input_name):
        """Check input is str"""
        if not isinstance(input_value, str):
            raise TypeError(f"{input_name} must be a single string")

    @staticmethod
    def is_float(input_value, input_name):
        """check input can be a float between 0 and 1"""
        if isinstance(input_value, int):
            input_value = float(input_value)
        InputChecker.validate_input(
            input_value,
            float,
            lambda x: 0 < x <= 1,
            error_message=f"{input_name} must be between 0 and 1",
        )
