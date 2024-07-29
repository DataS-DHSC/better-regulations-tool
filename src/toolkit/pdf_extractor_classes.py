"""
PDF Extractor Classes Module

Classes for the extraction of information from Impact Assessment PDFs.

This module contains two classes.
`Image2Document` uses tesseract-OCR to convert scanned PDFs into text.
`ClauseExtractor` processes PDF documents and extracts review clause
information


Example:
    Both classes use a single function call to process documents (either
    convert to text or extract information) after being initialized::

        image2doc = Image2Document(data_df)
        texts = image2doc.get_image_documents()

        extractor = ClauseExtractor(qa_model,
                                      data_df,
                                      query_settings,
                                      review_match_sentences)
        extractor.process_documents()

"""

import io
import subprocess  # noqa: F401
import os  # noqa: F401

import numpy as np
import pandas as pd
import pypdfium2 as pdfium
from PIL import Image
from pytesseract import image_to_string
import pytesseract  # noqa: F401
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_util
from tqdm import tqdm
from transformers import pipeline

import src.toolkit.process as process
import src.toolkit.scrape as scrape
from src.toolkit.util import TextManipulator


class Image2Document:
    """Converts scanned PDFs to text documents

    This takes a df containing links to scanned PDFs, converts them to images
        and uses tesseract to convert to text
    """

    def __init__(self, data_df: pd.DataFrame, image_scale=300 / 72):
        """

        Args:
            data_df (pd.DataFrame): df holding only scanned PDFs
            image_scale (int, optional): scale of created images. Defaults to 300/72.
        """
        self.text_util = TextManipulator()
        self.data_df = data_df
        self.scale = image_scale
        self.session = scrape.get_session_w_retry()

    def get_image_documents(self):
        """Gets text documents from scanned PDFs

        Returns:
            dict: dictionary containing text for each pdf in data_df
        """
        # tesseract_path = self.find_tesseract_in_conda_env('better-regs-env-old')
        # print(f'tess path {tesseract_path}')
        # pytesseract.pytesseract.tesseract_cmd  = tesseract_path
        img_doc_texts = {}
        for idx in self.data_df.index:
            print(f"Working on {self.data_df.loc[idx].title}")
            # check if this leg has pdfs to convert
            if isinstance(self.data_df.loc[idx].IAs, list):
                texts = {}
                for ia in self.data_df.loc[idx].IAs:
                    filename = ia[0]
                    pdf_as_img = self.convert_pdf_to_images(ia[1])
                    text = self.convert_image_to_text(pdf_as_img)
                    texts[filename] = self.text_util.correct_spacing(text)
                img_doc_texts[self.data_df.loc[idx].title] = texts
        return img_doc_texts

    def convert_pdf_to_images(self, url: str):
        """convert pdf urls to images

        Args:
            url (str): url of scanned pdf

        Returns:
            dict: dict of pdf pages as images
        """

        response = self.session.get(url)
        pdf_io_bytes = io.BytesIO(response.content)
        pdf_file = pdfium.PdfDocument(pdf_io_bytes)
        page_indices = [i for i in range(len(pdf_file))]

        renderer = pdf_file.render(
            pdfium.PdfBitmap.to_pil,
            page_indices=page_indices,
            scale=self.scale,
        )

        final_images = []
        for i, image in zip(page_indices, renderer):
            image_byte_array = io.BytesIO()
            image.save(image_byte_array, format="jpeg", optimize=True)
            image_byte_array = image_byte_array.getvalue()
            final_images.append(dict({i: image_byte_array}))
        pdf_file.close()
        return final_images

    def convert_image_to_text(self, list_dict_final_images: dict):
        """Converts pdfs in image form to a string

        Args:
            list_dict_final_images (dict): dict with image bytes for each page in pdf

        Returns:
            str: text from all pages of pdf as single string
        """
        image_list = [list(data.values())[0] for data in list_dict_final_images]
        image_content = []
        for index, image_bytes in enumerate(image_list):
            image = Image.open(io.BytesIO(image_bytes))
            # convert image to text using tesseract
            raw_text = str(image_to_string(image))
            image_content.append(raw_text)
        # return text from all pages as single string
        doc_text = " ".join(image_content)
        return doc_text

    # def find_tesseract_in_conda_env(self, conda_env_name):
    #     try:
    #         # Create the activation command based on your shell and environment
    #         activate_cmd = f'conda activate {conda_env_name} && where tesseract'

    #         # Run the activation command in a new subprocess
    #         result = subprocess.run(activate_cmd, shell=True,
    #  capture_output=True, text=True)

    #         # Check if the command was successful
    #         if result.returncode == 0:
    #             tesseract_path = result.stdout.strip()
    #             return tesseract_path
    #         else:
    #             print("Error:", result.stderr.strip())
    #             return None
    #     except Exception as e:
    #         print(f"An error occurred: {e}")
    #         return None


class ClauseExtractor:
    """Class for extracting review clauses from IAs

    Algorithm finds relevant section in text and feeds into question
        -answering BERT based model to extract details on review clauses

    For each pdf document for each row in data df ClauseExtractor will:

    1) First try to find a review clause section by searching for an exact match
        for every sentence in "review_match_sentences"
    2) If no exact match will try find a most relevant section in the doc via
        calculating cosine similarity between each sentence and an a generic
        review query ('will it be reviewed')
    3) If a relevant review clause section is found, will use "qa_model" and the queries
        supplied in "query_settings["clause_query"]" and "query_settings["date_query"]"
        to extract to extract review clause and review clause date info respectively

    """

    def __init__(
        self,
        qa_model: pipeline,
        data_df: pd.DataFrame,
        query_settings: dict,
        review_match_sentences: list,
        embedding_model=None,
        doc_texts=None,
    ):
        """initialization

        Args:
            qa_model (pipeline): Name of BERT model to create hugging face
                question-answering pipeline with.
            data_df (pd.DataFrame): dataframe with links to IAs in "IAs" column
            query_settings (dict): query parameters
            review_match_sentences (list): list of IA review section sentences to match
                against
            embedding_model (SentenceTransformer, optional): Name of model for creating
                text embeddings. Defaults to None.
            doc_texts (dict, optional): dictionary of texts from PDFs documents that
                exist in data_df - to have review clauses extracted. If no doc_texts
                is passed then one will be created to include every row of data_df
        """

        self.data_df = process.convert_list_cols(data_df, ["IAs"])
        self.query_settings = query_settings
        self.review_match_sentences = review_match_sentences
        self.clause_answers = {}
        self.date_answers = {}
        self.text_util = TextManipulator()
        if embedding_model:
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            self.embedding_model = SentenceTransformer("all-minilm-l6-v2")

        self.qa_model = pipeline("question-answering", model=qa_model)
        self.doc_texts = doc_texts

    def process_documents(self):
        """Main function

        For each row in data_df, extract review clause info and add to
            df in place
        """
        self.populate_documents()
        for leg in tqdm(self.doc_texts.keys()):
            leg_idx = self.data_df.query("title == @leg").index
            self.extract_review_details(leg, leg_idx)

    def populate_documents(self):
        """Check if doc_texts has been supplied and get documents if not"""
        if not self.doc_texts:
            self.get_documents()

    def get_documents(self):
        """Read text from pdf urls and store in dictionary"""
        doc_texts = {}
        for idx in self.data_df.index:
            # check if IA pdfs exist for this leg
            if isinstance(self.data_df.loc[idx].IAs, list):
                texts = {}
                for ia in self.data_df.loc[idx].IAs:
                    filename = ia[0]
                    text = process.get_pdf_text(ia[1])
                    texts[filename] = self.text_util.correct_spacing(text)
                doc_texts[self.data_df.loc[idx].title] = texts
        self.doc_texts = doc_texts

    def extract_review_details(self, leg: str, leg_idx: int):
        """Extract IA review clause from legislation

        Args:
            leg (str): legislation title
            leg_idx (int): index of leg row in data_df
        """

        all_rev_res = pd.DataFrame()
        # for every IA for this leg
        for doc in self.doc_texts[leg].keys():
            doc_text = self.doc_texts[leg][doc]
            # if doc text could be read ie not scanned
            if len(doc_text) > 80:  # NB magic number
                rev_results = self.qa_IA_details(doc_text, leg_idx)
            else:
                self.data_df.at[leg_idx[0], "Scanned PDF"] = "TRUE"
                rev_results = pd.DataFrame(
                    {
                        "review_clause": ["Unscrapeable PDF"],
                        "review_date": ["Unscrapeable PDF"],
                    }
                )
            all_rev_res = pd.concat([all_rev_res, rev_results])
        # if multiple pdfs, condense to fit in one row in output df
        wording, date = ClauseExtractor._condense_results(all_rev_res)
        self.data_df.at[leg_idx[0], "IA_review_clause"] = wording
        self.data_df.at[leg_idx[0], "IA_review_date"] = date

    @staticmethod
    def _condense_results(all_rev_res: pd.DataFrame):
        """condense multiple rows of df into a single row

        Args:
            all_rev_res (pd.DataFrame): df holding results from
                review clause extraction

        Returns:
            str, str: extracted review wording and review date
        """

        if not all_rev_res.empty:
            if len(all_rev_res) > 1:
                wording = [all_rev_res.review_clause.to_list()]
                date = [all_rev_res.review_date.to_list()]
            else:
                wording = all_rev_res.iloc[0].review_clause
                date = all_rev_res.iloc[0].review_date
        else:
            wording = np.nan
            date = np.nan
        return wording, date

    def qa_IA_details(self, doc_text: str, leg_idx: int) -> pd.DataFrame:
        """Extract information from IA using question-answering

        Args:
            doc_text (str): PDF document as string
            leg_idx (int): index of legislation in data_df

        Returns:
            pd.DataFrame: results df with columns 'review_clause', 'review date'
        """
        results = pd.DataFrame()
        doc_sentences = self.text_util.split_into_sentences(doc_text)

        # try to find exact match
        rev_clause_section = self._try_get_review_match_sentences(
            doc_text, doc_sentences, leg_idx
        )

        # otherwise try to find similar sentence w cosine similarity
        if not rev_clause_section:
            rev_clause_section = self.get_max_sim_sentences(
                doc_sentences, doc_text, leg_idx
            )

        if rev_clause_section:
            results.loc[0, "review_clause"] = self.extract_info_with_qa(
                self.query_settings["clause_query"], rev_clause_section
            )
            results.loc[0, "review_date"] = self.extract_info_with_qa(
                self.query_settings["date_query"], rev_clause_section
            )
            self.data_df.at[leg_idx[0], "section"] = rev_clause_section
        else:
            results.loc[0, ["review_clause", "review_date"]] = "", ""
        return results

    def _try_get_review_match_sentences(
        self, doc_text: str, doc_sentences: list, leg_idx: int
    ) -> list:
        """Try to find an exact match in the document to "review_match_sentences"

        Searches for a match in order, (ie will look for 1st element of
        review_match_sentences" 1st, and only look for the second element if
        the 1st cannot be found.
        If match found, take a section of the document including the match
        Otherwise return an empty list

        Args:
            doc_text (str): document as single string
            doc_sentences (list): document split into a list of sentences (str)
            leg_idx (int): index of legislation in data df

        Returns:
            list: list containing relevant section of document
        """

        stop_review_clause_search = False
        s = 0
        # search for review_match sentence in document in order
        while not stop_review_clause_search:
            query_idx = doc_text.find(self.review_match_sentences[s])
            # if match found stop searching
            if query_idx != -1:
                review_clause_idx = query_idx
                stop_review_clause_search = True
                self.data_df.at[leg_idx[0], "match"] = self.review_match_sentences[s]
            elif s < len(self.review_match_sentences) - 1:
                s += 1
            else:
                review_clause_idx = -1
                stop_review_clause_search = True

        if review_clause_idx != -1:
            rel_section = self.text_util.get_section(
                doc_text,
                doc_sentences,
                review_clause_idx,
                n_sentence=self.query_settings["n_sentences"],
            )
        else:
            rel_section = []
        return rel_section

    def get_max_sim_sentences(
        self, doc_sentences: list, doc_text: str, leg_idx: int, query="it be reviewed"
    ) -> list:
        """Finds most relevant section in document to the query using cosine sim

        If the match between most similar sentence and query is less than threshold,
            then return an empty list instead

        Args:
            doc_sentences (list): document split into a list of sentences (str)
            doc_text (str): document as a single string
            leg_idx (int): index of legislation in data df
            query (str, optional): query to use to find most similar sentence.
                                   Defaults to "it be reviewed".

        Returns:
            list: list containing relevant section of document
        """

        top_sentence_idx, score = self.find_sim_index(query, doc_sentences, doc_text)

        rel_section = self.text_util.get_section(
            doc_text,
            doc_sentences,
            top_sentence_idx,
            n_sentence=self.query_settings["n_sentences"],
        )

        self.data_df.at[leg_idx[0], "match"] = "similarity search"
        self.data_df.at[leg_idx[0], "cossim_score"] = score

        if score < self.query_settings["similarity_threshold"]:
            rel_section = []
        return rel_section

    def find_sim_index(self, query: str, doc_sentences: list, doc_text: str):
        """Find index of most similar sentences to the query

        Args:
            query (str): query to use to find most similar sentence
            doc_sentences (list):  document split into a list of sentences (str)
            doc_text (str): document as a single string

        Returns:
            int: index in doc_text of the start of the most similar sentence to
                the query
            float: cosine similarity score between the query and most similar
                sentence
        """
        query_embedding = self.embedding_model.encode(query)
        doc_sentence_embedding_df = self.get_embeddings(doc_sentences)

        cossim_scores = st_util.cos_sim(
            query_embedding,
            doc_sentence_embedding_df.astype("float32").to_numpy(),
        ).T

        max_sim_idx = cossim_scores.argmax().item()
        top_sentence = doc_sentences[max_sim_idx]
        top_sentence_idx = doc_text.find(top_sentence)
        score = cossim_scores[max_sim_idx]
        return top_sentence_idx, score.item()

    def extract_info_with_qa(self, question: str, section: str):
        """Extract info from a section of text using question-answering

        Args:
            question (str): question to pass to model
            section (str): section of document to pass as context to model

        Returns:
            str: info extracted from section as "answer" to the question
        """
        answer = self.qa_model(question=question, context=section)
        answer_df = pd.DataFrame(answer, index=[0])
        extracted_info = answer_df.iloc[0].answer

        return extracted_info

    def get_embeddings(self, doc_sentences: list) -> pd.DataFrame:
        """Convert a list of sentences into a df of embeddings

        Args:
            doc_sentences (list): : document split into a list of sentences (str)

        Returns:
            pd.DataFrame: df with each row the embedding for a sentence in the document
        """
        embeddings = [
            self.embedding_model.encode(sentence) for sentence in doc_sentences
        ]
        embedding_dfs = [
            pd.DataFrame(embedding.reshape(1, -1)) for embedding in embeddings
        ]
        all_embedding_df = pd.concat(embedding_dfs)
        return all_embedding_df
