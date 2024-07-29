"""
Process Module

General functions for the processing of data object and dataframes

"""

import ast
import io
import urllib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xlsxwriter
from langchain_community.document_loaders import PyPDFLoader

import src.toolkit.scrape as scrape
from src.toolkit.util import TextManipulator

pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore", category=UserWarning, module="xlsxwriter")


def extract_review_clause_text(
    link: str,
    session: requests.Session,
    search_terms: list,
    text_util: TextManipulator,
    n_sentences: int,
) -> list:
    """
    extracts review clauses from the main text of legislation

    extracts a section around an exact match of a search term
    within the legislation text

    Args:
        link (str): Link to legislation
        session (requests.Session): requests session
        search_terms (list): list of review clause terms (str)
            to extract

    Returns:
        list: list containing all sections of text with review
            clauses
    """
    full_text_link = scrape.prepend_govuk_url(link) + "made"
    fullpage_soup = scrape.get_soup_from_session(full_text_link, session)
    leg_text = fullpage_soup.get_text(" ", strip=True).replace("\xa0", " ")

    relevant_idxs = text_util.find_all_matches(leg_text, search_terms)
    if relevant_idxs:
        leg_sentences = text_util.split_into_sentences(leg_text)
        relevant_sections = [
            text_util.get_section(
                text=leg_text,
                sentences=leg_sentences,
                chosen_idx=idx,
                n_sentence=n_sentences,
            )
            for idx in relevant_idxs
        ]
        # remove duplicates - ie sentence matches 2+ search terms
        relevant_sections = list(set(relevant_sections))
    else:
        relevant_sections = np.nan
    return relevant_sections


def pdf_getter(url: str):
    """retrieves pdf from url as bytes object"""
    open_pdf = urllib.request.urlopen(url).read()
    return io.BytesIO(open_pdf)


def get_pdf_text(url: str):
    """extract text from pdf url"""
    loader = PyPDFLoader(url)
    pages = loader.load_and_split()
    pages_text = [page.page_content.replace("\n", " ") for page in pages]
    doc_text = "".join(pages_text)
    return doc_text


def tuplefy_cols(row: pd.Series, label: str) -> pd.Series:
    """Turn values in all columns in a row into tuples

    Tuple takes the format of (value, label)

    Args:
        row (pd.Series): row of df to operate on
        label (str): label to use in tuple

    Returns:
        pd.Series: output series with tuplefied columns
    """
    for col in row.columns:
        row.loc[0, col] = (row.loc[0, col], label)
    return row


def read_csv(path: Path, list_cols: list) -> pd.DataFrame:
    """Read function

    Reads from path whilst preserving nans,
    then converts list columns from list to str

    Args:
        path (Path): path to csv
        list_cols (list): columns containing lists

    Returns:
        pd.DataFrame: output dataframe
    """
    df = pd.read_csv(path, keep_default_na=True)
    df = convert_list_cols(df, list_cols)
    return df


def write_csv(df: pd.DataFrame, path: Path):
    """Write function - save csv while preserving nans

    Args:
        df (pd.DataFrame): df to write
        path (Path): path to save csv in
    """

    df.to_csv(path, na_rep="NaN", index=False)


def convert_list_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Evaluate list columns to literal lists

    When a df column with lists as values is saved and loaded,
    lists are loaded as str eg  '[4,5]'

    This function restores their list typing

    Args:
        df (pd.DataFrame): df containing loaded list columns
        cols (list): list of columns to convert

    Returns:
        pd.DataFrame: df with converted columns
    """
    for col in cols:
        intermediate = df[col]
        df.drop(columns=[col], inplace=True)
        df[col] = [
            x if not isinstance(x, str) else ast.literal_eval(x) for x in intermediate
        ]
    return df


def process_cols(df):
    """apply formatting and spreading"""
    df.columns = [x.title().replace("_", " ") for x in df.columns]
    df.columns = [x.replace("Ia", "IA") for x in df.columns]
    df.columns = [x.replace("Pirs", "PIRs") for x in df.columns]
    spread_cols(df, ["IAs", "PIRs", "Other Resources"])

    return df


def spread_cols(df: pd.DataFrame, cols: list):
    """Spread columns where each element is a list
    -> into n columns where each elements is a single value
    (where n in the longest list in the df)

    Args:
        df (pd.DataFrame): input df
        cols (list): list of columns to spread
    """
    for col in cols:
        max_len = df[col].dropna().apply(len).max()
        if not np.isnan(max_len):
            for i in range(max_len):
                df["_".join([col, str(i + 1)])] = df[col].apply(
                    lambda x: x[i] if isinstance(x, list) and i < len(x) else np.nan
                )
        df.drop(columns=[col], inplace=True)


def convert_to_combined_hyperlinks(
    workbook: pd.ExcelWriter.book,
    worksheet: xlsxwriter.worksheet,
    df: pd.DataFrame,
    column_name: str,
):
    """Convert tuples to hyperlinked text

    takes a column whose elements are (text, link)
    and converts to hyperlinked text

    Args:
        workbook (pd.ExcelWriter.book): workbook to be written to
        worksheet (xlsxwriter.worksheet): sheet of workbook to write to
        df (pd.DataFrame): dataframe being written to excel
        column_name (str): column to convert to hyperlinked text
    """
    col_idx = df.columns.get_loc(column_name)
    for row in range(len(df)):
        link = df.iloc[row, col_idx]
        if isinstance(link, tuple):
            output_text = f'=HYPERLINK("{link[1]}", "{link[0]}")'
        else:
            output_text = ""
        worksheet.write_formula(row + 1, col_idx, output_text)


def write_to_excel(excel_file: Path, df: pd.DataFrame):
    """Write final output to excel sheet

    Args:
        excel_file (Path): Location to save output file
        df (pd.DataFrame): dataframe to save
    """

    with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False)
        workbook = writer.book
        worksheet = writer.sheets["Sheet1"]

        link_cols = [
            x
            for x in df.columns
            if any(term in x for term in ["Other Resources", "IAs", "PIRs"])
        ]
        for col in link_cols:
            convert_to_combined_hyperlinks(workbook, worksheet, df, column_name=col)
        writer.close()
