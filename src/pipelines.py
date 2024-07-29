"""
Pipelines Module

Individual functions for each step in the tool and a master function run them in order.

The pipelines process contains a series of functions (pipelines)  that execute,
a single step in the process of extracting review clauses from legislation (eg
scraping legislation urls, accessing impact assessments, extracting info etc).
The master pipeline runs the sub-pipelines in order, according to boolean

"""

import logging
import os
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import swifter  # noqa: F401
from tqdm import tqdm

import src.toolkit.process as process
import src.toolkit.scrape as scrape
from src.toolkit.pdf_extractor_classes import ClauseExtractor, Image2Document
from src.toolkit.util import InputChecker, TextManipulator

tqdm.pandas()


def run_review_clause_scrape(config: dict, base_dir: Path):
    """Master function. Runs subpipelines

    Runs a subsection of subpipelines according boolean toggles set in config file

    Args:
        config (dict): Dictionary holding config settings
        base_dir (Path): Path to base directory
    """
    run_name = config["run_name"]
    checker = InputChecker()
    data_dir = base_dir / f"inputs/data/{run_name}"
    output_dir = base_dir / f"outputs/{run_name}"

    if config["scrape_links"]:
        scrape_custom_search(
            save_dir=data_dir / "raw",
            start_year=config["scraping"]["start_year"],
            end_year=config["scraping"]["end_year"],
            search_term=config["scraping"]["search_term"],
        )

    if config["scrape_more_resources"]:
        checker.check_file_path(data_dir / "raw", "scrape_links")
        scrape_more_resources(
            raw_data_dir=data_dir / "raw",
            save_dir=data_dir / "more_res",
        )

    if config["extract_review_clauses"]:
        checker.check_file_path(data_dir / "more_res", "scrape_more_resources")
        legs = process.read_csv(
            base_dir / data_dir / "more_res/legs_urls_from_search_more_res.csv",
            ["IAs", "PIRs", "other_resources"],
        )

        legs = get_review_clause_text(legs, config)
        legs = get_review_clause_IA(legs, config)

        process.write_csv(
            df=legs,
            path=data_dir / "legs_urls_from_search_extracted.csv",
        )
        process.read_csv(
            data_dir / "legs_urls_from_search_extracted.csv",
            ["IAs", "PIRs", "other_resources"],
        )

        process_and_save(
            legs,
            output_dir / 
            f"{run_name}_final_output.xlsx",
        )


def scrape_custom_search(
    save_dir: Path, start_year: int, end_year: int, search_term: str
):
    """Scrape links from legislation.gov.uk according to custom search

    Args:
        save_dir (Path): Folder to save df with scraped leg info
        start_year (int): year to scrape legs from
        end_year: (int): year to scrape legs up until
        search_term (str): term to search for within leg
    """
    checker = InputChecker()
    checker.check_search_years(start_year, end_year)

    save_dir.mkdir(parents=True, exist_ok=True)
    search_term_encoded = quote(search_term)
    search_url = (
        f"https://www.legislation.gov.uk/all/{start_year}-{end_year}?text=%22"
        f"{search_term_encoded}%22"
    )
    logging.info(
        "--------- Starting legislation scrape - scraping from %s  ---------",
        search_url,
    )
    retriever = scrape.UrlRetriever(url=search_url)
    urls = retriever.get_urls()
    urls.reset_index(inplace=True, drop=True)
    process.write_csv(df=urls, path=save_dir / "legs_urls_from_search.csv")


def scrape_more_resources(raw_data_dir: Path, save_dir: Path):
    """For scraped legs - scrape links to PDFs on "More Resources" page

    Takes output of "scrape_custom_search()" and adds columns containing links
        to PDFs found on the "More Resources" page for each leg on legislation.gov.
        Separates docs into Impact Assessments (IAs), Post implementation reviews (PIRs)
        and other documents
    Args:
        raw_data_dir (Path): Folder to find df containing scraped leg links
        save_dir (Path): Folder to save df updated with more resources links
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    session = scrape.get_session_w_retry()
    logging.info(
        "--------- Starting More Resources scrape ---------\n This may take a while!"
    )

    for file in raw_data_dir.glob("*.csv"):
        if not os.path.exists(save_dir / f"{file}_more_res.csv"):
            logging.info("-- working on %s --", (file))
            urls = pd.read_csv(file)
            urls.loc[:, ["IAs", "PIRs", "other_resources"]] = urls.swifter.apply(
                lambda row: scrape.get_resource_links(row.link, row.name, session),
                axis=1,
                result_type="expand",
            )
            process.write_csv(
                df=urls, path=save_dir / f"{file.name.split('.csv')[0]}_more_res.csv"
            )


def get_review_clause_text(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Extract review clauses from the main body of text for each leg

    Args:
        df (pd.DataFrame): df holding information for each leg
        config (dict): Dict holding config settings, including review clause
                       terms to search for in the text each leg

    Returns:
        pd.DataFrame: Updated df with 'review_clause_text' column
    """
    checker = InputChecker()
    search_terms = config["text_review_clause_extraction"]["search_terms"]
    checker.check_search_terms(search_terms, "text_review_clause_search_terms")
    session = scrape.get_session_w_retry()
    text_util = TextManipulator()
    logging.info(
        "--------- Starting Legislation Text Review clause extraction ---------"
    )
    logging.info("-- Working on %s pieces of legislation --", len(df))

    df.loc[:, "review_clause_text"] = [[] for x in df.index]
    for x in tqdm(df.index):
        df.at[x, "review_clause_text"] = process.extract_review_clause_text(
            df.loc[x].link,
            session,
            search_terms,
            text_util,
            n_sentences=config["text_review_clause_extraction"]["n_sentences"],
        )
    return df


def get_review_clause_IA(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Extract review clause information from Impact assessments

    Args:
        df (pd.DataFrame): df holding information for each leg
        config dict):  Dict holding config settings

    Returns:
        DataFrame: input with columns holding IA review info
    """
    logging.info(
        "--------- Starting Impact Assessment Text Review clause extraction ---------"
    )
    logging.info(
        "-- Working on %s legislations with Impact assessments --",
        len(df.query("IAs.notnull()")),
    )
    checker = InputChecker()
    query_settings = checker.check_query_settings(
        config["IA_review_clause_extraction"]["query_settings"]
    )
    IAextractor = ClauseExtractor(
        qa_model=config["IA_review_clause_extraction"]["models"]["qa_model"],
        data_df=df,
        review_match_sentences=query_settings["match_sentences"],
        query_settings=query_settings,
        embedding_model=config["IA_review_clause_extraction"]["models"][
            "embedding_model"
        ],
    )
    IAextractor.process_documents()

    scanned_pdfs = df.query('IA_review_clause == "Unscrapeable PDF"')
    if not scanned_pdfs.empty and config["convert_scanned_pdfs"]:
        logging.info(
            "--------- Converting scanned pdfs for %s legislations ---------",
            len(scanned_pdfs),
        )
        imd2doc = Image2Document(data_df=scanned_pdfs)
        scanned_doc_texts = imd2doc.get_image_documents()
        IAextractor.doc_texts = scanned_doc_texts
        logging.info("-- Extracting reivew clauses from converted pdfs --")
        IAextractor.process_documents()

    return df


def process_and_save(df: pd.DataFrame, save_path: Path):
    """Process dataframe and save output as .xlsx

    Args:
        df (pd.DataFrame): final dataframe with review clauses extracted
        save_path (Path): path to save .xlsx file
    """

    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.loc[:, "link"] = [
        scrape.prepend_govuk_url(df.loc[x].link) + "/contents" for x in df.index
    ]

    df.rename(
        columns={
            "review_clause_text": "text_review_clause",
        },
        inplace=True,
    )

    df.drop(
        columns=["cossim_score", "section"],
        axis=1,
        inplace=True,
    )

    ordered_columns = [
        x
        for x in [
            "title",
            "link",
            "text_review_clause",
            "IA_review_clause",
            "IA_review_date",
            "Scanned PDF",
            "match",
            "section",
            "IAs",
            "PIRs",
            "other_resources",
        ]
        if x in df.columns
    ]
    df = df[ordered_columns]
    df = process.process_cols(df)
    logging.info("--------- Saving output file to %s ---------", save_path)
    process.write_to_excel(save_path, df)
    logging.info("---------Output file saved successfully ---------")
