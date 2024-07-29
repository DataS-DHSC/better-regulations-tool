"""
Scrape Module

General functions for web scraping and the UrlRetriever class

Example:
    UrlRetriever class uses a single function call to scrape urls
    after being initialized::

        retriever = UrlRetriever(url = search_url)
        urls = retriever.get_urls()

"""

import logging
import re

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from ratelimit import limits, sleep_and_retry
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from src.toolkit.util import InputChecker

# Module level attributes
ia_substrs = ["Impact Assessment", "IA"]
"""list: Subtrings to identify Impact Assesments

When scraping for PDF links on More Resources pages on legislation.gov.uk,
filenames containing any of these substring will be counted as IAs
"""

pir_substrs = ["Post Implementation Review", "Regulatory Triage Assessment", "PIR"]
"""list: Subtrings to identify Post Implementation Reviews

When scraping for PDF links on More Resources pages on legislation.gov.uk,
filenames containing any of these substring will be counted as PIRs
"""


# Scraping Utility functions
def get_session_w_retry() -> requests.Session:
    """
    Return a requests session configured to retry if response times out
    """
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


@sleep_and_retry
@limits(calls=100, period=60)
def get_soup_from_session(url: str, session: requests.Session) -> BeautifulSoup:
    """Retrieves html soup from url

    uses requests sessions to speed up multiple calls

    Args:
        url (str): url of webpage
        session (requests.Session): requests session


    Returns:
        soup: parsed html of webpage
    """
    response = session.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup


def prepend_govuk_url(input_url: str):
    """Adds legislation.gov.uk to base url

    Args:
        input_url (str): url to modify

    Returns:
        output_url: input url modified to full web address
    """
    output_url = f"https://www.legislation.gov.uk{input_url}"
    return output_url


def get_resource_links(leg_link: str, idx: int, session: requests.Session) -> dict:
    """Scrapes more resources page of legislation

    returns dict containing resources sorted into IAS, PIRs and other

    Args:
        leg_link (str): url of legislation
        idx (int): idx of legislation in df
        session (requests.Session): requests Session

    Returns:
        dict: dictionary containing sorted resources
    """
    leg_soup = get_soup_from_session(prepend_govuk_url(leg_link), session)
    try:
        more_resources_link = leg_soup.select('li[id ="legResourcesLink"]')[0].find(
            "a"
        )["href"]
        more_res_soup = get_soup_from_session(
            prepend_govuk_url(more_resources_link), session
        )
        resources_links = get_page_links(more_res_soup)

        IAs = get_matching_resources(ia_substrs, resources_links)
        PIRs = get_matching_resources(pir_substrs, resources_links)
        other = [x for x in resources_links if x not in IAs and x not in PIRs]
    except IndexError:
        logging.error("Problem finding more resources link for %s", leg_link)
        IAs = []
        PIRs = []
        other = []
    IAs = convert_empty_to_nan(IAs)
    PIRs = convert_empty_to_nan(PIRs)
    other = convert_empty_to_nan(other)
    return {"IAs": IAs, "PIRs": PIRs, "other_resources": other}


def get_page_links(soup: BeautifulSoup) -> list:
    """Retrieve all pdf links leg.gov.uk more resources page

    Args:
        soup (BeautifulSoup): html of more resources page on
            legislation.gov.uk

    Returns:
        list: list of all links pdfs and their filenames
    """
    resources_links = soup.select('a[href$=".pdf"]')
    resources_links = [
        (x.get_text(), prepend_govuk_url(x["href"])) for x in resources_links
    ]
    return resources_links


def convert_empty_to_nan(input_list):
    """Convert an empty list to a nan"""
    if not input_list:
        output = np.nan
    else:
        output = input_list
    return output


def get_matching_resources(
    key_words: list,
    resource_links: list,
) -> list:
    """return resources whose filenames contain any of the key words

    Args:
        key_words (list): list of key words to search for eg ['Assessment']
        resource_links (list): List of tuples of resource links in the
                                format (filename, url)

    Returns:
        list: subset of resource_links whose filename contain a key word
    """
    resources = [
        x for x in resource_links if any(substr in x[0].strip() for substr in key_words)
    ]
    return resources


# URL retriever class
class InvalidSearchError(Exception):
    """Exception raised by invalid search URLs"""

    pass


class UrlRetriever:
    def __init__(self, year=None, url=None):
        """Retrieves urls from legislation.gov.uk

        Can supply a custom search URL to retreive URLS for legsilation that match,
        the custom search.  If no custom search url is provided, then can
        supply a year to retrieve all legs by year

        Args:
            year (int): year being scraped. Defaults to None.
            url (str, optional): search url. Defaults to None.
        """
        if url:
            self._first_page_url = url
        else:
            self._first_page_url = (
                f"https://www.legislation.gov.uk/primary+secondary/{year}"
            )
        self.retrieved_urls = pd.DataFrame()
        self.year = year
        self.session = get_session_w_retry()
        self.next_page_link = ""
        self.page_n = 0
        input_checker = InputChecker()
        input_checker.is_str(self._first_page_url, "search_url")

    def get_urls(self):
        """Scrape urls

        Returns:
            pd.DataFrame: df with urls and details for all legs
        """
        first_page_soup = get_soup_from_session(self._first_page_url, self.session)
        if "has returned no results" in first_page_soup.get_text(" ", strip=True):
            raise InvalidSearchError(
                "Your search returned no results on legislation.gov.uk."
                f"\n Please check search terms and try again {self._first_page_url}"
            )
        self.retrieve_page_urls(first_page_soup)
        while self.next_page_link:
            self.process_next_page()
        return self.retrieved_urls

    def retrieve_page_urls(self, page_soup: BeautifulSoup):
        """From parsed html of a legs.gov.uk search page, get all legislation urls.

        Each page has a table containing titles and urls for legs matching search on
        that page

        Args:
            page_soup (BeautifulSoup): parsed html of page to scrape
        """
        page_urls = pd.DataFrame()

        page_leg_table = self.get_leg_table(page_soup)

        page_urls.loc[:, "title"] = [x.get_text() for x in page_leg_table]
        page_urls.loc[:, "link"] = [
            x["href"].split("contents")[0] for x in page_leg_table
        ]

        self.retrieved_urls = pd.concat([self.retrieved_urls, page_urls])
        self.page_n += 1
        logging.info("-- Completed scrape of page %s  --", self.page_n)

        self.next_page_link = page_soup.select('li[class="pageLink next"]')

    @staticmethod
    def get_leg_table(page_soup: BeautifulSoup) -> list:
        """From search page of legislation.gov.uk, get table containing leg info

        Args:
            page_soup (_type_): _description_

        Returns:
            list: List with all urls for legs in the page_soup
        """
        page_tables = page_soup.find_all("table")
        # select results table
        page_leg_table = [x for x in page_tables if x.find_all("a", href=True)]

        page_leg_table = page_leg_table[0].select('a[href*="contents"]')

        # filtering to get one url per legislation
        page_leg_table = [
            x
            for x in page_leg_table
            if not (
                x.has_attr("xml:lang")
                or UrlRetriever.starts_with_four_digits(x.get_text())
            )
        ]
        return page_leg_table

    @staticmethod
    def starts_with_four_digits(s):
        """Check if input is a string that starts with 4 digits"""
        return bool(re.match(r"^\d{4}", s))

    def process_next_page(self):
        """continues processing

        parse html from the next page link and continue scrape
        """
        next_page_soup = get_soup_from_session(
            prepend_govuk_url(self.next_page_link[0].select("a")[0]["href"]),
            self.session,
        )
        self.retrieve_page_urls(next_page_soup)
