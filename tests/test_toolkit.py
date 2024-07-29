# general
import pickle
from pathlib import Path

# project specific
import pandas as pd
import pytest
from sentence_transformers import SentenceTransformer

from src.pipelines import get_review_clause_IA, get_review_clause_text

# custom
from src.toolkit import process, scrape
from src.toolkit.pdf_extractor_classes import ClauseExtractor, Image2Document
from src.toolkit.util import TextManipulator


base_dir = Path(__file__).parents[1]


class TestLegScrape:
    def test_prep_retreiver(slef, shared_data):
        retriever = scrape.UrlRetriever()
        shared_data["retriever"] = retriever

    @pytest.mark.parametrize(
        "mock_get_soup_from_session", ["invalid_soup"], indirect=True
    )
    def test_invalid_search_error(self, mock_get_soup_from_session, shared_data):
        retriever = shared_data["retriever"]
        with pytest.raises(scrape.InvalidSearchError):
            retriever.get_urls()

    @pytest.mark.parametrize("load_html", ["valid_first_page_soup"], indirect=True)
    def test_leg_table_retrieval(self, load_html, shared_data):
        retriever = shared_data["retriever"]

        table = retriever.get_leg_table(load_html)

        assert isinstance(table, list)
        assert table

    @pytest.mark.parametrize(
        "input_str, expected",
        [
            ("2004year", True),
            ("1990", True),
            ("01234578", True),
            ("allletters", False),
            ("l1998", False),
        ],
    )
    def test_starts_with_four_digits(self, input_str, expected, shared_data):
        retriever = shared_data["retriever"]
        assert retriever.starts_with_four_digits(input_str) == expected


class TestMoreResScrape:

    @pytest.mark.parametrize("load_html", ["valid_more_res_soup"], indirect=True)
    def test_resource_links_retrieval(self, load_html, shared_data):

        resource_links = scrape.get_page_links(load_html)
        shared_data["resource_links"] = resource_links

        assert isinstance(resource_links, list)
        assert resource_links

    def test_get_matching_resources(self, shared_data):
        matching_res = scrape.get_matching_resources(
            scrape.ia_substrs, shared_data["resource_links"]
        )
        assert matching_res
        assert isinstance(matching_res[0], tuple)


class TestReviewClauseText:

    def test_invalid_search_error(self):
        with pytest.raises(TypeError):
            get_review_clause_text(df=None, search_terms=[1, 2, 3])

    @pytest.mark.parametrize(
        "mock_get_soup_from_session", ["valid_leg_made_soup"], indirect=True
    )
    def test_extract_relevant_sections(self, mock_get_soup_from_session):

        section = process.extract_review_clause_text(
            link="example.com",
            session=None,
            search_terms=["Secretary of State for Health"],
            text_util=TextManipulator(),
            n_sentences = 4
        )

        assert isinstance(section, list)
        assert section

    def test_split_pdf_doc(self, mock_pypdf_loader):

        doc_text = process.get_pdf_text("example_url")

        assert isinstance(doc_text, str)
        assert doc_text.find("\n") == -1

    def test_convert_list_cols(self):
        df = pd.DataFrame()
        df.loc[:, "str_lists"] = [
            f"[{str(x)}]" for x in [0, "1,2,3", "", '"seven", "eight"']
        ]

        df = process.convert_list_cols(df, ["str_lists"])

        for x in df.str_lists:
            assert isinstance(x, list)

    @pytest.mark.parametrize("input_list_len", [1, 3, 7, 14])
    def test_spread_cols(self, input_list_len):
        df = pd.DataFrame()
        df.loc[:, "list_col"] = [["list"] * input_list_len for x in range(0, 5)]

        process.spread_cols(df, ["list_col"])

        assert len(df.columns) == input_list_len


# cant mock save to excel fn?


class TestReviewClauseImpactAssesment:
    def test_create_query_dict(self, shared_data):
        shared_data["query_settings"] = {
            "match_sentences": ["alist", "ofsentences"],
            "clause_query": "query",
            "date_query": "query",
            "n_sentences": 4,
            "similarity_threshold": 0.2,
        }

    @pytest.mark.parametrize(
        "query_settings_key, error_value, error_type",
        [
            ("match_sentences", [1, 2, 3], TypeError),
            ("clause_query", None, TypeError),
            ("date_query", 4, TypeError),
            ("n_sentences", "four", ValueError),
            ("similarity_threshold", 2, ValueError),
        ],
    )
    def test_query_error_raising(
        self, shared_data, query_settings_key, error_value, error_type
    ):
        query_settings = shared_data["query_settings"].copy()
        query_settings[query_settings_key] = error_value
        config = {}
        config["IA_review_clause_extraction"] ={}
        config["IA_review_clause_extraction"]['query_settings'] = query_settings
        with pytest.raises(error_type):
            get_review_clause_IA(df=pd.DataFrame(columns=["IAs"]), config=config)

    @pytest.mark.parametrize(
        "input_df, expected_output",
        [
            ({}, float),
            ({"review_clause": ["example"], "review_date": ["example"]}, str),
            (
                {
                    "review_clause": ["example1", "example2"],
                    "review_date": ["example3", "example4"],
                },
                list,
            ),
        ],
    )
    def test_condense_results(self, input_df, expected_output):
        df = pd.DataFrame(input_df)

        wording, date = ClauseExtractor._condense_results(df)

        assert isinstance(wording, expected_output)
        assert isinstance(date, expected_output)

    def test_get_max_sim_sentences(self, shared_data, mocker):

        for filename in ["doc_text", "query_embedding", "doc_sentences_embedding"]:
            with open(base_dir / f"tests/assets/{filename}.pickle", "rb") as f:
                shared_data[filename] = pickle.load(f)
        mock_encode_fn = mocker.patch.object(  # noqa: F841
            SentenceTransformer, "encode", return_value=shared_data["query_embedding"]
        )
        mocker.patch.object(
            ClauseExtractor,
            "get_embeddings",
            return_value=shared_data["doc_sentences_embedding"],
        )
        mocked_embedding_model = SentenceTransformer()
        extractor_instance = ClauseExtractor(
            qa_model=None,
            data_df=pd.DataFrame(columns=["IAs"]),
            review_match_sentences=shared_data["query_settings"]["match_sentences"],
            query_settings=shared_data["query_settings"],
            embedding_model=None,
        )
        extractor_instance.embedding_model = mocked_embedding_model

        idx, score = extractor_instance.find_sim_index(
            "",
            extractor_instance.text_util.split_into_sentences(shared_data["doc_text"]),
            shared_data["doc_text"],
        )

        assert isinstance(idx, int)
        assert idx == 2586
        assert isinstance(score, float)
        extractor_instance.embedding_model.encode.assert_called_once()
        extractor_instance.get_embeddings.assert_called_once()


class TestImage2Doc:
    # colud also try to mock the pdc savein gas image? but harder to do
    def test_prep_create_class(self, shared_data):
        shared_data["img2doc"] = Image2Document(data_df=pd.DataFrame())

    def test_image_to_pdf(self, shared_data, mock_image_to_string):
        with open(base_dir / "tests/assets/exmaple_pdf_images.pickle", "rb") as f:
            images = pickle.load(f)

        doc = shared_data["img2doc"].convert_image_to_text(images)

        assert isinstance(doc, str)


class TestTextUtil:
    def test_prep_create_class(self, shared_data):
        shared_data["text_util"] = TextManipulator()

    @pytest.mark.parametrize(
        "term, expected_outcome", [("review", 4), ("options", 7), ("NHS", 37)]
    )
    def test_find_all_matches(self, shared_data, term, expected_outcome):

        n_matches = shared_data["text_util"].find_all_matches(
            shared_data["doc_text"], [term]
        )

        assert isinstance(n_matches, list)
        assert len(n_matches) == expected_outcome

    def test_split_into_sentences(self, shared_data):

        sentences = shared_data["text_util"].split_into_sentences(
            shared_data["doc_text"]
        )

        assert isinstance(sentences, list)
        assert sentences

        shared_data["sentences"] = sentences

    def test_find_sentence_starts(self, shared_data):

        sentence_starts = shared_data["text_util"].find_sentence_starts(
            shared_data["doc_text"], shared_data["sentences"]
        )

        assert isinstance(sentence_starts, list)
        assert len(sentence_starts) == len(shared_data["sentences"])

        shared_data["sentence_starts"] = sentence_starts

    @pytest.mark.parametrize(
        "idx, expected_outcome",
        [(1, 228), (500, 536), (31938, 31936)],  # low value  # middle value
    )  # greater than last index
    def test_find_closest_index(self, shared_data, idx, expected_outcome):

        closest_idx = shared_data["text_util"].find_closest_index(
            shared_data["sentence_starts"], idx
        )

        assert isinstance(closest_idx, int)
        assert closest_idx == expected_outcome

    @pytest.mark.parametrize(
        "idx, expected_outcome",
        [(1, 4), (500, 4), (31938, 1)],  # low value  # middle value
    )  # greater than last index
    def test_get_section(self, shared_data, idx, expected_outcome):

        section = shared_data["text_util"].get_section(
            shared_data["doc_text"], shared_data["sentences"], idx, n_sentence=4
        )

        assert isinstance(section, str)
        assert (
            len(shared_data["text_util"].split_into_sentences(section.strip()))
            == expected_outcome
        )

    def test_correct_spacing(self, shared_data):
        input_text = "a li st of sp lit word s"

        corrected = shared_data["text_util"].correct_spacing(input_text)

        assert len(corrected.split(" ")) == 5
