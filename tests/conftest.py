import pickle
from pathlib import Path

import pytest

base_dir = Path(__file__).parents[1]

@pytest.fixture(scope="session")
def shared_data():
    shared_data_dict = {}
    yield shared_data_dict

    

@pytest.fixture(scope="function")
def mock_get_soup_from_session(request, mocker):
    print(base_dir)
    file_name = request.param
    with open(base_dir / f"tests/assets/{file_name}.pickle", "rb") as f:
        html = pickle.load(f)

    get_soup_from_session = mocker.patch("src.toolkit.scrape.get_soup_from_session")
    get_soup_from_session.return_value = html

    return get_soup_from_session


@pytest.fixture
def mock_pypdf_loader(mocker):
    with open(base_dir / "tests/assets/loaded_pdf.pickle", "rb") as f:
        doc = pickle.load(f)
    mock_loader_class = mocker.patch("src.toolkit.process.PyPDFLoader")
    mock_instance = mock_loader_class.return_value
    mock_instance.load_and_split.return_value = doc
    return mock_loader_class


@pytest.fixture(scope="function")
def load_html(request):
    file_name = request.param
    with open(base_dir / f"tests/assets/{file_name}.pickle", "rb") as f:
        html = pickle.load(f)
    return html


@pytest.fixture(scope="function")
def mock_image_to_string(mocker):

    mock_image_to_string = mocker.patch(
        "src.toolkit.pdf_extractor_classes.image_to_string"
    )

    mock_image_to_string.return_value = "some words"
    return
