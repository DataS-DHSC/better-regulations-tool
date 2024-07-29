# %%

import datetime as dt
import logging
from pathlib import Path

import yaml
from tqdm import tqdm

from src.pipelines import run_review_clause_scrape
import src.toolkit.util as util

tqdm.pandas()


# %%
def main():

    base_dir = Path(__file__).parents[0]
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    
    config_path = base_dir / "inputs/configs/example_config.yml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["timestamp"] = timestamp
    run_name = config["run_name"]

    util.set_up_logging( base_dir / f"outputs/{run_name}")
  
    logging.info("---------Starting run %s ---------", timestamp)

    run_review_clause_scrape(config, base_dir)

    with open(
        base_dir / f"outputs/{run_name}/{timestamp}_config.yml",
        "w",
        encoding="utf-8",
    ) as outfile:
        yaml.dump(config, outfile)

    logging.info("Finished")


if __name__ == "__main__":
    main()
