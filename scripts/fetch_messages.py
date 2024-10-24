import argparse
import requests
import os
from dotenv import load_dotenv
import subprocess as sp
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import logging
from datetime import datetime

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class TwitchVODFetcher:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://api.twitch.tv/helix"
        self.access_token = self._get_access_token()

    def _get_access_token(self):
        auth_url = "https://id.twitch.tv/oauth2/token"
        params = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials",
        }

        response = requests.post(auth_url, params=params)
        return response.json()["access_token"]

    def _get_user_id(self, username):
        headers = {
            "Client-ID": self.client_id,
            "Authorization": f"Bearer {self.access_token}",
        }

        response = requests.get(
            f"{self.base_url}/users",
            headers=headers,
            params={"login": username},
        )

        data = response.json()["data"]
        return data[0]["id"] if data else None

    def get_vods(self, username):
        user_id = self._get_user_id(username)
        if not user_id:
            raise ValueError(f"User {username} not found")

        headers = {
            "Client-ID": self.client_id,
            "Authorization": f"Bearer {self.access_token}",
        }

        vods = []
        pagination_cursor = None

        while True:
            params = {
                "user_id": user_id,
                "type": "archive",
                "first": 100,  # Max allowed per request
            }

            if pagination_cursor:
                params["after"] = pagination_cursor

            response = requests.get(
                f"{self.base_url}/videos",
                headers=headers,
                params=params,
            )

            data = response.json()
            vods.extend(data["data"])

            pagination = data.get("pagination", {})
            if not pagination.get("cursor"):
                break

            pagination_cursor = pagination["cursor"]

        return vods


def download_and_process_vod(args):
    """Worker function for processing a single VOD"""
    id, data_dir, capture_output, threads = args
    file_path = Path(data_dir) / f"{id}.txt"

    if file_path.exists() and file_path.stat().st_size > 0:
        logger.info(f"Skipping {file_path} as it already exists and is not empty")
        return id, True, "Skipped - already exists"

    logger.info(f"Downloading chat for https://www.twitch.tv/videos/{id}")
    start_time = datetime.now()

    try:
        # Download chat
        sp.run(
            [
                "twitchdownloadercli",
                "chatdownload",
                "--id",
                id,
                "-t",
                str(mp.cpu_count() // threads),
                "--output",
                str(file_path),
                "--timestamp-format",
                "None",
                "--collision",
                "Overwrite",
                "--banner",
                "false",
            ],
            text=True,
            stdout=sp.PIPE if capture_output else None,
            stderr=sp.PIPE if capture_output else None,
            check=True,
        )

        # Process with sd
        sp.run(
            ["sd", "^[^:]*: ", "", str(file_path)],
            text=True,
            stdout=sp.PIPE if capture_output else None,
            stderr=sp.PIPE if capture_output else None,
            check=True,
        )

        duration = datetime.now() - start_time
        logger.info(f"Completed VOD {id} in {duration.total_seconds():.2f}s")
        return id, True, "Success"

    except sp.CalledProcessError as e:
        logger.error(f"Subprocess error for VOD {id}: {e.stderr}")
        return id, False, f"Error: {e.stderr}"
    except Exception as e:
        logger.error(f"Unexpected error for VOD {id}: {str(e)}")
        return id, False, f"Error: {str(e)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--username",
        "-u",
        help="Twitch username to fetch VODs for",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Capture output instead of showing it",
        required=False,
    )
    parser.add_argument(
        "--jobs",
        "-j",
        help="Number of VODs to download in parallel (default: cpu_count() // 4)",
        required=False,
        type=int,
        default=mp.cpu_count() // 4,
    )

    args = parser.parse_args()

    # Check dependencies
    for cmd in ["twitchdownloadercli", "sd"]:
        if sp.run(["which", cmd], capture_output=True).returncode != 0:
            logger.error(f"{cmd} not found. Please install it first.")
            exit(1)

    fetcher = TwitchVODFetcher(
        os.getenv("TWITCH_CLIENT_ID"),
        os.getenv("TWITCH_CLIENT_SECRET"),
    )

    logger.info(f"Fetching VODs for user {args.username}")
    vods = fetcher.get_vods(args.username)
    logger.info(f"Found {len(vods)} VODs")

    data_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", args.username)
    )

    os.makedirs(data_dir, exist_ok=True)

    # Prepare work items
    work_items = [(vod["id"], data_dir, args.quiet, args.jobs) for vod in vods]

    logger.info(f"Starting download with {args.jobs} parallel processes")
    start_time = datetime.now()

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        results = list(executor.map(download_and_process_vod, work_items))

    duration = datetime.now() - start_time
    successful = sum(1 for _, success, _ in results if success)

    logger.info(f"Download completed in {duration.total_seconds():.2f}s")
    logger.info(f"Successfully processed {successful}/{len(vods)} VODs")

    # Print errors if any
    errors = [(id, msg) for id, success, msg in results if not success]
    if errors:
        logger.error("The following VODs had errors:")
        for id, msg in errors:
            logger.error(f"VOD {id}: {msg}")


if __name__ == "__main__":
    main()
