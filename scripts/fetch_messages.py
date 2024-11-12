from dataclasses import dataclass
from typing import Optional
import requests
import os
from dotenv import load_dotenv
import subprocess as sp
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import logging
from datetime import datetime
from colorlog import ColoredFormatter
import typer

load_dotenv()

formatter = ColoredFormatter(
    "%(log_color)s[%(levelname)s] - %(message)s",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)

handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class TwitchVODFetcher:
    def __init__(self, client_id: str, client_secret: str):
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

    def _get_user_id(self, username: str):
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

    def get_vods(self, username: str):
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


@dataclass
class Result:
    id: str
    success: bool
    message: Optional[str] = None


@dataclass
class Args:
    id: str
    data_dir: str
    hide_progress: bool
    threads: int


def download_and_process_vod(args: Args) -> Result:
    """Worker function for processing a single VOD"""
    file_path = Path(args.data_dir) / f"{args.id}.txt"

    if file_path.exists() and file_path.stat().st_size > 0:
        logger.info(f"Skipping {file_path} as it already exists and is not empty")
        return Result(args.id, True, "Skipped - already exists")

    logger.info(f"Downloading chat for https://www.twitch.tv/videos/{args.id}")
    start_time = datetime.now()

    try:
        # Download chat
        sp.run(
            [
                "twitchdownloadercli",
                "chatdownload",
                "--id",
                args.id,
                "-t",
                str(mp.cpu_count() // args.threads),
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
            stdout=sp.PIPE if not args.hide_progress else None,
            stderr=sp.PIPE if not args.hide_progress else None,
            check=True,
        )

        # Remove names (<name>: <message>)
        sp.run(
            ["sd", "^[^:]*: ", "", str(file_path)],
            text=True,
            stdout=sp.PIPE if not args.hide_progress else None,
            stderr=sp.PIPE if not args.hide_progress else None,
            check=True,
        )

        duration = datetime.now() - start_time
        logger.info(f"Completed VOD {args.id} in {duration.total_seconds():.2f}s")
        return Result(args.id, True, "Success")

    except sp.CalledProcessError as e:
        logger.error(f"Subprocess error for VOD {args.id}: {e.stderr}")
        return Result(args.id, False, f"Error: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error for VOD {args.id}: {str(e)}")
        return Result(args.id, False, f"Error: {str(e)}")


def main(
    username: str = typer.Argument(help="Twitch username to fetch VODs for"),
    show_progress: bool = typer.Option(False, help="Hide progress bar"),
    jobs: int = typer.Option(2, help="Number of VODs to download in parallel"),
    directory: Path = typer.Option(
        Path(os.getcwd()) / "data",
        help="Directory to save the downloaded VODs to",
    ),
):
    for cmd in ["twitchdownloadercli", "sd"]:
        if sp.run(["which", cmd], capture_output=True).returncode != 0:
            logger.error(f"{cmd} not found. Please install it first.")
            exit(1)

    fetcher = TwitchVODFetcher(
        os.getenv("TWITCH_CLIENT_ID"),
        os.getenv("TWITCH_CLIENT_SECRET"),
    )

    logger.info(f"Fetching VODs for user {username}")
    vods = fetcher.get_vods(username)
    logger.info(f"Found {len(vods)} VODs")

    data_dir = os.path.abspath(directory / username)

    os.makedirs(data_dir, exist_ok=True)

    work_items = [Args(vod["id"], data_dir, show_progress, jobs) for vod in vods]

    logger.info(f"Starting download with {jobs} parallel processes")
    start_time = datetime.now()

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        results = list(executor.map(download_and_process_vod, work_items))

    duration = datetime.now() - start_time
    successful = sum(1 for result in results if result.success)

    logger.info(f"Download completed in {duration.total_seconds():.2f}s")
    logger.info(f"Successfully processed {successful}/{len(vods)} VODs")

    # Print errors if any
    errors = [result for result in results if not result.success]
    if errors:
        logger.error("The following VODs had errors:")
        for result in errors:
            logger.error(f"VOD {result.id}: {result.message}")


if __name__ == "__main__":
    typer.run(main)
