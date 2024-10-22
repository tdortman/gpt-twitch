import argparse
import requests
import os
from dotenv import load_dotenv
import subprocess as sp
from pathlib import Path
import multiprocessing

load_dotenv()


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

            # Check if there are more pages
            pagination = data.get("pagination", {})
            if not pagination.get("cursor"):
                break

            pagination_cursor = pagination["cursor"]

        return vods


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
        "-j",
        "--jobs",
        type=int,
        help="Number of threads to use (default all available)",
        default=multiprocessing.cpu_count(),
        required=False,
    )
    args = parser.parse_args()

    # Require twitchdownloadercli and sd to be installed
    if (
        not sp.run(["which", "twitchdownloadercli"], capture_output=True).returncode
        == 0
    ):
        print(
            "twitchdownloadercli not found. Please install it from https://github.com/lay295/TwitchDownloader/releases (NOT THE GUI)"
        )
        exit(1)

    if not sp.run(["which", "sd"], capture_output=True).returncode == 0:
        print("sd not found. Please install it from https://github.com/chmln/sd")
        exit(1)

    fetcher = TwitchVODFetcher(
        os.getenv("TWITCH_CLIENT_ID"),
        os.getenv("TWITCH_CLIENT_SECRET"),
    )
    vods = fetcher.get_vods(args.username)

    ids = map(lambda x: x["id"], vods)

    data_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", args.username)
    )

    os.makedirs(data_dir, exist_ok=True)

    capture_output = args.quiet

    for id in ids:
        file_path = Path(data_dir) / f"{id}.txt"

        if file_path.exists() and file_path.stat().st_size > 0:
            print(f"Skipping {file_path} as it already exists and is not empty")
            continue

        print(f"Downloading chat for https://www.twitch.tv/videos/{id}")

        # Download chat
        result = sp.run(
            [
                "twitchdownloadercli",
                "chatdownload",
                "--id",
                id,
                "--output",
                str(file_path),
                "-t",
                str(args.jobs),
                "--timestamp-format",
                "None",
                "--collision",
                "Overwrite",
            ],
            text=True,
            stdout=sp.PIPE if capture_output else None,
            stderr=sp.PIPE if capture_output else None,
        )

        if result.returncode != 0:
            continue

        # Process with sd
        result = sp.run(
            ["sd", "^[^:]*: ", "", str(file_path)],
            text=True,
            stdout=sp.PIPE if capture_output else None,
            stderr=sp.PIPE if capture_output else None,
        )

        if result.returncode != 0:
            print(f"Error processing chat file for VOD {id}")


if __name__ == "__main__":
    main()
