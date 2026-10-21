"""Download the latest preprocessed dataset from the private B2 bucket."""

import argparse
import json
import os
from pathlib import Path

import boto3
from botocore.config import Config


REQUIRED_FILES = (
    "X_train.csv",
    "X_test.csv",
    "y_train.csv",
    "y_test.csv",
    "preprocessor.joblib",
    "preprocessing_metadata.json",
)
LATEST_MANIFEST_KEY = "datasets/preprocessed/latest.json"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    required_env = (
        "MLFLOW_S3_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "B2_BUCKET_NAME",
    )
    missing = [name for name in required_env if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"GitHub Secrets B2 belum diisi: {missing}")

    client = boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        ),
    )
    bucket = os.environ["B2_BUCKET_NAME"]
    response = client.get_object(Bucket=bucket, Key=LATEST_MANIFEST_KEY)
    manifest = json.loads(response["Body"].read())
    if manifest.get("bucket") != bucket:
        raise ValueError("Bucket di manifest latest tidak cocok dengan B2_BUCKET_NAME.")

    file_keys = manifest.get("files", {})
    missing_files = [name for name in REQUIRED_FILES if name not in file_keys]
    if missing_files:
        raise ValueError(f"Manifest preprocessing tidak lengkap: {missing_files}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES:
        client.download_file(bucket, file_keys[name], str(args.output_dir / name))

    manifest_path = args.output_dir / "b2_latest_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Downloaded preprocessing run: {manifest.get('run_id')}")
    print(f"B2 prefix: {manifest.get('prefix')}")
    print(f"Files saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
