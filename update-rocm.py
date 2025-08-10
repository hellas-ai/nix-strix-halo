#!/usr/bin/env python3
"""
Update script for fetching ROCm tarball information from TheRock's S3 bucket.
This script fetches the latest ROCm builds and updates the rocm-sources.json file
with URLs and SHA256 hashes for use by Nix.
"""

import json
import hashlib
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
import sys
import os
import argparse
import tempfile
from pathlib import Path


def get_s3_listing(prefix):
    """Fetch S3 bucket listing for a given prefix."""
    bucket_url = "https://therock-nightly-tarball.s3.amazonaws.com/"
    params = {"prefix": prefix}
    url = bucket_url + "?" + urllib.parse.urlencode(params)
    
    print(f"Fetching S3 listing from: {url}")
    
    with urllib.request.urlopen(url) as response:
        content = response.read().decode('utf-8')
    
    # Parse XML response
    root = ET.fromstring(content)
    namespace = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    
    files = []
    for contents in root.findall('s3:Contents', namespace):
        key = contents.find('s3:Key', namespace).text
        files.append(key)
    
    return files


def find_latest_rocm_file(target, platform="linux"):
    """Find the latest ROCm tarball for a given target."""
    # Map targets to S3 suffixes
    s3_target = target
    if target == "gfx110X":
        s3_target = f"{target}-dgpu"
    elif target == "gfx120X":
        s3_target = f"{target}-all"
    
    prefix = f"therock-dist-{platform}-{s3_target}-"
    files = get_s3_listing(prefix)
    
    if not files:
        print(f"No files found for prefix: {prefix}")
        return None, None
    
    # Sort to get the latest
    sorted_files = sorted(files)
    latest_file = sorted_files[-1] if sorted_files else None
    
    if latest_file:
        # Extract version from filename
        # Pattern: therock-dist-{platform}-{target}-{version}.tar.gz
        import re
        pattern = rf"therock-dist-{platform}-{re.escape(s3_target)}-(.+?)\.tar\.gz"
        match = re.search(pattern, latest_file)
        if match:
            version = match.group(1)
            return latest_file, version
    
    return None, None


def calculate_sha256(url, chunk_size=8192):
    """Download file and calculate SHA256 hash."""
    print(f"Downloading and hashing: {url}")
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_path = tmp_file.name
        
        # Download with progress
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            hash_sha256 = hashlib.sha256()
            
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                tmp_file.write(chunk)
                hash_sha256.update(chunk)
                downloaded += len(chunk)
                
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"  Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='\r')
        
        print()  # New line after progress
        
    # Clean up temp file
    os.unlink(tmp_path)
    
    return hash_sha256.hexdigest()


def update_sources(targets, platforms, output_file="rocm-sources.json"):
    """Update the rocm-sources.json file with latest tarball information."""
    
    sources = {}
    
    for platform in platforms:
        sources[platform] = {}
        
        for target in targets:
            print(f"\nProcessing {platform}/{target}...")
            
            # Find latest file
            filename, version = find_latest_rocm_file(target, platform)
            
            if not filename:
                print(f"  WARNING: No file found for {platform}/{target}")
                continue
            
            url = f"https://therock-nightly-tarball.s3.amazonaws.com/{filename}"
            
            # Calculate SHA256
            sha256 = calculate_sha256(url)
            
            sources[platform][target] = {
                "url": url,
                "sha256": sha256,
                "version": version,
                "filename": filename,
                "updated": datetime.utcnow().isoformat() + "Z"
            }
            
            print(f"  Version: {version}")
            print(f"  SHA256: {sha256}")
    
    # Write to JSON file
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(sources, f, indent=2)
    
    print(f"\nUpdated {output_file}")
    return sources


def main():
    parser = argparse.ArgumentParser(
        description="Update ROCm sources from TheRock S3 bucket"
    )
    parser.add_argument(
        "--targets",
        default="gfx110X,gfx1151,gfx120X",
        help="Comma-separated list of GPU targets (default: gfx110X,gfx1151,gfx120X)"
    )
    parser.add_argument(
        "--platforms",
        default="linux",
        help="Comma-separated list of platforms (default: linux)"
    )
    parser.add_argument(
        "--output",
        default="rocm-sources.json",
        help="Output JSON file (default: rocm-sources.json)"
    )
    
    args = parser.parse_args()
    
    targets = [t.strip() for t in args.targets.split(',')]
    platforms = [p.strip() for p in args.platforms.split(',')]
    
    print(f"Updating ROCm sources for:")
    print(f"  Targets: {targets}")
    print(f"  Platforms: {platforms}")
    
    try:
        sources = update_sources(targets, platforms, args.output)
        print("\nSuccess! ROCm sources updated.")
        
        # Print summary
        print("\nSummary:")
        for platform in sources:
            for target in sources[platform]:
                info = sources[platform][target]
                print(f"  {platform}/{target}: {info['version']}")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()