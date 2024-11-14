"""
This script downloads a nightlight satellite image from the eog website and uploads it to the "2301-09-estimating-gdp-a" 
S3 bucket. The current range is from 2012 to 2020 but can be adjusted as needed.
"""
import os
import boto3
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

def month_name_to_num(month_name):
    month_name = month_name.lower()
    month_mapping = {
        "january": "01",
        "february": "02",
        "march": "03",
        "april": "04",
        "may": "05",
        "june": "06",
        "july": "07",
        "august": "08",
        "september": "09",
        "october": 10,
        "november": 11,
        "december": 12,
    }

    return month_mapping.get(month_name, None)

def download_images(year, start_month):
    for month in ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]:
        if year == 2012 and month == "April":
            continue
        elif month_name_to_num(month) < month_name_to_num(start_month):
            continue

        month_num = month_name_to_num(month)

        # URL of the web page
        url = f"https://eogdata.mines.edu/nighttime_light/monthly_notile/v10/{year}/{year}{month_num}/vcmcfg/"
        response = requests.get(url)

        if response.status_code == 200:
            # Parse the HTML content with BeautifulSoup
            soup = BeautifulSoup(response.content, "html.parser")
            links = soup.find_all("a", href=lambda href: href and href.endswith("avg_rade9h.tif.gz"))

            for link in links:
                filename = link["href"]
                download_url = url + filename

                response = requests.get(download_url, stream=True)

                if response.status_code == 200:
                    total_size = int(response.headers.get("content-length", 0))

                    # Initialize the progress bar for download
                    with open(filename, "wb") as file, tqdm(
                        desc=filename,
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar:
                        for data in response.iter_content(chunk_size=1024):
                            file.write(data)
                            bar.update(len(data))

                    # Print message for S3 upload start
                    print(f"\nUploading {filename} to S3...")

                    # Initialize the progress bar for S3 upload
                    with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024) as bar:
                        def upload_progress(chunk):
                            bar.update(chunk)

                        # Upload the downloaded file to S3
                        s3_bucket_name = "2301-09-estimating-gdp-a"
                        s3_object_key = f"test_auto/nightlight/{year}/{filename}"

                        s3 = boto3.client("s3")
                        s3.upload_file(
                            filename,
                            s3_bucket_name,
                            s3_object_key,
                            Callback=upload_progress
                        )

                        # Remove the local file after uploading to S3
                        os.remove(filename)

                        # Print success message for S3 upload
                        print(f"\nUpload to S3 successful: {filename}")
                else:
                    print(f"Failed to download: {filename}")
        else:
            print(f"Failed to fetch the web page: {url}")

# Determine the range of years to download
start_year = 2012
end_year = 2020

start_month_dict = {2012: "January", 2013: "January", 2014: "January", 2015: "January", 2016: "January", 2017: "January", 2018: "January", 2019: "January", 2020: "January"}

for year in range(start_year, end_year + 1):
    download_images(year, start_month_dict[year])
