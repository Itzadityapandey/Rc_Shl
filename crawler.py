import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Gemini API configuration
GEMINI_API_KEY = "XYZ"  # Replace with your API key from https://ai.google.dev/
GEMINI_TEXT_MODEL = "gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36"
}

def get_gemini_classification(description):
    """Use Gemini-2.0-flash to classify test_type, adaptive_support, and remote_support."""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_TEXT_MODEL}:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        prompt = f"""
        Analyze the following assessment description: "{description}"
        Output a JSON object with:
        - test_type: One of "Knowledge & Skills", "Personality & Behaviour", or "Other"
        - adaptive_support: "yes" or "no"
        - remote_support: "yes" or "no"
        """
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"response_mime_type": "application/json"}
        }
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        result = json.loads(response.json()["candidates"][0]["content"]["parts"][0]["text"])
        time.sleep(0.1)  # Respect free tier rate limits (1500 requests/hour)
        return result
    except Exception as e:
        print(f"Error with Gemini classification: {e}")
        return {
            "test_type": "N/A",
            "adaptive_support": "no",
            "remote_support": "no"
        }

def get_gemini_embedding(text):
    """Generate embedding using text-embedding-004."""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_EMBEDDING_MODEL}:embedContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": f"models/{GEMINI_EMBEDDING_MODEL}",
            "content": {"parts": [{"text": text}]}
        }
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        embedding = response.json()["embedding"]["values"]
        time.sleep(0.1)  # Respect free tier rate limits (1500 requests/day)
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

def fetch_assessment_details(assessment):
    """Fetch description, duration, and embeddings from the assessment's detail page."""
    url = assessment["url"]
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Extract description using provided HTML structure
            description_elem = soup.select_one("div.product-catalogue-training-calendar__row.typ p")
            description = description_elem.get_text(strip=True) if description_elem else "N/A"
            
            # Extract duration
            duration_info = "N/A"
            detail_sections = soup.find_all("div", class_="product-detail__section")
            for section in detail_sections:
                section_text = section.text.lower()
                if "duration" in section_text or "time" in section_text or "minutes" in section_text:
                    duration_match = re.search(r'(\d+)\s*(?:min|minute)', section_text)
                    if duration_match:
                        duration_info = f"{duration_match.group(1)} minutes"
                        break
            
            # Use Gemini to classify if fields are ambiguous
            if assessment["test_type"] == "N/A" or assessment["adaptive_support"] == "N/A" or \
               assessment["remote_support"] == "N/A":
                gemini_result = get_gemini_classification(description)
                assessment["test_type"] = gemini_result["test_type"]
                assessment["adaptive_support"] = gemini_result["adaptive_support"]
                assessment["remote_support"] = gemini_result["remote_support"]
            
            # Generate embedding for description
            embedding = get_gemini_embedding(description) if description != "N/A" else []
            
            # Update assessment
            assessment.update({
                "description": description,
                "duration": duration_info,
                "embedding": embedding
            })
        
    except Exception as e:
        print(f"Error fetching details for {url}: {e}")
        assessment.update({
            "description": "N/A",
            "duration": "N/A",
            "embedding": []
        })
    
    return assessment

def scrape_table(table):
    """Extract data from a single table."""
    assessments = []
    rows = table.find_all("tr")[1:]  # Skip header

    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 4:
            continue

        name_col = cols[0]
        name_tag = name_col.find("a")
        name = name_tag.text.strip() if name_tag else "Unknown"
        url = name_tag["href"] if name_tag and "href" in name_tag.attrs else ""

        remote_col = cols[1]
        remote_support = "yes" if remote_col.find("span", class_="catalogue__circle -yes") else "no"

        adaptive_col = cols[2]
        adaptive_support = "yes" if adaptive_col.find("span", class_="catalogue__circle -yes") else "no"

        test_type_col = cols[3]
        test_keys = test_type_col.find_all("span", class_="product-catalogue__key")
        test_type = ", ".join(key.text.strip() for key in test_keys) if test_keys else "N/A"

        assessments.append({
            "name": name,
            "url": "https://www.shl.com" + url,
            "description": "N/A",  # Updated in fetch_assessment_details
            "duration": "N/A",     # Updated in fetch_assessment_details
            "test_type": test_type,
            "remote_support": remote_support,
            "adaptive_support": adaptive_support,
            "embedding": []        # Updated in fetch_assessment_details
        })

    return assessments

def scrape_pages_for_type(type_param, max_pages, label):
    """Scrape paginated assessment tables for a given type (1 or 2)."""
    all_assessments = []
    for page_start in range(0, max_pages * 12, 12):
        url = f"{BASE_URL}?start={page_start}&type={type_param}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(f"[{label}] Failed to fetch {url}: {response.status_code}")
            break

        soup = BeautifulSoup(response.content, "html.parser")
        print(f"[{label}] Scraping: {url}")

        table = soup.find("table")
        if not table:
            print(f"[{label}] No table found, stopping.")
            break

        assessments = scrape_table(table)
        if not assessments:
            print(f"[{label}] No assessments found, stopping.")
            break

        all_assessments.extend(assessments)
        time.sleep(1)  # Respect server

    return all_assessments

def scrape_shl_catalog():
    """Main function to scrape the SHL catalog."""
    print("⚠️ Ensure scraping complies with https://www.shl.com/robots.txt")

    print("🔍 Scraping Pre-packaged Job Solutions...")
    prepackaged = scrape_pages_for_type(type_param=2, max_pages=12, label="Prepackaged")

    print("🔍 Scraping Individual Test Solutions...")
    individual = scrape_pages_for_type(type_param=1, max_pages=32, label="Individual")

    all_assessments = prepackaged + individual
    
    print(f"Found {len(all_assessments)} assessments. Fetching details...")
    
    # Fetch details in parallel, respecting Gemini free tier limits
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_assessment = {executor.submit(fetch_assessment_details, assessment): assessment 
                               for assessment in all_assessments}
        completed = 0
        for future in as_completed(future_to_assessment):
            completed += 1
            if completed % 10 == 0:
                print(f"Progress: {completed}/{len(all_assessments)} assessments processed")
    
    # Convert embeddings to string for CSV
    for assessment in all_assessments:
        assessment["embedding"] = json.dumps(assessment["embedding"])
    
    df = pd.DataFrame(all_assessments)
    return df

def save_to_csv(df, filename="shl_assessments.csv"):
    """Save DataFrame to CSV."""
    if df is not None and not df.empty:
        df.to_csv(filename, index=False)
        print(f"✅ Saved {len(df)} assessments to {filename}")
    else:
        print("❌ No data to save")

if __name__ == "__main__":
    print("🚀 Starting SHL catalog scrape...")
    df = scrape_shl_catalog()
    save_to_csv(df)

