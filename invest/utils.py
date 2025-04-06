import requests, pdb, json, os, re, copy
from datetime import datetime
from ts_data_struct import BiHashList
from openai import OpenAI

FINANCIAL_KEY = "1e347f859bc1eaa56334ad8c5dc10924"
OPENAI_KEY = "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"

def find_file_in_dir(dir, reg_pattern):
    files_and_dirs = os.listdir(dir)
    regex = re.compile(reg_pattern)
    res = []
    for file in files_and_dirs:
        if regex.match(file):
            res.append(file)
    return res 

def read_json_file(file_path):
    """
    Reads a JSON file and returns its content.

    Args:
        file_path: The path to the JSON file.

    Returns:
        The content of the JSON file.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def find_closest_datetime(datetime_list, target_datetime):
    """
    Finds the closest datetime to a target datetime in a list.

    Args:
        datetime_list: A list of datetime objects.
        target_datetime: The datetime object to find the closest to.

    Returns:
        The closest datetime object in the list to the target datetime, or None if the list is empty.
    """
    """
    # Example usage:
    dates = [
        datetime(2025, 3, 20),
        datetime(2025, 3, 22),
        datetime(2025, 3, 25),
        datetime(2025, 3, 28),
    ]
    target_date = datetime(2025, 3, 23)

    closest_date = find_closest_datetime(dates, target_date)
    print(f"The closest date to {target_date} is {closest_date}") # Output: The closest date to 2025-03-23 00:00:00 is 2025-03-22 00:00:00"
    """

    if not datetime_list:
        return None

    return min(datetime_list, key=lambda x: abs(x - target_datetime))

def get_finance_api_data(url, max_retries=3, wait_time=5):
    retries = 0

    while retries < max_retries:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(f"{url}&apikey={FINANCIAL_KEY}",headers=headers)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"Rate limit hit! Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2
                retries += 1
            else:
                print(f"HTTP Error {e.response.status_code}: {e.response.reason}")
                break

        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            break

        except Exception as e:
            print(f"Error: {e}")
            break

    return None

def build_price_volume_chart_data(stock_list, start_date, end_date, url_str):
    D = {}
    count = 0
    full_count = len(stock_list)
    for item in stock_list:
        symbol = item['symbol']
        count += 1
        print(f'{count}/{str(full_count)} Requesting price/volume data for {url_str} chart from {start_date} to {end_date} for {symbol} ...')
        url = f'https://financialmodelingprep.com/stable/{url_str}?symbol={symbol}&from={start_date}&to={end_date}'
        res = get_finance_api_data(url=url)
        if not res:
            print(f"Failed to get data for {symbol}")
            continue
        res.reverse()
        prices = BiHashList()
        volumes = BiHashList()
        for item in res:
            price_key = 'price' if 'light in url_str' else 'close'
            prices.append(item['date'], item[price_key])
            volumes.append(item['date'], item['volume'])
        D[symbol] = {'prices':prices, 'volumes':volumes}
    return D

def get_news_full_string_ticker(tickers, from_date, to_date, page_limit=3):
    text_str = ""
    for page in range(page_limit):
        url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={tickers}&page={page+1}&from={from_date}&to={to_date}"
        res = get_finance_api_data(url)
        additional_str = ""
        if res: 
            for r in res:
                #print(r["publishedDate"])
                #print(len(r["text"].split(" ")))
                additional_str += " published date:"+r["publishedDate"]+" title: "+r["title"]+" text: "+r["text"]
            text_str += additional_str 
    return text_str

#def get_openai_embedding(url, ): 

def get_openai_embedding(text_str, max_retries=3, wait_time=5):
    client = OpenAI(api_key=OPENAI_KEY)
    max_char_len = 30000 #49152
    if len(text_str) > max_char_len:
        text_str = copy.deepcopy(text_str[:max_char_len])
    retries = 0
    while retries < max_retries:
        try:
            response = client.embeddings.create(
                input=text_str,
                model="text-embedding-3-large"
            )
            emb = response.data[0].embedding
            return emb 

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"Rate limit hit! Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2
                retries += 1
            else:
                print(f"HTTP Error {e.response.status_code}: {e.response.reason}")
                break

        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            break

        except Exception as e:
            print(f"Error: {e}")
            break
    default_vec = [0.0 for i in range(3072)]
    return default_vec

def get_news_embedding(tickers, from_date, to_date, page_limit=3):
    print(f"getting news embedding for ticker {tickers}, from date: {from_date}, to date: {to_date}")
    text_str = get_news_full_string_ticker(tickers, from_date, to_date, page_limit=3)
    res = get_openai_embedding(text_str)
    return res 

