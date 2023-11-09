import requests
from bs4 import BeautifulSoup


def get_company_name():
    # Define the Bloomberg India company search URL
    url = f"https://www.nseindia.com/get-quotes/equity?symbol=AXISBANK"

    try:
        # Send an HTTP GET request to the Bloomberg website
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            # Find the element containing the company name
            company_name_element = soup.find("h1", class_="Basic Industry")
            if company_name_element:
                return company_name_element.text.strip()
            else:
                return "Company not found"

        else:
            return "Failed to fetch data"

    except Exception as e:
        return str(e)


get_company_name()
# Input your company symbols here
# company_symbols = ["RELI", "TCS", "HDFC"]
#
# for symbol in company_symbols:
#     company_name = get_company_name(symbol)
#     print(f"{symbol}: {company_name}")
