# Personal Financial Stock Analyzer 📈

## Project Overview

The Personal Financial Stock Analyzer is a web application that helps users analyze their stock portfolios from stock websites like Fidelity, Robinhood. It uses real-time financial data from Yahoo Finance, generates detailed stock analysis, and provides investment recommendations. The application also includes a chatbot feature that answers finance-related questions using a large language model.

## Features

1. **Stock Data Analysis**: Fetches real-time stock data and generates detailed analysis including financial performance, key metrics, valuation, growth prospects, and recent news.
2. **Performance Analysis**: Provides additional analysis on total gain/loss, percentage gain/loss, today's gain/loss, allocation, and cost basis.
3. **Risk Assessment**: Evaluates the risk category and volatility of each stock.
4. **Chatbot**: An AI assistant that provides answers to finance-related questions.

## Technologies Used

- **Python**is the main programming language.
- **Streamlit**: For building the web application.
- **Yahoo Finance**: For fetching real-time stock data.
- **Plotly**: For creating interactive plots.
- **BeautifulSoup**: For web scraping recent news articles.
- **Langchain and NVIDIA AI Endpoints**: For integrating the large language model (meta/llama3-8b-instruct) and generating text analysis.

## Detailed Methodology

![Detailed Methodology](https://github.com/sowmyakuruba20/NVIDIAGenAI_Personal_Finance_Analyst/assets/131414180/43d1348b-4dad-4d90-a6e7-2c22aebe6487)

## Desired and Required Libraries

Here are the versions of the libraries used in this project:

- `streamlit==1.10.0`
- `yfinance==0.1.70`
- `pandas==1.4.2`
- `nemoguardrails==0.9.0`
- `beautifulsoup4==4.10.0`
- `plotly==5.7.0`
- `langchain-core==0.0.38`
- `langchain-nvidia-ai-endpoints==0.2.1`
- `dotenv==1.0.1`

Hardware NVIDIA RTX 3080

## Setup and Installation

### Prerequisites

Ensure you have the following installed:

- **Python 3.7 or higher**
- **pip** (Python package installer)

### Clone the Repository

 ```sh
git clone https://github.com/sowmyakuruba20/Finance_Analyst_NVIDIA_NIMS_llama3.git
cd Finance_Analyst_NVIDIA_NIMS_llama3
```

## Create a Virtual Environment

It's recommended to create a virtual environment to manage dependencies.

```sh
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

## Install Dependencies
Install the required packages using pip.
```sh
pip install -r requirements.txt
```
## Set Up Environment Variables
Export NVIDIA_API_KEY, generate key from NVIDIA
```sh
export NVIDIA_API_KEY=‘your_nvidia_api_key_here’
```
## Run the Application
Start the Streamlit application.

```sh
streamlit run nim.py
```
## Usage
1. Upload Your Portfolio: Upload your stock portfolio CSV file by clicking on the browse file button. In my case, I have downloaded my portfolio from my Fidelity stocks account. The CSV should have the following columns: Symbol, Total Gain/Loss Dollar, Total Gain/Loss Percent, Today's Gain/Loss Dollar, Today's Gain/Loss Percent, Percent Of Account, Average Cost Basis, Last Price.
2. Analyze Stocks: Click on the respective stocks buttons displayed at the top to view the analysis of each stock. By default, the analysis of the first stock is shown.
3. Ask Questions: Use the chatbot in the sidebar to ask finance-related questions.

## Project Structure
- nim.py: The main application script.
- requirements.txt: A list of Python dependencies.
- guardrail_config.yaml: To control non-finance input and output

## Contributing
If you want to contribute to this project, feel free to create a pull request. Please ensure that your code adheres to the existing style and includes relevant tests.

## References
- Yahoo Finance API
- Streamlit Documentation
- Plotly Documentation
- BeautifulSoup Documentation
- Langchain Documentation
- NVIDIA AI Endpoints

## Video Demonstration
Here is a video demonstration of the application in action:

