import os
import yfinance as yf
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import langchain.globals

# Set the verbose setting
langchain.globals.set_verbose(True)

# Get the current verbose setting
is_verbose = langchain.globals.get_verbose()

# Define the system prompt for the financial assistant
system_prompt = """You are a knowledgeable and professional financial assistant. Your role is to provide helpful advice and recommendations on personal finance topics such as budgeting, saving, investing, retirement planning, tax strategies, and more. You have access to a financial API that can provide real-time stock prices, company information, and market data. Always aim to provide actionable and practical guidance tailored to the user's specific situation."""

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "portfolio_analysis_done" not in st.session_state:
    st.session_state.portfolio_analysis_done = False
if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}
if "selected_stock" not in st.session_state:
    st.session_state.selected_stock = None

class DataFetcher:
    @staticmethod
    @st.cache_data
    def get_stock_data(symbol):
        try:
            ticker_yahoo = yf.Ticker(symbol)
            data = ticker_yahoo.history(period="5y")
            last_quote = data['Close'].iloc[-1]
            return data, f"Symbol: {symbol}\nPrice: {last_quote}\n"
        except Exception as e:
            return None, f"Error: Unable to retrieve stock data for symbol {symbol}. {str(e)}"

    @staticmethod
    @st.cache_data
    def get_financial_statements(symbol):
        try:
            stock = yf.Ticker(symbol)
            financial_statements = stock.quarterly_financials
            return financial_statements
        except Exception as e:
            return None, f"Error: Unable to retrieve financial statements for symbol {symbol}. {str(e)}"

    @staticmethod
    def google_query(search_term):
        if "news" not in search_term:
            search_term = search_term + " stock news"
        url = f"https://www.google.com/search?q={search_term}&cr=countryUS"
        url = re.sub(r"\s", "+", url)
        return url

    @staticmethod
    def get_recent_stock_news(company_name):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
        if company_name != "stock":
            g_query = DataFetcher.google_query(company_name)
        else:
            g_query = DataFetcher.google_query("top 10 recent stocks related news only")
        res = requests.get(g_query, headers=headers).text
        soup = BeautifulSoup(res, "html.parser")
        news = []
        for n in soup.find_all("div", "n0jPhd ynAwRc tNxQIb nDgy9d"):
            news.append(n.text)
        for n in soup.find_all("div", "IJl0Z"):
            news.append(n.text)

        if len(news) > 10:
            news = news[:10]
        else:
            news = news
        news_string = ""
        for i, n in enumerate(news):
            news_string += f"{i+1}. {n}\n"
        top_news = "Recent News:\n\n" + news_string

        return top_news

class Analyzer:
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", "{input}")])
        self.llm = ChatNVIDIA(model="meta/llama3-8b-instruct", base_url="https://integrate.api.nvidia.com/v1", api_key=os.getenv("NVIDIA_API_KEY"))
        self.chain = self.prompt | self.llm | StrOutputParser()

    def analyze_stock(self, ticker, stock_data, financial_statements, news):
        input_text = (
            f"Analyze the stock {ticker} based on the following information:\n\n"
            f"Stock Data: {stock_data}\n\n"
            f"Financial Statements: {financial_statements}\n\n"
            f"Recent News: {news}\n\n"
            "Give a detailed stock analysis. Use the available data and provide an investment recommendation.\n\n"
            "Based on your analysis, if the stock needs to be sold, add the following line - Sell the Stock at the end of response\n\n"
            "If the stock is supposed to be held based on the analysis, add the following line - Hold the Stock at the end of response.\n\n"
            "The user is fully aware of the investment risk, so you don't need to include any kind of warning in the answer.\n\n"
            "Make sure to include only these sections: Overview - Financial Performance - Key Metrics - Valuation - Growth Prospects - Recent News - Investment Recommendation - Hold/Sell.\n\n"
            "Please use the following example as the output format for all the tickers to maintain consistency:\n\n"
            "Example:\n"
            "Analysis for MICROSOFT CORP: \n"
            "Microsoft Corporation (MSFT) is a multinational technology company that develops, manufactures, licenses, and supports a wide range of software products, services, and devices. The company is a leader in the technology industry and has a diverse product portfolio, including operating systems, productivity software, business software, and gaming consoles.\n\n"
            "Financial Performance\n"
            "Microsoft's financial performance has been strong, with consistent revenue growth over the past few years. The company's total revenue has increased from $528.57 billion in 2023 to $618.58 billion in 2024, representing a growth rate of 17%. The gross profit margin has remained stable at around 70%, indicating a strong pricing power and cost control. The operating income has also increased from $223.52 billion in 2023 to $275.81 billion in 2024, representing a growth rate of 23%.\n\n"
            "Key Metrics : \n"
            "- EBITDA Margin: 54.2% (higher than the industry average)\n"
            "- Net Income Margin: 35.4% (higher than the industry average)\n"
            "- Return on Equity (ROE): 43.5% (higher than the industry average)\n"
            "- Debt-to-Equity Ratio: 0.65 (lower than the industry average)\n"
            "- Interest Coverage Ratio: 15.33 (higher than the industry average)\n\n"
            "Valuation : \n"
            "Microsoft's valuation ratios are slightly higher than the industry average, indicating a premium valuation. The forward price-to-earnings (P/E) ratio is 32.12, compared to the industry average of 25.49. The price-to-book (P/B) ratio is 11.34, compared to the industry average of 8.13.\n\n"
            "Growth Prospects\n"
            "Microsoft has strong growth prospects, driven by its leadership in the technology industry, innovative products, and expanding presence in emerging markets. The company's cloud computing business, Azure, continues to grow rapidly, and its gaming console business is expected to benefit from the growing demand for online gaming.\n\n"
            "Recent News : \n"
            "There has been no significant recent news that may impact the company's stock price.\n\n"
            "Investment Recommendation : \n"
            "Based on Microsoft's strong financial performance, attractive valuation, and growth prospects, I recommend HOLDING the stock. The company's consistent revenue growth, high margins, and strong return on equity indicate a strong underlying business. While the valuation is slightly higher than the industry average, I believe it is justified by the company's leadership in the technology industry and its growth prospects.\n\n"
        )
        assistant_response = ""
        response_gen = self.chain.stream({"input": input_text})
        for chunk in response_gen:
            assistant_response += chunk
        return assistant_response

    def risk_assessment(self, ticker, stock_data):
        try:
            stock_returns = stock_data['Close'].pct_change().dropna()
            volatility = stock_returns.std()

            if volatility < 0.01:
                risk_category = "Low Risk"
                color = "green"
            elif volatility < 0.02:
                risk_category = "Moderate Risk"
                color = "orange"
            else:
                risk_category = "High Risk"
                color = "red"

            return risk_category, color, volatility
        except Exception as e:
            return f"Error: Unable to calculate risk for {ticker}. {str(e)}", "gray", None

def plot_stock_trend(ticker, stock_data):
    try:
        time_periods = {
            "5 Years": stock_data.loc[stock_data.index >= (stock_data.index.max() - pd.DateOffset(years=5))],
            "1 Year": stock_data.loc[stock_data.index >= (stock_data.index.max() - pd.DateOffset(years=1))],
            "1 Month": stock_data.loc[stock_data.index >= (stock_data.index.max() - pd.DateOffset(months=1))],
            "5 Days": stock_data.loc[stock_data.index >= (stock_data.index.max() - pd.DateOffset(days=5))]
        }

        fig = make_subplots(rows=1, cols=len(time_periods), subplot_titles=[f'{key}' for key in time_periods.keys()])

        for i, (period, data) in enumerate(time_periods.items()):
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name=period), row=1, col=i+1)

        fig.update_layout(title=f'{ticker} Stock Price Trends', height=400, width=1200, showlegend=False)
        st.plotly_chart(fig)

    except Exception as e:
        st.write(f"Error: Unable to plot the trends for {ticker}. {str(e)}")

def plot_stock_trend_all(ticker):
    try:
        stock = yf.Ticker(ticker)
        time_periods = ["5y", "1y", "6mo", "1mo", "5d"]
        
        fig = go.Figure()

        for time_period in time_periods:
            data = stock.history(period=time_period)
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name=f'{time_period} trend'))

        fig.update_layout(
            title=f'{ticker} Stock Price Trends',
            xaxis_title='Date',
            yaxis_title='Stock Price',
            legend_title='Time Period',
            template='plotly_white'
        )

        st.plotly_chart(fig)

    except Exception as e:
        st.write(f"Error: Unable to plot the trends for {ticker}. {str(e)}")

def handle_user_input():
    query = st.session_state["follow_up_query"]
    if query:
        with st.spinner('Generating response...'):
            response = answer_follow_up_question(query)
            st.session_state.conversation.append({"query": query, "response": response})
        st.session_state["follow_up_query"] = ""  # Clear input box

def answer_follow_up_question(query):
    news = DataFetcher.get_recent_stock_news("stock")
    input_text = (
        f"If the query is not related to finance or investment, strictly respond with the message: "
        f"I'm sorry, but I can only provide information related to finance and stocks.\n\n"
        f"Query: {query}\n\n"
        f"Otherwise, based on the following query, provide a detailed response:\n\n{query}\n\n and strictly don't include anything from news and portfolio unless relevant"
        f"If the query is related to current/recent news only then include the following:\nRecent News: {news}\n\n"
        f"If the query is related to the portfolio stocks only then refer to the following data:\n{st.session_state.get('portfolio_df')}\n\n"
        f"Use your knowledge, the available data, and the recent news (if applicable) to provide a precise and intuitive response.\n\n"
        f"Use these examples to understand the context and how the question should be answered:\n\n"
        "EXAMPLE 1:\n"
        "Query: What is the coronavirus?\n"
        "Response: I'm sorry, but I can only provide information related to finance and stocks.\n\n"
        "EXAMPLE 2:\n"
        "Query: How did the COVID-19 pandemic impact the US stock market?\n"
        "Response: The COVID-19 pandemic had a significant impact on the US stock market. Here's a concise overview:\n"
        "Initial Crash: In February and March 2020, the US stock market experienced a rapid decline as the severity of the pandemic became apparent. The S&P 500 dropped by about 34% from its peak in February to its low in March.\n"
        "Volatility: The stock market saw extreme volatility during the early months of the pandemic, with large daily swings in stock prices due to uncertainty and panic among investors.\n"
        "Government Response: The US government and the Federal Reserve implemented substantial fiscal and monetary measures to support the economy, including stimulus packages and interest rate cuts. These actions helped stabilize the market and restore investor confidence.\n"
        "Recovery and Growth: After the initial shock, the stock market began a recovery that continued through 2020 and 2021. Technology and healthcare sectors, in particular, performed well as they were seen as benefiting from the changes brought by the pandemic.\n"
        "Sector Disparities: Not all sectors recovered equally. While technology, healthcare, and consumer discretionary sectors saw strong gains, sectors such as travel, hospitality, and energy were more negatively affected and took longer to recover.\n"
        "Long-term Changes: The pandemic accelerated trends such as remote work, e-commerce, and digital transformation, benefiting companies in these areas.\n"
        "Overall, despite the initial crash, the US stock market rebounded strongly, with indices like the S&P 500 and NASDAQ reaching new highs in the months following the onset of the pandemic.\n\n"
        "EXAMPLE 3:\n"
        "Query: What are the recent stocks in the news?\n"
        "Response: Using my financial API, I'm able to provide you with a list of recent stocks making news. Here's a rundown of some of the latest updates:\n"
        "1. GameStop Corp. (GME) - The video game retailer announced a 10.6% increase in sales, despite a decline in same-store sales, indicating a shift towards online shopping.\n"
        "2. Amazon.com, Inc. (AMZN) - The e-commerce giant saw its stock price surge after it announced a partnership with JPMorgan Chase & Co. (JPM) to launch a new online bank, marking a significant move into the financial services sector.\n"
        "3. NVIDIA Corporation (NVDA) - The graphics processing unit (GPU) manufacturer announced a new visual computing platform, focusing on artificial intelligence (AI) and deep learning applications.\n"
        "4. Chevron Corporation (CVX) - The oil and gas company reported a slight increase in revenue, despite a decline in profits, due to lower oil prices.\n"
        "5. Netflix, Inc. (NFLX) - The streaming service provider announced plans to expand its services into more international markets, which could lead to further growth.\n"
        "6. Microsoft Corporation (MSFT) - Microsoft reported quarterly earnings that beat expectations, driven by strong sales of its cloud-based services and Azure computing platform.\n"
        "7. Intel Corporation (INTC) - The chipmaker announced plans to enter the smartphone manufacturing business, shifting its focus from PC-based processors to mobile devices.\n"
        "8. Coca-Cola Company (KO) - The beverage company reported a slight decline in revenue, citing higher operating costs and increased competition in the beverage market.\n"
        "9. Tesla, Inc. (TSLA) - The electric vehicle manufacturer announced plans to increase production and capacity at its factories, anticipating strong demand for its new models.\n"
        "10. Home Depot (HD) - The home improvement retailer reported robust sales, driven by increased spending on home renovations and construction.\n"
    )

    assistant_response = ""
    response_gen = analyzer.chain.stream({"input": input_text})
    for chunk in response_gen:
        assistant_response += chunk
    return assistant_response

def display_conversation():
    conversation = st.session_state.conversation
    st.sidebar.markdown(
        """
        <style>
        .chat-message {
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }
        .user-message {
            background-color: #76B900;
            color: #000000;
        }
        .ai-message {
            background-color: #000000;
            color: #FFFFFF;
        }
        .message-divider {
            height: 1px;
            background-color: #ccc;
            margin: 10px 0;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    for message in conversation:
        user_message_html = f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {message['query']}
        </div>
        """
        ai_message_html = f"""
        <div class="chat-message ai-message">
            <strong>AI:</strong> {message['response']}
        </div>
        """
        st.sidebar.markdown(user_message_html, unsafe_allow_html=True)
        st.sidebar.markdown(ai_message_html, unsafe_allow_html=True)
        st.sidebar.markdown('<div class="message-divider"></div>', unsafe_allow_html=True)

def analyze_portfolio():
    portfolio_df = st.session_state.portfolio_df
    for index, row in portfolio_df.iterrows():
        ticker = row['Symbol']
        stock_data, stock_summary = DataFetcher.get_stock_data(ticker)
        financial_statements = DataFetcher.get_financial_statements(ticker)
        news = DataFetcher.get_recent_stock_news(ticker)
        with st.spinner('Generating analysis...'):
            analysis = analyzer.analyze_stock(ticker, stock_summary, financial_statements, news)
        st.session_state.analysis_results[ticker] = analysis

def performance_analysis(row):
    total_gain_loss = row["Total Gain/Loss Dollar"]
    percentage_gain_loss = row["Total Gain/Loss Percent"]
    todays_gain_loss_dollar = row["Today's Gain/Loss Dollar"]
    todays_gain_loss_percent = row["Today's Gain/Loss Percent"]
    percent_of_account = row["Percent Of Account"]
    average_cost_basis = row["Average Cost Basis"]
    current_price = row["Last Price"]
    total_gain_loss_percent = row["Total Gain/Loss Percent"]

    analysis = (
        f"\n"
        f"\n### Performance Analysis\n\n"
        f"- Total Gain/Loss: {total_gain_loss}\n\n"
        f"- Percentage Gain/Loss: {percentage_gain_loss}%\n\n"
        f"- Today's Gain/Loss: {todays_gain_loss_dollar} ({todays_gain_loss_percent}%)\n\n"
        f"\n### Allocation Analysis\n\n"
        f"- Percent of Account: {percent_of_account}%\n\n"
        f"\n### Cost Basis Analysis\n\n"
        f"- Average Cost Basis: {average_cost_basis}\n\n"
        f"- Current Price: {current_price}\n\n"
        f"- Total Gain/Loss Percent: {total_gain_loss_percent}%\n\n"
    )

    return analysis

def format_analysis(analysis):
    sections = analysis.split("\n\n")
    formatted_sections = []

    for section in sections:
        if section.startswith("Analysis for") or section.startswith("Overview") or section.startswith("Financial Performance") or section.startswith("Key Metrics") or section.startswith("Valuation") or section.startswith("Growth Prospects") or section.startswith("Recent News") or section.startswith("Investment Recommendation") or section.startswith("Performance Analysis") or section.startswith("Allocation Analysis") or section.startswith("Cost Basis Analysis"):
            formatted_sections.append(f"### {section}")
        else:
            formatted_sections.append(section)

    return "\n\n".join(formatted_sections)

def display_analysis_results(ticker):
    if ticker in st.session_state.analysis_results:
        analysis = st.session_state.analysis_results[ticker]
        st.write(f"**Analysis for {ticker}:**")
        formatted_analysis = format_analysis(analysis)
        st.markdown(formatted_analysis)
        st.write("---")

        row = st.session_state.portfolio_df[st.session_state.portfolio_df['Symbol'] == ticker].iloc[0]
        additional_analysis = performance_analysis(row)
        green_analysis = f"<div style='color: green;'>{additional_analysis}</div>"
        st.markdown(green_analysis, unsafe_allow_html=True)
        st.write("---")

        stock_data, _ = DataFetcher.get_stock_data(ticker)
        plot_stock_trend(ticker, stock_data)
        plot_stock_trend_all(ticker)
        st.write(f"**Risk Assessment for {ticker}:**")
        risk_category, color, volatility = analyzer.risk_assessment(ticker, stock_data)

        button_html = f"""
        <button style="background-color:{color}; color:white; border:none; padding:10px 20px; cursor:pointer;">
            {risk_category} (Volatility: {volatility:.4f})
        </button>
        """
        st.markdown(button_html, unsafe_allow_html=True)
        st.write("---")

# Existing code for main function and other functionalities
def main():
    st.set_page_config(page_title="Personal Financial Stock Analyzer", layout="wide")
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 32px;
            font-weight: bold;
            color: #76B900;
            text-align: center;
            margin-top: 20px;
        }
        .sidebar-title {
            font-size: 24px;
            font-weight: bold;
            color: #76B900;
            margin-top: 20px;
        }
        .sidebar-section {
            margin-bottom: 20px;
        }
        .stock-button {
            margin-right: 10px;
        }
        .stButton>button:hover {
            background-color: #76B900 !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        """
        <style>
        .sidebar-title {
            font-size: 24px;
            font-weight: bold;
            color: #76B900;
            margin-top: 20px;
            text-align: center;
        }
        .sidebar-input {
            width: 100%;
            padding: 10px;
            margin-top: 10px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        .stTextInput>div>input:hover {
            border-color: #76B900 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="main-title">Personal Financial Stock Analyzer</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload your stock portfolio CSV file", type="csv")
    if uploaded_file is not None:
        st.session_state.portfolio_df = pd.read_csv(uploaded_file)
        st.session_state.portfolio_analysis_done = False

    if st.session_state.portfolio_df is not None and not st.session_state.portfolio_analysis_done:
        analyze_portfolio()
        st.session_state.portfolio_analysis_done = True

    if st.session_state.portfolio_df is not None:
        tickers = st.session_state.portfolio_df['Symbol'].tolist()
        if st.session_state.selected_stock is None:
            st.session_state.selected_stock = tickers[0]

        cols = st.columns(len(tickers))
        for i, ticker in enumerate(tickers):
            if cols[i].button(ticker, key=ticker, use_container_width=True):
                st.markdown(f"""
                <style>
                .stButton button {{
                    background-color: #000000;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    font-size: 16px;
                    border-radius: 5px;
                    cursor: pointer;
                }}
                .stButton button:hover {{
                    background-color: #76B900;
                }}
                </style>
                """, unsafe_allow_html=True)
                st.session_state.selected_stock = ticker

        display_analysis_results(st.session_state.selected_stock)

    st.sidebar.markdown('<div class="sidebar-title">Learn More About Investment:</div>', unsafe_allow_html=True)
    display_conversation()
    user_query = st.sidebar.text_input(" ", key="follow_up_query", on_change=handle_user_input, placeholder="Type your question here...")

if __name__ == "__main__":
    analyzer = Analyzer()
    main()
