import ollama
import requests
from datetime import datetime, timedelta
import json
import pandas as pd
from bs4 import BeautifulSoup
import trafilatura
from textblob import TextBlob
import time
import colorama
from colorama import Fore, Style
import streamlit as st
from visualization import plot_correlation_matrix

# Initialize colorama for colored terminal output
colorama.init()

class OllamaSearchAgent:
    """Search agent powered by Ollama for retrieving and analyzing stock news"""
    
    def __init__(self, model="llama3.1", debug_to_streamlit=False):
        """
        Initialize the Ollama search agent
        
        Args:
            model (str): Ollama model to use (default: llama3.1)
            debug_to_streamlit (bool): Whether to output debug info to Streamlit
        """
        self.model = model
        self.debug_to_streamlit = debug_to_streamlit
        print(f"{Fore.GREEN}Ollama Search Agent initialized with model: {model}{Style.RESET_ALL}")
        
    def _log(self, message, color="white"):
        """Log messages to either terminal or both terminal and Streamlit"""
        # Always log to terminal with color
        if color == "green":
            print(f"{Fore.GREEN}{message}{Style.RESET_ALL}")
        elif color == "yellow":
            print(f"{Fore.YELLOW}{message}{Style.RESET_ALL}")
        elif color == "red":
            print(f"{Fore.RED}{message}{Style.RESET_ALL}")
        elif color == "cyan":
            print(f"{Fore.CYAN}{message}{Style.RESET_ALL}")
        else:
            print(message)
        
        # Only log to Streamlit if debug mode is enabled
        if self.debug_to_streamlit:
            if color == "red":
                st.error(message)
            elif color == "yellow":
                st.warning(message)
            elif color == "green":
                st.success(message)
            elif color == "cyan":
                st.info(message)
            else:
                st.text(message)
        
    def generate(self, prompt):
        """
        Generate a response using Ollama
        
        Args:
            prompt (str): The prompt to send to Ollama
            
        Returns:
            str: The generated response
        """
        try:
            self._log("Querying Ollama...", "yellow")
            
            # Don't show an additional spinner
            response = ollama.generate(model=self.model, prompt=prompt)
                
            return response['response']
        except Exception as e:
            error_msg = f"Error generating response with Ollama: {e}"
            self._log(error_msg, "red")
            return f"Error: Could not connect to Ollama service. Make sure Ollama is running locally. Details: {e}"
    
    def search_stock_news(self, ticker, period=30):
        """
        Search for news about a stock using Ollama as a search agent
        
        Args:
            ticker (str): Stock ticker symbol
            period (int): Number of days to look back for news
            
        Returns:
            list: List of news items with metadata
        """
        self._log(f"Searching for news about {ticker} for the past {period} days...", "cyan")
        
        # Skip UI updates if debug mode is disabled
        if not self.debug_to_streamlit:
            with st.spinner("Loading..."):
                # Get company name
                company_name = self._get_company_name(ticker)
                
                # Generate search queries
                search_prompt = self._create_search_prompt(company_name, ticker, period)
                search_queries_text = self.generate(search_prompt)
                search_queries = [q.strip() for q in search_queries_text.split('\n') if q.strip()]
                
                # Collect news
                all_news = []
                for query in search_queries:
                    news_items = self._search_news(query, period)
                    all_news.extend(news_items)
                
                # Process results
                unique_news = self._remove_duplicates(all_news)
                analyzed_news = self._analyze_news_content(unique_news)
                
                return analyzed_news
        else:
            # Original implementation with detailed UI updates
            progress_container = st.container()
            with progress_container:
                status_text = st.empty()
                progress_bar = st.progress(0)
                search_results = st.expander("Search Progress Details", expanded=True)
                search_results.info(f"Starting search for {ticker} news...")
            
            # Generate search queries based on the ticker
            status_text.text("Identifying company...")
            company_name = self._get_company_name(ticker)
            
            search_prompt = self._create_search_prompt(company_name, ticker, period)
            
            status_text.text("Generating search queries...")
            progress_bar.progress(10)
                
            search_queries_text = self.generate(search_prompt)
            search_queries = [q.strip() for q in search_queries_text.split('\n') if q.strip()]
            
            self._log(f"Generated {len(search_queries)} search queries:", "green")
            for i, query in enumerate(search_queries, 1):
                self._log(f"  {i}. {query}")
                
            search_results.success(f"Generated {len(search_queries)} search queries:")
            for i, query in enumerate(search_queries, 1):
                search_results.text(f"{i}. {query}")
            progress_bar.progress(20)
            
            # Perform searches and collect news
            all_news = []
            
            total_queries = len(search_queries)
            for i, query in enumerate(search_queries, 1):
                self._log(f"Searching: {query}", "yellow")
                
                status_text.text(f"Searching ({i}/{total_queries}): {query}")
                progress_value = 20 + (i / total_queries) * 60
                progress_bar.progress(int(progress_value))
                    
                news_items = self._search_news(query, period)
                self._log(f"Found {len(news_items)} articles", "green")
                
                search_results.info(f"Query {i}: Found {len(news_items)} articles")
                if news_items:
                    news_df = pd.DataFrame(news_items)
                    search_results.dataframe(news_df[['title', 'source', 'date']])
                    
                all_news.extend(news_items)
            
            # Remove duplicates
            status_text.text("Removing duplicate articles...")
            progress_bar.progress(85)
                
            unique_news = self._remove_duplicates(all_news)
            self._log(f"Identified {len(unique_news)} unique articles after removing duplicates", "cyan")
            
            search_results.success(f"Found {len(unique_news)} unique articles after removing duplicates")
            
            # Analyze sentiment for each news item
            status_text.text("Analyzing sentiment...")
            progress_bar.progress(90)
                
            analyzed_news = self._analyze_news_content(unique_news)
            
            status_text.text("Search complete!")
            progress_bar.progress(100)
                
            # Show final results
            if analyzed_news:
                sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
                for item in analyzed_news:
                    sentiment_counts[item['sentiment_category']] += 1
                
                # Display sentiment distribution
                search_results.markdown("### Sentiment Distribution")
                col1, col2, col3 = search_results.columns(3)
                col1.metric("Positive", sentiment_counts['positive'])
                col2.metric("Neutral", sentiment_counts['neutral'])
                col3.metric("Negative", sentiment_counts['negative'])
            
            return analyzed_news
            
    def _create_search_prompt(self, company_name, ticker, period):
        """Create the search prompt for generating queries"""
        return f"""
        I need to search for recent financial news about {company_name} (stock ticker: {ticker}).
        Please provide me with 3-5 specific search queries that would help me find the most relevant
        financial news from the past {period} days. Focus on queries that would return:
        
        1. Recent earnings reports or financial results
        2. Major company announcements or events
        3. Analyst ratings or price target changes
        4. Industry trends affecting the company
        
        Format your response as a list of search queries only, one per line.
        """
    
    def _get_company_name(self, ticker):
        """Get the company name from the ticker symbol"""
        try:
            # Use Ollama to get company information
            prompt = f"What is the full company name for the stock ticker {ticker}? Reply with just the company name, nothing else."
            company_name = self.generate(prompt).strip()
            self._log(f"Identified company name: {company_name} for ticker: {ticker}", "green")
            return company_name
        except Exception as e:
            self._log(f"Error getting company name: {e}", "red")
            return ticker
    
    def _search_news(self, query, period):
        """
        Search for news based on a query
        
        Args:
            query (str): Search query
            period (int): Number of days to look back
            
        Returns:
            list: List of news items
        """
        # This is a simplified implementation
        # In a real implementation, you might use news APIs or web scraping
        try:
            # Use Ollama to generate simulated search results
            date_range = (datetime.now() - timedelta(days=period)).strftime('%Y-%m-%d')
            
            prompt = f"""
            I need you to act as a news search engine. Search for financial news articles about:
            
            "{query}"
            
            from {date_range} until today.
            
            For each relevant article you find, provide:
            1. Title
            2. Source (publication name)
            3. Date (in YYYY-MM-DD format)
            4. URL
            5. A brief 1-2 sentence summary
            
            Return the results in JSON format like this:
            [
                {{
                    "title": "Article Title",
                    "source": "Publication Name",
                    "date": "2023-04-20",
                    "url": "https://example.com/article",
                    "summary": "Brief summary of the article"
                }}
            ]
            
            Limit to 3-5 most relevant articles. Return ONLY the JSON, no other text.
            """
            
            response = self.generate(prompt)
            
            # Extract and parse JSON
            try:
                # Find JSON in the response
                json_start = response.find('[')
                json_end = response.rfind(']') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    news_items = json.loads(json_str)
                    return news_items
                else:
                    # Fallback if JSON parsing fails
                    self._log("Could not extract JSON from response", "red")
                    return []
            except json.JSONDecodeError:
                self._log(f"Failed to parse JSON response: {response[:100]}...", "red")
                if self.debug_to_streamlit:
                    st.code(response, language="json")
                return []
                
        except Exception as e:
            self._log(f"Error searching news: {e}", "red")
            return []
            
    def _remove_duplicates(self, news_items):
        """Remove duplicate news items based on title similarity"""
        unique_news = []
        titles = []
        
        for item in news_items:
            title = item.get('title', '').lower()
            is_duplicate = False
            
            for existing_title in titles:
                # Simple similarity check - can be improved
                if self._calculate_similarity(title, existing_title) > 0.7:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                titles.append(title)
                unique_news.append(item)
                
        return unique_news
        
    def _calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts (simple implementation)"""
        # Count common words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        common_words = words1.intersection(words2)
        
        if not words1 or not words2:
            return 0
            
        # Jaccard similarity
        return len(common_words) / len(words1.union(words2))
        
    def _analyze_news_content(self, news_items):
        """
        Analyze news content for sentiment
        
        Args:
            news_items (list): List of news items
            
        Returns:
            list: News items with sentiment analysis
        """
        analyzed_items = []
        
        for item in news_items:
            # Extract content if URL is provided
            text = item.get('summary', '')
            
            # Calculate sentiment
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            
            # Convert date string to proper format if needed
            date_str = item.get('date', datetime.now().strftime('%Y-%m-%d'))
            
            # Determine sentiment category
            if sentiment > 0.1:
                sentiment_category = 'positive'
            elif sentiment < -0.1:
                sentiment_category = 'negative'
            else:
                sentiment_category = 'neutral'
                
            # Add sentiment analysis
            analyzed_item = {
                'date': date_str,
                'title': item.get('title', ''),
                'source': item.get('source', 'Unknown'),
                'url': item.get('url', ''),
                'sentiment': sentiment,
                'sentiment_category': sentiment_category
            }
            
            analyzed_items.append(analyzed_item)
            
        return analyzed_items
    
    def get_custom_insights(self, ticker, query):
        """
        Get custom insights for a stock or general financial question
        
        Args:
            ticker (str): Stock ticker symbol or "GENERAL" for non-stock queries
            query (str): Custom query about the stock or a general financial question
            
        Returns:
            dict: Response with insights
        """
        if self.debug_to_streamlit:
            with st.spinner("Loading..."):
                # For general queries
                if ticker == "GENERAL":
                    st.info(f"Asking Ollama about: {query}")
                    
                    prompt = f"""
                    I need you to provide insights about the following financial question:
                    
                    "{query}"
                    
                    Please provide a comprehensive answer that includes:
                    1. Direct response to the query
                    2. Additional context or relevant information
                    3. Any notable recent developments related to this query
                    
                    Base your response on your knowledge and provide a thoughtful analysis.
                    """
                # For stock-specific queries
                else:
                    company_name = self._get_company_name(ticker)
                    st.info(f"Asking Ollama about {company_name} ({ticker})...")
                    
                    prompt = f"""
                    I need you to provide financial insights about {company_name} (ticker: {ticker}) 
                    based on the following question:
                    
                    "{query}"
                    
                    Please provide a comprehensive answer that includes:
                    1. Direct response to the query
                    2. Additional context or relevant information
                    3. Any notable recent developments related to this query
                    
                    Base your response on recent data and developments.
                    """
                
                response = self.generate(prompt)
        else:
            # Without debug info
            if ticker == "GENERAL":
                prompt = f"""
                I need you to provide insights about the following financial question:
                
                "{query}"
                
                Please provide a comprehensive answer that includes:
                1. Direct response to the query
                2. Additional context or relevant information
                3. Any notable recent developments related to this query
                
                Base your response on your knowledge and provide a thoughtful analysis.
                """
            else:
                company_name = self._get_company_name(ticker)
                
                prompt = f"""
                I need you to provide financial insights about {company_name} (ticker: {ticker}) 
                based on the following question:
                
                "{query}"
                
                Please provide a comprehensive answer that includes:
                1. Direct response to the query
                2. Additional context or relevant information
                3. Any notable recent developments related to this query
                
                Base your response on recent data and developments.
                """
            
            response = self.generate(prompt)
        
        return {
            'ticker': ticker,
            'query': query,
            'response': response,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        } 