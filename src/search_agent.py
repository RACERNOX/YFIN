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

# Initialize colorama for colored terminal output
colorama.init()

class OllamaSearchAgent:
    """Search agent powered by Ollama for retrieving and analyzing stock news"""
    
    def __init__(self, model="llama2"):
        """
        Initialize the Ollama search agent
        
        Args:
            model (str): Ollama model to use (default: llama2)
        """
        self.model = model
        print(f"{Fore.GREEN}Ollama Search Agent initialized with model: {model}{Style.RESET_ALL}")
        
    def generate(self, prompt):
        """
        Generate a response using Ollama
        
        Args:
            prompt (str): The prompt to send to Ollama
            
        Returns:
            str: The generated response
        """
        try:
            print(f"{Fore.YELLOW}Querying Ollama...{Style.RESET_ALL}")
            response = ollama.generate(model=self.model, prompt=prompt)
            return response['response']
        except Exception as e:
            print(f"{Fore.RED}Error generating response with Ollama: {e}{Style.RESET_ALL}")
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
        print(f"{Fore.CYAN}Searching for news about {ticker} for the past {period} days...{Style.RESET_ALL}")
        
        # Generate search queries based on the ticker
        company_name = self._get_company_name(ticker)
        
        search_prompt = f"""
        I need to search for recent financial news about {company_name} (stock ticker: {ticker}).
        Please provide me with 3-5 specific search queries that would help me find the most relevant
        financial news from the past {period} days. Focus on queries that would return:
        
        1. Recent earnings reports or financial results
        2. Major company announcements or events
        3. Analyst ratings or price target changes
        4. Industry trends affecting the company
        
        Format your response as a list of search queries only, one per line.
        """
        
        # Get search queries
        search_queries_text = self.generate(search_prompt)
        search_queries = [q.strip() for q in search_queries_text.split('\n') if q.strip()]
        
        print(f"{Fore.GREEN}Generated {len(search_queries)} search queries:{Style.RESET_ALL}")
        for i, query in enumerate(search_queries, 1):
            print(f"  {i}. {query}")
        
        # Perform searches and collect news
        all_news = []
        for query in search_queries:
            print(f"{Fore.YELLOW}Searching: {query}{Style.RESET_ALL}")
            news_items = self._search_news(query, period)
            print(f"{Fore.GREEN}Found {len(news_items)} articles{Style.RESET_ALL}")
            all_news.extend(news_items)
        
        # Remove duplicates
        unique_news = self._remove_duplicates(all_news)
        print(f"{Fore.CYAN}Identified {len(unique_news)} unique articles after removing duplicates{Style.RESET_ALL}")
        
        # Analyze sentiment for each news item
        analyzed_news = self._analyze_news_content(unique_news)
        
        return analyzed_news
    
    def _get_company_name(self, ticker):
        """Get the company name from the ticker symbol"""
        try:
            # Use Ollama to get company information
            prompt = f"What is the full company name for the stock ticker {ticker}? Reply with just the company name, nothing else."
            company_name = self.generate(prompt).strip()
            print(f"{Fore.GREEN}Identified company name: {company_name} for ticker: {ticker}{Style.RESET_ALL}")
            return company_name
        except Exception as e:
            print(f"{Fore.RED}Error getting company name: {e}{Style.RESET_ALL}")
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
                    print(f"{Fore.RED}Could not extract JSON from response{Style.RESET_ALL}")
                    return []
            except json.JSONDecodeError:
                print(f"{Fore.RED}Failed to parse JSON response: {response[:100]}...{Style.RESET_ALL}")
                return []
                
        except Exception as e:
            print(f"{Fore.RED}Error searching news: {e}{Style.RESET_ALL}")
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
        Get custom insights for a stock based on a specific query
        
        Args:
            ticker (str): Stock ticker symbol
            query (str): Custom query about the stock
            
        Returns:
            dict: Response with insights
        """
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