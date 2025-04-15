import requests
from bs4 import BeautifulSoup
import trafilatura
import streamlit as st
import ollama
import time
from colorama import Fore, Style

class OllamaWebSearcher:
    """Web search capability for Ollama models"""
    
    def __init__(self, model="llama3.1", debug_to_streamlit=False):
        """
        Initialize the Ollama web searcher
        
        Args:
            model (str): Ollama model to use
            debug_to_streamlit (bool): Whether to show debug info in Streamlit
        """
        self.model = model
        self.debug_to_streamlit = debug_to_streamlit
        
    def _log(self, message, color="white"):
        """Log messages to terminal and optionally to Streamlit"""
        # Terminal logging with color
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
        
        # Streamlit logging if enabled
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
        """Generate a response from Ollama"""
        try:
            self._log("Querying Ollama...", "yellow")
            
            # Don't show an additional spinner since we already have an outer one
            response = ollama.generate(model=self.model, prompt=prompt)
                
            return response['response']
        except Exception as e:
            error_msg = f"Error generating response with Ollama: {e}"
            self._log(error_msg, "red")
            return f"Error: Could not connect to Ollama service. Make sure Ollama is running locally. Details: {e}"
    
    def search_needed(self, query):
        """Determine if web search is needed for this query"""
        prompt = f"""
        You are not an AI assistant. Your only task is to decide if the following financial question 
        requires web search to be answered accurately with recent information:
        
        "{query}"
        
        If searching the web for current information would be necessary to answer this question accurately, 
        respond with "True". Otherwise, respond with "False".
        
        Only generate "True" or "False" as a response, no other text.
        """
        
        response = self.generate(prompt).strip().lower()
        return "true" in response
    
    def create_search_query(self, query):
        """Create an effective search query for the given question"""
        prompt = f"""
        You are an AI web search query generator. You will be given a financial question, and your task
        is to generate the best possible search query for a search engine to find recent and relevant information.
        
        The financial question is: "{query}"
        
        Generate only the search query text that should be entered in a search engine, nothing else.
        Keep the query concise and focused on getting the most relevant results.
        """
        
        search_query = self.generate(prompt).strip()
        self._log(f"Generated search query: {search_query}", "cyan")
        
        # Remove quotes if present
        if search_query[0] == '"' and search_query[-1] == '"':
            search_query = search_query[1:-1]
            
        return search_query
    
    def duckduckgo_search(self, query):
        """Search DuckDuckGo for the given query"""
        self._log(f"Searching DuckDuckGo for: {query}", "cyan")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        url = f'https://html.duckduckgo.com/html/?q={query}'
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            for i, result in enumerate(soup.find_all('div', class_='result'), start=0):
                if i >= 10:  # Limit to 10 results
                    break
                    
                title_tag = result.find('a', class_='result__a')
                if not title_tag:
                    continue
                    
                title = title_tag.text.strip()
                href = title_tag.get('href', '')
                
                # Extract actual URL from DuckDuckGo redirect URL
                if href and 'duckduckgo.com/l/?uddg=' in href:
                    start_idx = href.find('uddg=') + 5
                    end_idx = href.find('&', start_idx)
                    if end_idx > start_idx:
                        href = href[start_idx:end_idx]
                
                snippet_tag = result.find('a', class_='result__snippet')
                snippet = snippet_tag.text.strip() if snippet_tag else 'No description available'
                
                results.append({
                    'id': i,
                    'title': title,
                    'link': href,
                    'snippet': snippet
                })
            
            return results
        except Exception as e:
            self._log(f"Error searching DuckDuckGo: {e}", "red")
            return []
    
    def select_best_result(self, search_results, user_query, search_query):
        """Select the best search result for the given query"""
        if not search_results:
            return None
            
        # Format the results for Ollama
        results_text = ""
        for result in search_results:
            results_text += f"[{result['id']}] Title: {result['title']}\n"
            results_text += f"    URL: {result['link']}\n"
            results_text += f"    Snippet: {result['snippet']}\n\n"
        
        prompt = f"""
        You are an AI model trained to select the best search result for a financial question.
        
        USER QUESTION: "{user_query}"
        SEARCH QUERY USED: "{search_query}"
        
        Here are the search results:
        
        {results_text}
        
        Select the index number (0-9) of the most relevant result that would best answer the user's question.
        Respond with ONLY the index number, nothing else.
        """
        
        try:
            response = self.generate(prompt).strip()
            # Extract just the number
            index = ''.join(filter(str.isdigit, response))
            if index:
                index = int(index)
                if 0 <= index < len(search_results):
                    return index
            
            # Default to first result if there's an issue
            return 0
        except:
            return 0
    
    def scrape_webpage(self, url):
        """Scrape the content of a webpage"""
        self._log(f"Scraping content from: {url}", "cyan")
        
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                content = trafilatura.extract(downloaded, include_formatting=True, include_links=True)
                if content:
                    return content
                    
            # Fallback to BeautifulSoup if trafilatura fails
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
                
            # Get text
            text = soup.get_text(separator='\n')
            
            # Break into lines and remove leading and trailing space
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Remove blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text[:15000]  # Limit to 15K chars
        except Exception as e:
            self._log(f"Error scraping webpage: {e}", "red")
            return None
    
    def content_is_useful(self, content, user_query):
        """Check if the scraped content is useful for answering the query"""
        if not content:
            return False
            
        # Take first 10K chars to avoid context limits
        content_sample = content[:10000] + "..." if len(content) > 10000 else content
        
        prompt = f"""
        You are an AI content evaluator. Determine if the following web content is useful for answering 
        this financial question:
        
        USER QUESTION: "{user_query}"
        
        WEB CONTENT:
        {content_sample}
        
        Is this content useful and relevant for answering the question? Respond with ONLY "Yes" or "No".
        """
        
        response = self.generate(prompt).strip().lower()
        return "yes" in response
    
    def answer_with_content(self, user_query, content):
        """Generate an answer based on the web content"""
        # Take first 15K chars to avoid context limits
        content_sample = content[:15000] + "..." if len(content) > 15000 else content
        
        # Check if this is a request for brief insights
        if "exactly 4-5 brief key insights" in user_query.lower() or "bulleted list" in user_query.lower():
            prompt = f"""
            You are a financial expert AI assistant. Use the following web content to provide exactly 4-5 key insights about the stock mentioned in the query.
            
            USER QUESTION: "{user_query}"
            
            WEB CONTENT:
            {content_sample}
            
            Provide exactly 4-5 key insights that:
            1. Are the most important, recent facts about the stock
            2. Focus on price movements, company news, analyst opinions, or significant events
            3. Are brief and to the point - one sentence per insight
            4. Are formatted as a bulleted list
            
            Don't include any introduction or conclusion. Just list the 4-5 key insights in a bulleted format.
            """
        else:
            # Regular comprehensive answer
            prompt = f"""
            You are a financial expert AI assistant. Use the following web content to answer the user's question as accurately as possible.
            If the content doesn't fully answer the question, acknowledge that and provide what information you can based on the content.
            
            USER QUESTION: "{user_query}"
            
            WEB CONTENT:
            {content_sample}
            
            Provide a comprehensive answer that:
            1. Directly addresses the user's question
            2. References specific information from the web content
            3. Notes any limitations in the information available
            4. Is well-structured and easy to read
            
            Your answer should be informative and helpful for someone seeking financial information.
            """
        
        answer = self.generate(prompt)
        return answer
    
    def search_and_answer(self, query):
        """Main method to search for information and answer a query"""
        start_time = time.time()
        
        if self.debug_to_streamlit:
            progress_container = st.container()
            with progress_container:
                status = st.empty()
                progress_bar = st.progress(0)
                debug_expander = st.expander("Process Details", expanded=True)
                
            def update_progress(step, total_steps, message):
                progress_value = min(step / total_steps, 1.0)
                progress_bar.progress(progress_value)
                status.text(message)
                
            # Step 1: Check if search is needed
            update_progress(1, 6, "Loading...")
            debug_expander.info("Evaluating if this query requires web search...")
            
            need_search = self.search_needed(query)
            if need_search:
                debug_expander.success("Web search determined to be helpful for this query.")
                
                # Step 2: Generate search query
                update_progress(2, 6, "Loading...")
                debug_expander.info("Generating search query...")
                
                search_query = self.create_search_query(query)
                debug_expander.success(f"Search query: {search_query}")
                
                # Step 3: Perform search
                update_progress(3, 6, "Loading...")
                debug_expander.info("Searching DuckDuckGo...")
                
                search_results = self.duckduckgo_search(search_query)
                if search_results:
                    result_titles = "\n".join([f"- {r['title']}" for r in search_results[:5]])
                    debug_expander.success(f"Found {len(search_results)} results. Top results:\n{result_titles}")
                    
                    # Step 4: Select best result
                    update_progress(4, 6, "Loading...")
                    debug_expander.info("Selecting most relevant result...")
                    
                    best_index = self.select_best_result(search_results, query, search_query)
                    if best_index is not None and best_index < len(search_results):
                        best_result = search_results[best_index]
                        debug_expander.success(f"Selected: {best_result['title']}")
                        
                        # Step 5: Scrape webpage
                        update_progress(5, 6, "Loading...")
                        debug_expander.info(f"Retrieving content from {best_result['link']}...")
                        
                        content = self.scrape_webpage(best_result['link'])
                        if content and self.content_is_useful(content, query):
                            content_preview = content[:200] + "..." if len(content) > 200 else content
                            debug_expander.success(f"Retrieved useful content. Preview: {content_preview}")
                            
                            # Step 6: Generate answer
                            update_progress(6, 6, "Loading...")
                            debug_expander.info("Generating answer based on web content...")
                            
                            answer = self.answer_with_content(query, content)
                            
                            # Add source citation
                            answer += f"\n\nSource: [{best_result['title']}]({best_result['link']})"
                            
                            elapsed_time = time.time() - start_time
                            debug_expander.success(f"Answer generated in {elapsed_time:.2f} seconds")
                            
                            return {
                                "answer": answer,
                                "source": best_result['link'],
                                "source_title": best_result['title'],
                                "search_performed": True
                            }
                        else:
                            debug_expander.warning("Content was not useful or couldn't be retrieved.")
                    else:
                        debug_expander.warning("Could not select a valid search result.")
                else:
                    debug_expander.warning("No search results found.")
            else:
                debug_expander.info("Web search determined not necessary. Using model's knowledge.")
            
            # Fallback to model's built-in knowledge
            update_progress(6, 6, "Loading...")
            debug_expander.info("Generating answer from model's knowledge...")
            
            # Check if this is a request for brief insights
            if "exactly 4-5 brief key insights" in query.lower() or "bulleted list" in query.lower():
                prompt = f"""
                You are a financial expert AI assistant. Provide exactly 4-5 key insights about the stock mentioned in this query:
                
                "{query}"
                
                Provide exactly 4-5 key insights that:
                1. Are the most important, recent facts about the stock based on your knowledge
                2. Focus on price movements, company news, analyst opinions, or significant events
                3. Are brief and to the point - one sentence per insight
                4. Are formatted as a bulleted list
                
                Don't include any introduction or conclusion. Just list the 4-5 key insights in a bulleted format.
                
                Note: You're answering based on your existing knowledge, not real-time data.
                """
            else:
                prompt = f"""
                You are a financial expert AI assistant. Answer the following financial question using your knowledge:
                
                "{query}"
                
                Provide a comprehensive answer that:
                1. Directly addresses the question
                2. Gives relevant financial context
                3. Is well-structured and informative
                
                Note: You're answering based on your existing knowledge, not real-time data.
                """
            
            answer = self.generate(prompt)
            elapsed_time = time.time() - start_time
            debug_expander.success(f"Answer generated in {elapsed_time:.2f} seconds")
            
            return {
                "answer": answer,
                "search_performed": False
            }
        else:
            # Non-debug version with less verbose output
            with st.spinner("Loading..."):
                # Check if search is needed
                need_search = self.search_needed(query)
                
                if need_search:
                    # Generate search query
                    search_query = self.create_search_query(query)
                    
                    # Perform search
                    search_results = self.duckduckgo_search(search_query)
                    
                    if search_results:
                        # Select best result
                        best_index = self.select_best_result(search_results, query, search_query)
                        
                        if best_index is not None and best_index < len(search_results):
                            best_result = search_results[best_index]
                            
                            # Scrape webpage
                            content = self.scrape_webpage(best_result['link'])
                            
                            if content and self.content_is_useful(content, query):
                                # Generate answer
                                answer = self.answer_with_content(query, content)
                                
                                # Add source citation
                                answer += f"\n\nSource: [{best_result['title']}]({best_result['link']})"
                                
                                return {
                                    "answer": answer,
                                    "source": best_result['link'],
                                    "source_title": best_result['title'],
                                    "search_performed": True
                                }
                
                # Fallback to model's built-in knowledge
                # Check if this is a request for brief insights
                if "exactly 4-5 brief key insights" in query.lower() or "bulleted list" in query.lower():
                    prompt = f"""
                    You are a financial expert AI assistant. Provide exactly 4-5 key insights about the stock mentioned in this query:
                    
                    "{query}"
                    
                    Provide exactly 4-5 key insights that:
                    1. Are the most important, recent facts about the stock based on your knowledge
                    2. Focus on price movements, company news, analyst opinions, or significant events
                    3. Are brief and to the point - one sentence per insight
                    4. Are formatted as a bulleted list
                    
                    Don't include any introduction or conclusion. Just list the 4-5 key insights in a bulleted format.
                    
                    Note: You're answering based on your existing knowledge, not real-time data.
                    """
                else:
                    prompt = f"""
                    You are a financial expert AI assistant. Answer the following financial question using your knowledge:
                    
                    "{query}"
                    
                    Provide a comprehensive answer that:
                    1. Directly addresses the question
                    2. Gives relevant financial context
                    3. Is well-structured and informative
                    
                    Note: You're answering based on your existing knowledge, not real-time data.
                    """
                
                answer = self.generate(prompt)
                
                return {
                    "answer": answer,
                    "search_performed": False
                } 