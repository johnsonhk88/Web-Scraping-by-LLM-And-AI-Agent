# Web Scraping by LLM And AI Agents


### Introdctions
#### Business/ Use case 
Web Scraping has long been an esssentail tool for data collecting and analysis. For static website with structured data can be using traditional tools like BeautifulSoup or Scrapy are often sufficient.
However, Now, a lot of website written in Dynamic change with unstructured data which can be used LLM and AI agents to extract data more intelligently and adaptable web scraper, combining traditional scraping techniques with the power of language models



### Technology use in this project
1. HTML tags structured analysis
- use beautifulSoup or Scrapy framework for extracting
- use selenium framework for extracting html tags data 

2. LLM Model 
- try to use different open LLM models (e.g. LLama3, gemma 2) , prefer use local open LLM models(planning inference LLM model at offline in local machine)

3. AI agent
- there are several frameworks supported AI agents for Web Scraping (such as langchain, crewai ,ScrapeGraphAI)
- use AI agent automatically executes functions
- this project 


4. LLM Model evaluation
- use truLens or W&B framework for evaluation and debug LLM performance
- LLM evaluation : Content relevance, Answer Relevance, accuary, recall, precision 


5. VectorDB 
- use Vector DataBase to store the converted Document context into embedding vector
- use Vector Database can find document similarity 

6. SQL Database
- use to store the conference record
- use for query history conference record

7. FrontEnd UI
- first version will be used Streamlit for Frontend UI
- later versions will be Full stack with Backend Restful API



### Installation and Setup
1. use requirements.txt for installation package dependencies
2. you can setup virual environment by venv 
3. if you used openai for LLM model, add your openai api key to .env file  for enviroment variables 

### Run Application
- dev 

- there are several different version of web scarping application with different framework

