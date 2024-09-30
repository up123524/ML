#Use a crawler and cohere to get the content of a webpage and then use cohere to generate a response, replace the LLM to get better results but this 
#will likely require payment to an company such as OpenAI.

import asyncio
from crawl4ai import AsyncWebCrawler
import cohere

# Initialize the cohere client with your API key
co = cohere.Client('your-cohere-api-key')

async def process_with_cohere(content):
    # Use Cohere's generate method to process the webpage content
    response = co.generate(
        model='command-xlarge-nightly',  # You can adjust the model 
        prompt=f"Your Qury Here:\n\n{content}",
        max_tokens=100
    )
    return response.generations[0].text

async def main():
    url = "Your URL here"
    
    # Fetch the webpage content asynchronously
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(url=url)
        content = result.markdown 

    # Process the fetched content using Cohere
    llm_response = await process_with_cohere(content)
    
    # Display the LLM's output
    print("LLM Output:", llm_response)

if __name__ == "__main__":
    asyncio.run(main())
