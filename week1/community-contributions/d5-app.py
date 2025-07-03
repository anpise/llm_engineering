import os
import openai
import gradio as gr
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import json

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.Client(api_key=OPENAI_API_KEY)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

MODEL = "gpt-4o-mini"

# Step 1: Scrape and analyze website

def get_website_contents(url):
    response = requests.get(url, headers=headers)
    body = response.content
    soup = BeautifulSoup(body, 'html.parser')
    title = soup.title.string if soup.title else "No title found"
    if soup.body:
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        text = soup.body.get_text(separator="\n", strip=True)
    else:
        text = ""
    links = [link.get('href') for link in soup.find_all('a')]
    links = [link for link in links if link]
    return title, text, links

def get_links_llm(url, links):
    link_system_prompt = (
        "You are provided with a list of links found on a webpage. "
        "You are able to decide which of the links would be most relevant to include in a brochure about the company, "
        "such as links to an About page, or a Company page, or Careers/Jobs pages.\n"
        "You should respond in JSON as in this example:"
        """
        {\n    \"links\": [\n        {\"type\": \"about page\", \"url\": \"https://full.url/goes/here/about\"},\n        {\"type\": \"careers page\", \"url\": \"https://another.full.url/careers\"}\n    ]\n}\n"""
    )
    user_prompt = f"Here is the list of links on the website of {url} - please decide which of these are relevant web links for a brochure about the company, respond with the full https URL in JSON format. Do not include Terms of Service, Privacy, email links.\nLinks (some might be relative links):\n"
    user_prompt += "\n".join(links)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )
    result = response.choices[0].message.content
    print(result)
    return json.loads(result)

def get_all_details(url):
    title, text, links = get_website_contents(url)
    details = f"Landing page:\n{title}\n{text}\n"
    links_json = get_links_llm(url, links)
    relevant_links = links_json.get("links", [])
    for link in relevant_links:
        link_url = link["url"]
        link_type = link.get("type", "Relevant page")
        try:
            ltitle, ltext, _ = get_website_contents(link_url)
            details += f"\n\n{link_type}: {ltitle}\n{ltext}\n"
        except Exception as e:
            details += f"\n\n{link_type}: (Failed to fetch {link_url})\n"
    return details, relevant_links

# Step 2: Generate brochure

def generate_brochure_from_details(details, tone):
    prompt = (
        f"You are a brochure generator AI. Given the following company details, generate a detailed, engaging, and visually beautiful company brochure in markdown format. "
        f"You can use emoticons and creative formatting to make the brochure as visually appealing as possible. "
        f"The tone of the brochure should be: {tone}. "
        f"Include sections like About Us, Services/Products, Why Choose Us, and Contact Information if available. "
        f"Output in markdown format. Do not include any other text or comments in your response.\n\n"
        f"Company details:\n{details}"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=800,
    )
    reply = response.choices[0].message.content
    print(reply)
    return reply.strip().replace("```", "").replace("markdown", "").strip()

with gr.Blocks() as demo:
    gr.Markdown("""<h1 style='text-align:center;'><span id='icon'>📄✨</span>Two-Step Brochure Generator</h1>""")
    with gr.Tabs():
        with gr.TabItem("Step 1: Scrape Website"):
            website = gr.Textbox(label="Enter Company Website", placeholder="https://example.com", scale=3)
            scrape_btn = gr.Button("Scrape and Analyze Website")
            summary = gr.Textbox(label="Scraped Content Summary", lines=10)
            links_output = gr.JSON(label="Relevant Links")
            hidden_details = gr.State()
            def step1_scrape(url):
                details, relevant_links = get_all_details(url)
                return details[:2000], relevant_links, details
            scrape_btn.click(step1_scrape, inputs=website, outputs=[summary, links_output, hidden_details], show_progress=True)
        with gr.TabItem("Step 2: Generate Brochure"):
            tone = gr.Dropdown(
                label="Choose Brochure Tone",
                choices=[
                    "Professional",
                    "Casual",
                    "Funny",
                    "Inspirational",
                    "Luxury",
                    "Minimalist",
                    "Bold",
                    "Friendly",
                    "Adventurous"
                ],
                value="Professional",
                scale=1
            )
            gen_btn = gr.Button("Generate Brochure")
            output = gr.Markdown(elem_id="output-md")
            gen_btn.click(generate_brochure_from_details, inputs=[hidden_details, tone], outputs=output, show_progress=True)

demo.launch()
