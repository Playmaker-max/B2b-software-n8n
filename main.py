import os
import json
import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="B2B Lead Enrichment Engine")

client = OpenAI(
    api_key=os.environ.get("LLM_API_KEY"),
    base_url=os.environ.get("LLM_BASE_URL", "https://groq.com")
)
MODEL_NAME = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")


class AnalyzeRequest(BaseModel):
    website_url: str
    company_name: str


class CopyRequest(BaseModel):
    first_name: str
    company_name: str
    core_offer: str
    pain_point: str


async def scrape_homepage(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    try:
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as http:
            resp = await http.get(url, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            text = " ".join(soup.get_text(separator=" ").split())
            return text[:4000]
    except Exception:
        return f"Company name: {url}. Website text unavailable."


@app.post("/v1/analyze-lead")
async def analyze_lead(payload: AnalyzeRequest):
    site_text = await scrape_homepage(payload.website_url)

    prompt = f"""
    Analyze this B2B company for an outbound lead-gen/AI automation agency.
    Company: {payload.company_name}
    Website Text: {site_text}

    Return ONLY valid JSON with these exact keys:
    - "icp_score": integer from 1 to 10 (7+ if B2B service, agency, SaaS, or high-ticket consulting)
    - "core_offer": 1-sentence summary of what they sell
    - "pain_point": 1 likely operational or lead-gen bottleneck they face
    - "is_qualified": boolean (true if icp_score >= 7, else false)
    """

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/generate-copy")
async def generate_copy(payload: CopyRequest):
    prompt = f"""
    Write a hyper-personalized cold email opening line and CTA for {payload.first_name} at {payload.company_name}.
    Their Core Offer: {payload.core_offer}
    Their Bottleneck: {payload.pain_point}

    Rules: Sound like a peer, zero fluff, no "I hope this finds you well."
    Return ONLY valid JSON with these exact keys:
    - "custom_first_line": 1 sharp, specific observation about their offer/growth
    - "tailored_cta": 1 low-friction 10-word question pitching a workflow audit
    """

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.5,
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
