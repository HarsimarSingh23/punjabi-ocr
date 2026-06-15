"""Optional OCR-text cleanup through OpenAI or Azure OpenAI chat completions."""

import httpx
from fastapi import HTTPException

SYSTEM_PROMPT = (
    "You are an expert Punjabi (Gurmukhi script) proofreader. The user gives you raw "
    "OCR output. Fix obvious OCR mistakes — broken matras, wrong conjuncts, stray "
    "punctuation, bad spacing — while keeping the original wording and line breaks. "
    "Return only the corrected text, with no commentary."
)


async def refine_text(text: str, settings: dict[str, str]) -> str:
    provider = (settings.get("ai_provider") or "").lower()
    if provider == "openai":
        url = "https://api.openai.com/v1/chat/completions"
        key = settings.get("openai_api_key")
        if not key:
            raise HTTPException(400, "OpenAI API key is not set. Configure it in the Admin portal.")
        headers = {"Authorization": f"Bearer {key}"}
        params = {}
        model = settings.get("openai_model") or "gpt-4o-mini"
    elif provider == "azure":
        endpoint = (settings.get("azure_endpoint") or "").rstrip("/")
        key = settings.get("azure_api_key")
        deployment = settings.get("azure_deployment")
        if not (endpoint and key and deployment):
            raise HTTPException(
                400,
                "Azure OpenAI needs an endpoint, API key and deployment name. "
                "Configure them in the Admin portal.",
            )
        headers = {"api-key": key}
        api_version = (settings.get("azure_api_version") or "").strip()
        if api_version:
            # legacy per-deployment route, for older Azure OpenAI resources
            url = f"{endpoint}/openai/deployments/{deployment}/chat/completions"
            params = {"api-version": api_version}
            model = None
        else:
            # modern v1 route — works with current deployments, no api-version
            url = f"{endpoint}/openai/v1/chat/completions"
            params = {}
            model = deployment
    else:
        raise HTTPException(
            400,
            "No AI provider configured. Choose OpenAI or Azure OpenAI in the Admin portal.",
        )

    # no temperature: some models (o-series, gpt-5 family) reject non-default values
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    }
    if model:
        payload["model"] = model

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, params=params, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Could not reach the AI provider: {exc}") from exc

    try:
        body = resp.json()
    except ValueError:
        body = {}
    if resp.status_code != 200:
        msg = body.get("error", {}).get("message", resp.text[:300])
        raise HTTPException(502, f"AI provider error: {msg}")

    try:
        return body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise HTTPException(502, "AI provider returned an unexpected response.") from exc
