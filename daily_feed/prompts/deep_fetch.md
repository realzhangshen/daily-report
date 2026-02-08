You are helping decide whether to fetch additional related pages.
Return strict JSON with keys: need_deep_fetch (boolean), urls (array), rationale (string).
Do not wrap the JSON in markdown/code fences.
Choose up to 5 URLs from the candidate list if deeper context would materially improve analysis.
If not needed, set need_deep_fetch=false and urls=[].

Title: {title}
Site: {site}
Author: {author}

Candidate links:
{candidate_links}

Content (truncated):
{content}
