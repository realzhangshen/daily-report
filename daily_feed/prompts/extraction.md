You are extracting structured metadata from an article for a daily tech briefing.
Return strict JSON only. No markdown fences, no explanation.

JSON schema:
{{"one_line_summary": "One sentence summarizing what this article is about", "category": "One of: ai, dev-tools, product, startup, open-source, security, science, business, design, other", "tags": ["3-5 short lowercase tags"], "importance": 3, "content_type": "One of: product_launch, technical, news, opinion, announcement, tutorial", "key_takeaway": "The single most important or novel point from this article"}}

Importance scale (1-5):
1 = Routine/minor update, low general interest
2 = Somewhat interesting but not urgent
3 = Noteworthy, worth a brief mention
4 = Important development, most readers would care
5 = Major news, industry-shifting or highly impactful

Title: {title}
Site: {site}
Author: {author}

Content:
{content}
