import re
import queue
import threading

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


WEB_READER_TIMEOUT = 10
SEARCH_TIMEOUT = 10
DDGS_TIMEOUT = 5
WEB_READER_MAX_LENGTH = 12000
WEB_READER_USER_AGENT = (
    "Mozilla/5.0 (compatible; AI-Technical-Analyst/1.0; +https://example.com/bot)"
)

def search_web(query):
    print(f"\n    [Tool] 正在精准检索: {query}")
    if not isinstance(query, str) or not query.strip():
        return {
            "success": False,
            "source": "web",
            "error": "搜索关键词不能为空。",
        }

    result_queue = queue.Queue(maxsize=1)

    def run_search():
        try:
            result_queue.put((True, DDGS(timeout=DDGS_TIMEOUT).text(
                query,
                max_results=3,
            )))
        except Exception as error:
            result_queue.put((False, error))

    worker = threading.Thread(target=run_search, daemon=True)
    worker.start()
    worker.join(SEARCH_TIMEOUT)

    if worker.is_alive():
        return {
            "success": False,
            "source": "web",
            "error": f"搜索请求超过 {SEARCH_TIMEOUT} 秒，已超时。",
        }

    try:
        succeeded, payload = result_queue.get_nowait()
        if not succeeded:
            raise payload
        results = payload
        if not results:
            return {
                "success": False,
                "source": "web",
                "error": "未找到相关结果。",
            }
        
        context = []
        for r in results:
            title = r.get("title", "未知")
            url = r.get("href") or r.get("url", "未知")
            body = r.get("body", "无内容")
            context.append(
                f"【标题】: {title}\n【URL】: {url}\n【摘要】: {body}"
            )
        return {
            "success": True,
            "source": "web",
            "content": "\n\n".join(context),
            "metadata": {"result_count": len(context)},
        }
    except Exception as error:
        return {
            "success": False,
            "source": "web",
            "error": str(error),
        }


def read_webpage(url):
    """读取网页正文并返回适合进入 LLM 上下文的文本。"""
    if not isinstance(url, str) or not url.strip():
        return {
            "success": False,
            "source": "web_reader",
            "error": "网页 URL 不能为空。",
            "metadata": {},
        }

    url = url.strip()
    try:
        response = requests.get(
            url,
            headers={"User-Agent": WEB_READER_USER_AGENT},
            timeout=WEB_READER_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        return {
            "success": False,
            "source": "web_reader",
            "error": f"网页请求失败: {error}",
            "metadata": {"url": url},
        }

    try:
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(
            ["script", "style", "nav", "footer", "header", "aside", "form"]
        ):
            element.decompose()

        text = soup.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        text = text[:WEB_READER_MAX_LENGTH]

        if not text:
            return {
                "success": False,
                "source": "web_reader",
                "error": "网页中未提取到可读正文。",
                "metadata": {"url": url, "status_code": response.status_code},
            }

        return {
            "success": True,
            "source": "web_reader",
            "content": text,
            "metadata": {
                "url": url,
                "status_code": response.status_code,
                "content_length": len(text),
                "truncated": len(text) >= WEB_READER_MAX_LENGTH,
            },
        }
    except Exception as error:
        return {
            "success": False,
            "source": "web_reader",
            "error": f"网页解析失败: {error}",
            "metadata": {"url": url},
        }