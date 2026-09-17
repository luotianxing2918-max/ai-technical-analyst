import uuid

import chromadb
import ollama
from PyPDF2 import PdfReader


CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "local_knowledge"
EMBEDDING_MODEL = "nomic-embed-text:latest"


def extract_text_from_pdf(file_path):
	"""读取 PDF 所有页面并返回完整文本。"""
	try:
		reader = PdfReader(file_path)
		pages = []

		for page in reader.pages:
			pages.append(page.extract_text() or "")

		return "\n".join(pages)
	except Exception as error:
		print(f"[RAG] PDF 文本提取失败: {error}")
		return ""


def chunk_text(text, chunk_size=500, overlap=50):
	"""使用滑动窗口将文本切分为重叠文本块。"""
	if not text:
		return []

	if chunk_size <= 0:
		raise ValueError("chunk_size 必须大于 0")

	if overlap < 0 or overlap >= chunk_size:
		raise ValueError("overlap 必须大于等于 0 且小于 chunk_size")

	step = chunk_size - overlap
	return [
		text[start:start + chunk_size]
		for start in range(0, len(text), step)
		if text[start:start + chunk_size]
	]


def _get_collection():
	client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
	return client.get_or_create_collection(name=COLLECTION_NAME)


def build_vector_db(pdf_path):
	"""解析 PDF，生成向量并重建本地 Chroma 知识库。"""
	try:
		text = extract_text_from_pdf(pdf_path)
		chunks = chunk_text(text)

		if not chunks:
			print("[RAG] PDF 中没有可存储的文本内容。")
			return False

		collection = _get_collection()
		existing_ids = collection.get().get("ids", [])

		if existing_ids:
			collection.delete(ids=existing_ids)

		for chunk in chunks:
			embedding = ollama.embeddings(
				model=EMBEDDING_MODEL,
				prompt=chunk,
			)["embedding"]
			collection.add(
				ids=[str(uuid.uuid4())],
				embeddings=[embedding],
				documents=[chunk],
			)

		print(f"[RAG] 向量知识库构建完成，共存入 {len(chunks)} 个文本块。")
		return True
	except Exception as error:
		print(f"[RAG] 向量知识库构建失败: {error}")
		return False


def retrieve_knowledge(query, top_k=3):
	"""向量化查询并返回最相关的本地知识片段。"""
	try:
		if not query:
			return {
				"success": False,
				"source": "local_rag",
				"error": "检索关键词为空。",
			}

		if top_k <= 0:
			return {
				"success": False,
				"source": "local_rag",
				"error": "top_k 必须大于 0。",
			}

		query_embedding = ollama.embeddings(
			model=EMBEDDING_MODEL,
			prompt=query,
		)["embedding"]
		collection = _get_collection()

		if collection.count() == 0:
			return {
				"success": False,
				"source": "local_rag",
				"error": "本地知识库为空。",
			}

		results = collection.query(
			query_embeddings=[query_embedding],
			n_results=top_k,
		)
		documents = results.get("documents", [[]])[0]

		content = "\n\n".join(
			f"【本地知识片段】\n{document}"
			for document in documents
			if document
		)
		return {
			"success": bool(content),
			"source": "local_rag",
			"content": content,
			"metadata": {"top_k": top_k, "result_count": len(documents)},
			**({} if content else {"error": "未检索到本地知识片段。"}),
		}
	except Exception as error:
		print(f"[RAG] 知识检索失败: {error}")
		return {
			"success": False,
			"source": "local_rag",
			"error": str(error),
		}
if __name__ == "__main__":
	print("正在构建向量数据库...")
	build_vector_db("data/test.pdf")
	print("构建完成！正在测试检索...")
	result = retrieve_knowledge("总结一下这篇文档的核心内容")
	print(result)