import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


class Cache:
    def __init__(
        self,
        vectorstore_path="./vectorstore/cache",
    ):
        self.vectorstore_path = vectorstore_path
        self.embeddings = OpenAIEmbeddings()

    def load_vectorstore(self):
        index_faiss = os.path.join(self.vectorstore_path, "index.faiss")
        index_pkl = os.path.join(self.vectorstore_path, "index.pkl")
        if os.path.exists(index_faiss) and os.path.exists(index_pkl):
            return FAISS.load_local(
                self.vectorstore_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            return None

    def save(self, query, answer):
        """(첫 번째 질문에 대한) 답변을 캐시로 저장"""
        self.vectorstore = self.load_vectorstore()
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_texts(
                texts=[query], metadatas=[{"answer": answer}], embedding=self.embeddings
            )
        else:
            self.vectorstore.add_texts(texts=[query], metadatas=[{"answer": answer}])
        self.vectorstore.save_local(self.vectorstore_path)

    def search(self, query):
        """유사한 과거 질문을 검색해 그 답변을 반환"""
        self.vectorstore = self.load_vectorstore()
        if self.vectorstore is None:
            return None

        docs = self.vectorstore.similarity_search_with_score(
            query=query,
            k=1,
            # 유사도 임계값은 조정이 필요함 / L2 거리이므로 값이 작을수록 유사도가 높음
            score_threshold=0.05,
        )
        if docs:
            return docs[0][0].metadata["answer"]
        else:
            return None
