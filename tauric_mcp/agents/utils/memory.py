from openai import OpenAI
import dashscope
from dashscope import TextEmbedding
import os
import threading
from typing import Dict, Optional

# 延迟导入 chromadb，避免在模块加载时读取环境变量
# chromadb 会在导入时读取所有环境变量并验证，导致 pydantic 验证错误


class ChromaDBManager:
    """单例ChromaDB管理器，避免并发创建集合的冲突"""

    _instance = None
    _lock = threading.Lock()
    _collections: Dict[str, any] = {}
    _client = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ChromaDBManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            # 延迟导入 chromadb，避免在模块加载时读取环境变量
            # 临时保存所有环境变量，导入 chromadb 后再恢复
            env_backup = dict(os.environ)
            
            # 只保留 chromadb 可能需要的环境变量
            chroma_env_keys = [
                'CHROMA_DISABLE_TELEMETRY',
                'ANONYMIZED_TELEMETRY',
                'CHROMA_HOST',
                'CHROMA_PORT',
                'CHROMA_PERSISTENT',
                'CHROMA_DATA_DIR',
            ]
            
            # 清除所有环境变量（除了 chromadb 相关的）
            for key in list(os.environ.keys()):
                if key not in chroma_env_keys:
                    del os.environ[key]
            
            try:
                # 现在导入 chromadb，它只会看到很少的环境变量
                import chromadb
                from chromadb.config import Settings
                
                # 使用更兼容的ChromaDB配置
                settings = Settings(
                    allow_reset=True,
                    anonymized_telemetry=False,
                    is_persistent=False
                )
                self._client = chromadb.Client(settings)
                self._initialized = True
                print("📚 [ChromaDB] 单例管理器初始化完成")
            except Exception as e:
                print(f"❌ [ChromaDB] 初始化失败: {e}")
                # 使用最简单的配置作为备用
                try:
                    import chromadb
                    self._client = chromadb.Client()
                    self._initialized = True
                    print("📚 [ChromaDB] 使用备用配置初始化完成")
                except Exception as e2:
                    print(f"❌ [ChromaDB] 备用配置也失败: {e2}")
                    self._client = None
                    self._initialized = True
            finally:
                # 恢复所有环境变量
                os.environ.clear()
                os.environ.update(env_backup)

    def get_or_create_collection(self, name: str):
        """线程安全地获取或创建集合"""
        with self._lock:
            # 检查 chromadb 是否可用
            if self._client is None:
                raise Exception("ChromaDB 不可用，内存功能已禁用")
            
            if name in self._collections:
                print(f"📚 [ChromaDB] 使用缓存集合: {name}")
                return self._collections[name]

            try:
                # 尝试获取现有集合
                collection = self._client.get_collection(name=name)
                print(f"📚 [ChromaDB] 获取现有集合: {name}")
            except Exception:
                try:
                    # 创建新集合
                    collection = self._client.create_collection(name=name)
                    print(f"📚 [ChromaDB] 创建新集合: {name}")
                except Exception as e:
                    # 可能是并发创建，再次尝试获取
                    try:
                        collection = self._client.get_collection(name=name)
                        print(f"📚 [ChromaDB] 并发创建后获取集合: {name}")
                    except Exception as final_error:
                        print(f"❌ [ChromaDB] 集合操作失败: {name}, 错误: {final_error}")
                        raise final_error

            # 缓存集合
            self._collections[name] = collection
            return collection


class FinancialSituationMemory:
    def __init__(self, name, config):
        self.config = config
        self.llm_provider = config.get("llm_provider", "openai").lower()

        # 根据LLM提供商选择嵌入模型和客户端
        if self.llm_provider == "dashscope" or self.llm_provider == "alibaba":
            self.embedding = "text-embedding-v3"
            self.client = None  # DashScope不需要OpenAI客户端
            # 设置DashScope API密钥
            dashscope_key = os.getenv('DASHSCOPE_API_KEY')
            if dashscope_key:
                dashscope.api_key = dashscope_key
        elif self.llm_provider == "deepseek":
            # 检查是否强制使用OpenAI嵌入
            force_openai = os.getenv('FORCE_OPENAI_EMBEDDING', 'false').lower() == 'true'

            if not force_openai:
                # 尝试使用阿里百炼嵌入
                dashscope_key = os.getenv('DASHSCOPE_API_KEY')
                if dashscope_key:
                    try:
                        # 测试阿里百炼是否可用
                        dashscope.api_key = dashscope_key
                        # 验证TextEmbedding可用性（不需要实际调用）
                        from dashscope import TextEmbedding
                        self.embedding = "text-embedding-v3"
                        self.client = None
                        print("💡 DeepSeek使用阿里百炼嵌入服务")
                    except Exception as e:
                        print(f"⚠️ 阿里百炼嵌入初始化失败: {e}")
                        dashscope_key = None  # 强制降级
            else:
                dashscope_key = None  # 跳过阿里百炼

            if not dashscope_key or force_openai:
                # 降级到OpenAI嵌入
                self.embedding = "text-embedding-3-small"
                openai_key = os.getenv('OPENAI_API_KEY')
                if openai_key:
                    self.client = OpenAI(
                        api_key=openai_key,
                        base_url=config.get("backend_url", "https://api.openai.com/v1")
                    )
                    print("⚠️ DeepSeek回退到OpenAI嵌入服务")
                else:
                    # 最后尝试DeepSeek自己的嵌入
                    deepseek_key = os.getenv('DEEPSEEK_API_KEY')
                    if deepseek_key:
                        try:
                            self.client = OpenAI(
                                api_key=deepseek_key,
                                base_url="https://api.deepseek.com"
                            )
                            print("💡 DeepSeek使用自己的嵌入服务")
                        except Exception as e:
                            print(f"❌ DeepSeek嵌入服务不可用: {e}")
                            # 禁用内存功能
                            self.client = "DISABLED"
                            print("🚨 内存功能已禁用，系统将继续运行但不保存历史记忆")
                    else:
                        # 禁用内存功能而不是抛出异常
                        self.client = "DISABLED"
                        print("🚨 未找到可用的嵌入服务，内存功能已禁用")
        elif self.llm_provider == "google":
            # Google AI使用阿里百炼嵌入（如果可用），否则使用OpenAI
            dashscope_key = os.getenv('DASHSCOPE_API_KEY')
            if dashscope_key:
                self.embedding = "text-embedding-v3"
                self.client = None
                dashscope.api_key = dashscope_key
                print("💡 Google AI使用阿里百炼嵌入服务")
            else:
                self.embedding = "text-embedding-3-small"
                self.client = OpenAI(base_url=config["backend_url"])
                print("⚠️ Google AI回退到OpenAI嵌入服务")
        elif config["backend_url"] == "http://localhost:11434/v1":
            self.embedding = "nomic-embed-text"
            self.client = OpenAI(base_url=config["backend_url"])
        else:
            self.embedding = "text-embedding-3-small"
            self.client = OpenAI(base_url=config["backend_url"])

        # 使用单例ChromaDB管理器
        self.chroma_manager = ChromaDBManager()
        self.situation_collection = None
        
        try:
            self.situation_collection = self.chroma_manager.get_or_create_collection(name)
        except Exception as e:
            print(f"⚠️ ChromaDB 不可用，内存功能已禁用: {e}")
            self.situation_collection = None

    def get_embedding(self, text):
        """Get embedding for a text using the configured provider"""

        if (self.llm_provider == "dashscope" or
            self.llm_provider == "alibaba" or
            (self.llm_provider == "google" and self.client is None) or
            (self.llm_provider == "deepseek" and self.client is None)):
            # 使用阿里百炼的嵌入模型
            try:
                response = TextEmbedding.call(
                    model=self.embedding,
                    input=text
                )
                if response.status_code == 200:
                    return response.output['embeddings'][0]['embedding']
                else:
                    raise Exception(f"DashScope embedding error: {response.code} - {response.message}")
            except Exception as e:
                raise Exception(f"Error getting DashScope embedding: {str(e)}")
        else:
            # 使用OpenAI兼容的嵌入模型
            if self.client is None:
                raise Exception("嵌入客户端未初始化，请检查配置")
            elif self.client == "DISABLED":
                # 内存功能已禁用，返回空向量
                print("⚠️ 内存功能已禁用，返回空向量")
                return [0.0] * 1024  # 返回1024维的零向量

            response = self.client.embeddings.create(
                model=self.embedding, input=text
            )
            return response.data[0].embedding

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice. Parameter is a list of tuples (situation, rec)"""
        
        # 检查 chromadb 是否可用
        if self.situation_collection is None:
            print("⚠️ ChromaDB 不可用，跳过添加记忆")
            return

        situations = []
        advice = []
        ids = []
        embeddings = []

        offset = self.situation_collection.count()

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + i))
            embeddings.append(self.get_embedding(situation))

        self.situation_collection.add(
            documents=situations,
            metadatas=[{"recommendation": rec} for rec in advice],
            embeddings=embeddings,
            ids=ids,
        )

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using embeddings"""
        
        # 检查 chromadb 是否可用
        if self.situation_collection is None:
            print("⚠️ ChromaDB 不可用，返回空记忆")
            return []
        
        query_embedding = self.get_embedding(current_situation)

        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_matches,
            include=["metadatas", "documents", "distances"],
        )

        matched_results = []
        for i in range(len(results["documents"][0])):
            matched_results.append(
                {
                    "matched_situation": results["documents"][0][i],
                    "recommendation": results["metadatas"][0][i]["recommendation"],
                    "similarity_score": 1 - results["distances"][0][i],
                }
            )

        return matched_results


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory()

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors 
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
