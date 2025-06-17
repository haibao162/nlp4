from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

model_path = "/Users/mac/Documents/text2vec-base-chinese"
embeddings = HuggingFaceEmbeddings(model_name=model_path)

# 直接加载已存储的 Chroma 数据库
vectorstore = Chroma(
    persist_directory="./chroma_db",  # 之前保存的路径
    embedding_function=embeddings
)

query = '根据垫款退回-明细中付款单ID等于20230313001，查找付款单银行卡卡号和垫款退回明细中的客户名称'

# 再次查询
results = vectorstore.similarity_search(query, k=2)
print(results, '11')

results2 = vectorstore.similarity_search(query, k=2)
print(results2, '22')