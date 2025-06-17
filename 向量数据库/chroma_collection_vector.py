import chromadb
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions

class LocalEmbedding:
    def __init__(self, model_path):
        self.model = SentenceTransformer(model_path)
    
    def __call__(self, input):
        # 返回向量列表（每个文本对应一个 float 列表）
        return self.model.encode(input, convert_to_tensor=False).tolist()


# 使用
model_path = "/Users/mac/Documents/text2vec-base-chinese"
local_embedder = LocalEmbedding(model_path)
# bf_offline_payment_form
sentences = '''
线下业务-付款单表，字段信息有：主键id、企业ID、批次号、付款单编号、付款类型,1对公,2对私、
申请类型,1常规业务,2垫付退回,3到款退回,4折扣单、业务大类ID、冗余业务大类、业务小类ID、
冗余业务小类、请款金额、付款银行名称、付款银行卡号、支付附言、收款银行名称、收款银行卡号、
收款方公司ID、收款账户名、收款方公司名称、明细数量、付款单备注、审批单号、提交审批人、
提交审批人账号、审批模版id、审批终端,1系统内审批,2钉钉端审批、
付款审批状态,1待审批,2审批中,3审批通过,4审批被拒绝,5撤销,6作废、状态变更时间、
审批备注、状态变更人、状态变更人账号、变更原因、是否需要调整,0否,1是、调整人账号、
调整人姓名、调整建议、最新支付号、最新支付备注、最新支付状态,1待支付,2支付中,3已支付(部分成功),
4已经支付(全部成功),5已支付(全部失败),6未支付被驳回,7支付成功后被退回、支付中金额、实付(未退回)、
实付(被退回)、剩余可支付、已负冲金额、已取消金额、更新时间、修改人、修改人名字、创建时间、
创建人、创建人名字、删除状态,0未删除,1已删除、traceId、乐观锁版本号。
'''
# bf_no_order_advance_payment_form_normal_payment
sentences2 = '''
无订单垫付-付款单-无订单常规请款缴费-明细表，字段信息有：主键id、
企业ID、付款单ID、付款单支付方式,1普通支付、订单类型,1正常订单、明细单号、
项目ID、项目名称、客户ID、客户名称、费用大类ID、冗余费用大类名称、费用小类ID、
冗余费用小类名称、金额、最新支付号、最新支付状态,1待支付,2支付中,3已支付(部分成功),4已经支付(全部成功),5已支付(全部失败),6未支付被驳回,7支付成功后被退回,8已取消、支付状态时间、支付状态原因、更新时间、修改人、修改人名字、创建时间、创建人、创建人名字、删除状态,0未删除,1已删除、traceId、乐观锁版本号。
'''
# bf_offline_payment_form_advance_back_detail
sentences3 = '''
线下业务-付款单-垫款退回-明细表，字段信息有：主键id、企业ID、付款单ID、付款单支付方式,1普通支付,2被负冲,3负冲、
垫款单编号、项目ID、项目名称、客户ID、客户名称、垫款退回金额、最新支付号、最新支付备注、
最新支付状态,1待支付,2支付中,3已支付(部分成功),4已经支付(全部成功),5已支付(全部失败),6未支付被驳回,7支付成功后被退回、
支付状态原因、支付状态时间、更新时间、修改人、修改人名字、创建时间、创建人、创建人名字、删除状态,0未删除,1已删除、
traceId、乐观锁版本号。
'''

# print(local_embedder.get_embedding(sentences), 'embed_model')
# [0.8805457949638367, -0.16168220341205597, 0.5009350180625916, -0.19151932001113892, 
# 0.8805458 -0.1616822 0.500935 -0.19151932

# 1. 准备数据
documents = [sentences, sentences2, sentences3]

client = chromadb.Client()
collection = client.create_collection(
    name="my_collection",
    embedding_function=local_embedder  # 直接传入函数名
)

collection.add(
    documents=documents,
    ids=["id1", "id2", "id3"]
)

query = '根据垫款退回-明细中付款单ID等于20230313001，查找付款单银行卡卡号和垫款退回明细中的客户名称'

# 4. 查询
results = collection.query(
    query_texts=[query],
    n_results=2
)
print(results)

