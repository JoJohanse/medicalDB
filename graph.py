from transformers import BertModel, BertTokenizer
from py2neo import Graph
from py2neo import Node, Relationship
import numpy as np

neo4j_url = ''
user = ''
pwd = ''

graph = Graph(neo4j_url,  auth=(user, pwd))

# # 加载预训练的BERT模型和分词器
# vocab_file = 'model/vocab.txt'
# tokenizer = BertTokenizer(vocab_file)
# bert = BertModel.from_pretrained("model/bert-base-chinese/")

# # 查询病人节点
# patients = graph.nodes.match("Patient")
# 查询药品节点
drugs = graph.nodes.match("Medicine")

# 获取药品节点的'label'属性
drug_labels = [drug['label'] for drug in drugs]
# 保存为csv文件
np.savetxt('drug_list.csv', drug_labels, fmt='%s', delimiter=',')

# # 获取所有病人的'出院诊断'属性
# diagnoses = [patient['出院诊断'] for patient in patients]

# # 定义要获取的属性列表
# properties = ["妊娠", "癌症", "感染", "糖尿病酮症", "糖尿病视网膜病变", "糖尿病肾病", "糖尿病周围神经病变", "下肢动脉病变", "颈动脉病变", "脑血管病", "冠心病", "高血压病"]

# # 获取所有病人节点的指定属性
# patient_properties = [{prop: patient[prop] for prop in properties} for patient in patients]

# # 输出示例{'妊娠': False, '癌症': False, '感染': False, '糖尿病酮症': False, '糖尿病视网膜病变': True, '糖尿病肾病': False, '糖尿病周围神经病变': False, '下肢动脉病变': False, '颈动脉病变': False, '脑血管病': False, '冠心病': False, '高血压病': False}

# # 向量
# patient_vectors_bool = []
# for patient in patient_properties:
#     vector = [int(patient[prop]) for prop in properties]
#     patient_vectors_bool.append(vector)


# # 将每个病人的'出院诊断'属性编码为一个向量
# patient_vectors = []
# for diagnosis in diagnoses:
#     # 对诊断进行分词
#     inputs = tokenizer(diagnosis, return_tensors='pt')
#     # 获取诊断的BERT编码
#     outputs = model(**inputs)
#     # 使用最后一层的CLS标记作为诊断的向量表示
#     patient_vectors.append(outputs.last_hidden_state[0][0].detach().numpy())

# # 保存
# np.save('patient_vectors.npy', np.array(patient_vectors))


# def sliding_window(seq,window_size,step_size):
#     return [seq[i:i+window_size] for i in range(0,len(seq),step_size)]
# window_size = 512
# step_size = 100
# drug_vecs = []
# use_relationships = graph.relationships.match(r_type="use").all()
# for drug in drugs:
#     # 如果药品节点有use关系
#     if drug in [relationship.end_node for relationship in use_relationships]:
#         # 获取药品节点的'药品说明'属性
#         # 检测药品indication是否为空
#         if drug['indications'] == None:
#             drug_vecs.append(np.zeros(768))
#             continue
#         drug_description = drug['indications']
#         # 对药品说明进行分词
#         inputs = tokenizer(text=drug_description, return_tensors='pt')
#         # 获取药品说明的BERT编码
#         outputs = bert(**inputs)
#         # 使用最后一层的CLS标记作为药品说明的向量表示
#         drug_vec = outputs.last_hidden_state[0][0].detach().numpy()
#         drug_vecs.append(drug_vec)
#     else:
#         drug_vecs.append(np.zeros(768))

# # 保存
# np.save('drug_vectors.npy', np.array(drug_vecs))




# # 获取关系‘use’,从病人节点指向药品节点
# use_relationships = graph.relationships.match(r_type="use").all()
# patient_list = list(patients)
# drug_list = list(drugs)

# # 创建一个向量，记录每个病人使用的药品，若使用则为1，否则为0，维度为药品的所有种类（包括病人未使用的）
# patient_drug = np.zeros((len(patients), len(drugs)))
# for relationship in use_relationships:
#     patient_index = patient_list.index(relationship.start_node)
#     drug_index = drug_list.index(relationship.end_node)
#     patient_drug[patient_index][drug_index] = 1

# # 保存
# print(patient_drug.shape)
# np.save('patient_Drug.npy', patient_drug)
