# AICUP-2023-Spring-NLP

## 建立環境

1. 安裝 Elasticsearch，請參考 [此處](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html)
2. 使用 [conda](https://docs.conda.io/en/latest/miniconda.html) 建立環境，另外推薦使用 [mamba](https://mamba.readthedocs.io/en/latest/installation.html) 比 conda 更快更穩

```bash
conda env create -f environment.yml
# or
mamba env create -f environment.yml

# 啟用環境
conda activate aicup-2023-spring-nlp
```

## 使用範例

請參考 [examples.ipynb](examples.ipynb)

## 核心模組

### [`NounExtractor`](src/aicup/utils/noun_extractor.py)

此模組封裝了 `hanlp` 的 API 用來提取文字中的名詞關鍵字

- `extract()`: 輸入字串列表，回傳字串集合列表，字串集合中的內容是名詞關鍵字


### [`WikiDataset`](src/aicup/data/wiki_dataset.py)

封裝了 `datasets`、`NounExtractor` 及 `elasticsearch` 主要用來進行 Document Retrieval 及一些文件的對應及查找，若需要進行 Document Retrieval 應確定以下幾點：
1. Elasticsearch 的伺服器已經正確運行
2. 有設定以下環境變數
   - ES_USERNAME: Elasticsearch 的使用者名稱
   - ES_PASSWORD: Elasticsearch 的密碼
   - ES_HOSTS: Elasticsearch 的 Host Name 或 IP
   - ES_CA_CERTS: Elasticsearch 的憑證路徑
3. 若尚未建立索引需呼叫 `add_elasticsearch_index()` 建立索引，若已經建立過索引則需呼叫 `load_elasticsearch_index()` 載入索引

- `from_json()`: 用來將原始 JSON Lines 格式的 Wiki 資料轉換成 `datasets` 的格式方便使用
- `save()`: 將轉換好的 `WikiDataset` 儲存至硬碟
- `load()`: 從硬碟載入轉換好的 `WikiDataset`
- `add_elasticsearch_index()`: 建立 Elasticsearch 索引
- `load_elasticsearch_index()`: 載入 Elasticsearch 索引
- `retrieve()`: 執行 Document Retrieval

### [`PairwiseRankingSentenceRetriever`](src/aicup/models/sentence_retriever/pairwise_ranking_sentence_retriever.py)

用來做 Sentence Retrieval 的模型

- `load_from_checkpoint()`: 載入訓練好的權重
- `predict()`: 預測 Claim-Sentence Pair 的分數

### [`ClassifierClaimVerifier`](src/aicup/models/claim_verifier/classifier_claim_verifier.py)

用來做 Claim Verification 的模型

- `load_from_checkpoint()`: 載入訓練好的權重
- `predict()`: 預測 Claim 的類別

## 核心腳本

部分腳本會利用到 Elasticsearch，請參考[此處](#wikidataset)說明確保 Elasticsearch 正確設定

### [scripts/utils/create_claim_dataset.py](scripts/utils/create_claim_dataset.py)

用來將原始 JSON Lines 訓練資料轉換為 `datasets` 格式，並切分驗證集

### [scripts/utils/create_wiki_dataset.py](scripts/utils/create_wiki_dataset.py)

用來將原始 JSON Lines Wiki 資料轉換為 `WikiDataset` 格式，並建立 Elasticsearch 索引

### [scripts/sentence_retrieval/prepare_pairwise_ranking_dataset.py](scripts/sentence_retrieval/prepare_pairwise_ranking_dataset.py)

準備用來訓練 Pairwise Ranking Sentence Retriever 的資料集

### [scripts/sentence_retrieval/train_pairwise_ranking.py](scripts/sentence_retrieval/train_pairwise_ranking.py)

訓練 Pairwise Ranking Sentence Retriever

### [scripts/claim_verification/prepare_classifier_dataset_prsr.py](scripts/claim_verification/prepare_classifier_dataset_prsr.py)

使用 Pairwise Ranking Sentence Retriever 來準備 Classifier Claim Verifier 的資料集

### [scripts/claim_verification/train_classifier.py](scripts/claim_verification/train_classifier.py)

訓練 Classifier Claim Verifier

### [scripts/e2e/prsr_ccv.py](scripts/e2e/prsr_ccv.py)

End-to-End 的腳本，輸入原始 JSON Lines 的測試資料，輸出用來提交至 AI 實戰吧的檔案，其中使用了 Pairwise Ranking Sentence Retriever 及 Classifier Claim Verifier 來進行預測

## 資料 & 權重

預處理的資料及預訓練的模型權重存放於 [Hugging Face](https://huggingface.co/ShinoharaHare/AICUP-2023-Spring-NLP) 的 Repository

- `data/claim_dataset`: 使用[此腳本](#scriptsutilscreate_claim_datasetpy) 處理過的訓練資料集
- `data/wiki_dataset`: 使用[此腳本](#scriptsutilscreate_wiki_datasetpy) 處理過的 Wiki 資料集
- `data/sentence_retrieval/pairwise_ranking`: 使用[此腳本](#scriptssentence_retrievalprepare_pairwise_ranking_datasetpy) 生成的資料集，用來訓練 Pairwise Ranking Sentence Retriever
- `data/claim_verification/classifier_prsr`: 使用[此腳本](#scriptsclaim_verificationprepare_classifier_dataset_prsrpy) 生成的資料集，用來訓練 Classifier Claim Verifier
- `sentence_retriever/e8hneqtg/e2.weights.ckpt`: Pairwise Ranking Sentence Retriever 的模型權重
- `claim_verifier/7xet7u1m/e9.weights.ckpt`: Classifier Claim Verifier 的模型權重
