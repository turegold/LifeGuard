# 🚑 LifeGuard  
**실시간 공공데이터와 AI를 활용한 응급실 수용 가능성기반 병원 추천 시스템**

https://lifeguard-front-ov6o.vercel.app/
<p>1. 사용자 상태 입력</p>
<img width="421" height="355" alt="image" src="https://github.com/user-attachments/assets/0f6a8a7b-0714-4c0f-97dd-b8e28845231a" />
<p>2. 상황별 즉각적인 응급처치 가이드 제공</p>
<img width="486" height="354" alt="image" src="https://github.com/user-attachments/assets/956126cf-b605-44f2-a916-d88aef58038e" />
<p>3. 환자 상태 + 병원 상태 기반 병원 추천</p>
<img width="424" height="486" alt="image" src="https://github.com/user-attachments/assets/8f4ca47a-edbc-4fde-bc15-c3cc6dc6e612" />



## 📌 프로젝트 개요

**LifeGuard**는  
실시간 응급실 병상 가용 공공데이터와 AI 기술을 활용하여  
**환자의 위치·상태·중증도를 종합적으로 고려해  
‘수용 가능성이 가장 높은 응급실’을 추천**하는 서비스입니다.

응급 상황에서 환자나 보호자는  
- 의료 용어를 정확히 설명하기 어렵고  
- 병원마다 일일이 전화 문의를 해야 하며  
- 그 과정에서 골든타임을 놓치는 문제(응급실 뺑뺑이)가 반복되고 있습니다.

LifeGuard는 이 문제를 해결하기 위해  
**공공데이터 + LLM + ML + RAG**를 결합한 의사결정 지원 시스템을 제안합니다.



## 🧠 전체 시스템 흐름

```
[환자 입력 (텍스트 + 위치)]
↓ (LLM)
[환자 상태 구조화]
↓
[공공데이터 기반 후보 병원 필터링]
↓ (ML)
[병원별 수용 가능성 예측 & 랭킹]
↓ (RAG)
[응급 행동 가이드 제공]
```

---

## 🔍 Step 1. 환자 상황의 의료적 구조화 (LLM)

- 사용자는 자연어로 현재 상황을 입력
- LLM이 문장을 분석하여 다음 정보를 구조화
  - 중증도(severity)
  - 외상 여부(trauma)
  - ICU, 인공호흡기 필요 여부
  - CT / MRI 등 필요 자원

📌 **의료 진단이 아닌, 병원 판단에 필요한 정보 정리 단계**

---

## 🏥 Step 2. 공공데이터 기반 후보 병원 탐색

- 보건복지부/응급의료 공공데이터 활용
- 실시간 정보 기반 병원 후보 생성
  - 거리 / 이동 시간
  - 병상 가용 여부
  - 중환자실, CT, MRI, 인공호흡기 보유 여부
- 환자 조건과 맞지 않는 병원은 사전 필터링

---

## 📊 Step 3. ML 기반 수용 가능성 예측 & 랭킹

- **Logistic Regression 기반 모델** 사용
- 입력 특징:
  - 환자 상태 feature
  - 병원 자원 및 실시간 상태 feature
- 병원별 **수용 가능성 확률(accept_prob)** 예측
- Top-K 병원 랭킹 생성

📌 “가장 가까운 병원”이 아니라  
👉 **“받아줄 가능성이 가장 높은 병원” 기준**

---

## 📘 Step 4. RAG 기반 응급 행동 가이드 제공

- 사전에 검증된 응급 대응 문서를 벡터 DB로 구축
- **FAISS 기반 임베딩 검색**
- 사용자 상황과 가장 관련성 높은 문서만 검색
- 검색된 문서 **내용만 근거로** LLM이 응답 생성

✔️ 모델 추론이 아닌 **문서 근거 기반 응답**

✔️ 환각(Hallucination) 최소화

---
## 🧩 기술 스택

### Backend
- Python, FastAPI
- Pandas, NumPy

### Frontend
- React, Vite

### AI / ML
- LLM (환자 상태 구조화, 설명 생성)
- Logistic Regression (수용 가능성 예측)
- FAISS (Vector DB)
- RAG (Retrieval-Augmented Generation)
