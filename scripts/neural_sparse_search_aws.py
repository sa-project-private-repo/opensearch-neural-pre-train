"""
Neural Sparse Search experiments on Amazon OpenSearch Service.

Comprehensive experiment script comparing SEISMIC (ANN), exact,
and rank_features sparse search with client-side SPLADE encoding.

Model: SPLADEModernBERT (skt/A.X-Encoder-base, 50K vocab)
Architecture: AutoModelForMaskedLM -> log(1+ReLU) -> max pool

Usage:
    python scripts/neural_sparse_search_aws.py
    python scripts/neural_sparse_search_aws.py --experiment baseline
    python scripts/neural_sparse_search_aws.py --experiment seismic
    python scripts/neural_sparse_search_aws.py --experiment query
    python scripts/neural_sparse_search_aws.py --experiment hybrid
    python scripts/neural_sparse_search_aws.py --cleanup
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ENDPOINT = "https://ltr-vector.awsbuddy.com"
DEFAULT_MODEL_DIR = "huggingface/v33"
DEFAULT_REGION = "us-east-1"

INDEX_ANN = "sparse-ann-seismic"
INDEX_EXACT = "sparse-exact"
INDEX_RANK = "sparse-rank-features"

QUERY_MAX_LENGTH = 64
DOC_MAX_LENGTH = 256

RESULTS_DIR = "outputs/experiments"
RESULTS_FILE = "neural_sparse_aws_results.json"

DATA_DIR = "data/v29.0"
DEFAULT_NUM_DOCS = 10000

# ---------------------------------------------------------------------------
# Sample Documents (50 documents, 10 domains)
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS: List[Dict[str, str]] = [
    # -- history (5) --
    {
        "title": "한국 전쟁의 배경",
        "content": (
            "1950년 6월 25일 북한의 남침으로 시작된 한국 전쟁은 "
            "냉전 체제 속에서 이념 대립이 군사적 충돌로 이어진 대표적 사례이다. "
            "3년간의 전쟁은 한반도에 막대한 인적, 물적 피해를 남겼으며 "
            "1953년 7월 정전 협정으로 현재의 휴전선이 형성되었다."
        ),
        "category": "history",
    },
    {
        "title": "조선 시대의 과학과 문화",
        "content": (
            "조선 시대에는 세종대왕의 훈민정음 창제를 비롯하여 "
            "측우기, 해시계, 혼천의 등 다양한 과학 기기가 발명되었다. "
            "장영실은 자격루와 앙부일구를 제작하여 조선의 과학 수준을 "
            "세계적으로 끌어올린 인물로 평가받고 있다."
        ),
        "category": "history",
    },
    {
        "title": "고려 시대와 금속활자",
        "content": (
            "고려는 1234년경 세계 최초의 금속 활자를 발명하여 "
            "직지심체요절을 인쇄하였다. 이는 구텐베르크보다 약 200년 앞선 "
            "기술로서 한국 인쇄 문화의 우수성을 보여준다. "
            "또한 고려청자는 독특한 비색과 상감 기법으로 유명하다."
        ),
        "category": "history",
    },
    {
        "title": "삼국시대 고구려의 군사력",
        "content": (
            "고구려는 동북아시아의 강대국으로 광개토대왕 시기에 "
            "영토를 크게 확장하였다. 을지문덕은 살수대첩에서 수나라 "
            "30만 대군을 격퇴하였으며, 안시성 전투에서는 당나라 "
            "태종의 침략을 막아내 동아시아 역사의 흐름을 바꾸었다."
        ),
        "category": "history",
    },
    {
        "title": "일제강점기와 독립운동",
        "content": (
            "1910년 한일합방으로 시작된 일제강점기에 한국인들은 "
            "3.1운동, 의열단 활동, 광복군 창설 등 다양한 독립운동을 "
            "전개하였다. 1945년 8월 15일 광복을 맞이하기까지 "
            "수많은 독립투사들의 희생이 있었다."
        ),
        "category": "history",
    },
    # -- food (5) --
    {
        "title": "김치의 종류와 담그는 법",
        "content": (
            "김치는 배추김치, 깍두기, 총각김치, 파김치 등 200여 종이 있다. "
            "배추김치는 절인 배추에 고춧가루, 젓갈, 마늘, 생강을 넣어 양념하고 "
            "발효시켜 만든다. 유산균이 풍부하여 면역력 향상과 "
            "장 건강에 도움을 주는 세계적인 발효 식품이다."
        ),
        "category": "food",
    },
    {
        "title": "한국 전통 음식 비빔밥",
        "content": (
            "비빔밥은 밥 위에 나물, 고기, 달걀, 고추장을 올려 비벼 먹는 "
            "한국의 대표적인 한 그릇 음식이다. 전주 비빔밥이 가장 유명하며 "
            "다양한 채소를 사용하여 영양 균형이 뛰어나다. "
            "2011년 CNN이 선정한 세계 50대 음식에 포함되었다."
        ),
        "category": "food",
    },
    {
        "title": "부산 해운대 맛집 투어",
        "content": (
            "부산 해운대에는 신선한 해산물 요리가 유명하다. "
            "자갈치 시장에서 회를 즐기고, 밀면과 돼지국밥은 "
            "부산을 대표하는 서민 음식이다. 해운대 해변가에는 "
            "분위기 좋은 카페와 레스토랑도 많아 미식 여행지로 인기가 높다."
        ),
        "category": "food",
    },
    {
        "title": "한국 전통주와 막걸리 문화",
        "content": (
            "막걸리는 쌀을 발효시켜 만든 한국 전통 술로 "
            "유산균과 식이섬유가 풍부하다. 최근 프리미엄 막걸리 시장이 "
            "성장하며 해외 수출도 늘고 있다. 전통주 양조장 투어는 "
            "새로운 관광 콘텐츠로 인기를 끌고 있다."
        ),
        "category": "food",
    },
    {
        "title": "한국 길거리 음식 문화",
        "content": (
            "떡볶이, 순대, 어묵, 호떡은 한국 대표 길거리 음식이다. "
            "명동과 광장시장은 외국인 관광객에게 인기 있는 먹거리 명소이며 "
            "최근에는 K-푸드 열풍으로 한국 길거리 음식이 "
            "해외에서도 큰 인기를 얻고 있다."
        ),
        "category": "food",
    },
    # -- travel (5) --
    {
        "title": "서울 여행 필수 코스",
        "content": (
            "서울 여행에서는 경복궁, 북촌 한옥마을, 남산타워가 필수 코스이다. "
            "명동과 강남은 쇼핑의 중심지이며, 홍대 거리에서는 "
            "젊은 예술가들의 공연과 독특한 카페를 즐길 수 있다. "
            "한강 유람선은 서울의 야경을 감상하기에 최적의 방법이다."
        ),
        "category": "travel",
    },
    {
        "title": "제주도 자연 관광",
        "content": (
            "제주도는 유네스코 세계자연유산으로 등재된 한라산과 "
            "성산일출봉이 대표적인 관광지이다. 올레길 트레킹, "
            "용머리해안 탐방, 만장굴 동굴 탐험 등 자연을 만끽할 수 있으며 "
            "흑돼지 구이와 해녀가 잡은 해산물도 유명하다."
        ),
        "category": "travel",
    },
    {
        "title": "경주 역사 문화 여행",
        "content": (
            "경주는 신라 천년의 수도로 불국사, 석굴암, 첨성대 등 "
            "유네스코 세계문화유산이 도시 곳곳에 있다. "
            "대릉원의 고분군과 안압지 야경은 역사와 자연이 어우러진 "
            "대한민국 최고의 역사 여행지이다."
        ),
        "category": "travel",
    },
    {
        "title": "강원도 겨울 여행",
        "content": (
            "강원도는 평창, 정선, 횡계 등 스키 리조트가 밀집해 있으며 "
            "2018년 동계올림픽 개최지로 세계적 인지도를 높였다. "
            "속초 중앙시장의 먹거리와 설악산 국립공원의 겨울 등산은 "
            "강원도 겨울 여행의 하이라이트이다."
        ),
        "category": "travel",
    },
    {
        "title": "전남 순천만 생태 관광",
        "content": (
            "순천만 습지는 유네스코 생물권보전지역으로 지정된 "
            "세계적인 생태 관광지이다. 갈대밭과 갯벌에는 흑두루미 등 "
            "희귀 철새가 도래하며, 순천만국가정원은 한국 최초의 "
            "국가정원으로 사계절 아름다운 풍경을 자랑한다."
        ),
        "category": "travel",
    },
    # -- tech (5) --
    {
        "title": "삼성전자 반도체 산업",
        "content": (
            "삼성전자는 메모리 반도체 분야에서 세계 1위를 차지하고 있으며 "
            "3나노 GAA 공정 기술을 선도하고 있다. 평택 캠퍼스는 "
            "세계 최대 규모의 반도체 제조 시설로, "
            "연간 수십조 원의 투자가 이루어지고 있다."
        ),
        "category": "tech",
    },
    {
        "title": "한국 인공지능 스타트업 생태계",
        "content": (
            "한국의 AI 스타트업은 자연어 처리, 컴퓨터 비전, 자율주행 등 "
            "다양한 분야에서 빠르게 성장하고 있다. 네이버와 카카오의 "
            "대규모 언어 모델 개발이 활발하며, 정부의 AI 국가전략에 따라 "
            "2025년까지 AI 인재 10만 명 양성을 목표로 하고 있다."
        ),
        "category": "tech",
    },
    {
        "title": "한국 5G 통신 기술",
        "content": (
            "한국은 2019년 세계 최초로 5G 상용 서비스를 시작하였다. "
            "SK텔레콤, KT, LG유플러스가 전국 네트워크를 구축하였으며 "
            "스마트 팩토리, 원격 의료, 자율주행 등 5G 기반의 "
            "다양한 산업 융합 서비스가 확대되고 있다."
        ),
        "category": "tech",
    },
    {
        "title": "한국 전기차 배터리 산업",
        "content": (
            "LG에너지솔루션, SK온, 삼성SDI는 글로벌 전기차 배터리 "
            "시장의 핵심 기업으로 미국과 유럽에 대규모 공장을 건설하고 있다. "
            "차세대 전고체 배터리 기술 개발에서도 선두를 달리며 "
            "2030년까지 세계 배터리 시장 점유율 40% 목표를 세우고 있다."
        ),
        "category": "tech",
    },
    {
        "title": "한국 로봇 산업과 자동화",
        "content": (
            "한국은 제조업 로봇 밀도 세계 1위로 자동차, 반도체 공장에서 "
            "산업용 로봇이 광범위하게 활용되고 있다. 최근에는 서빙 로봇, "
            "배달 로봇, 의료 로봇 등 서비스 로봇 분야가 급성장하며 "
            "일상생활 속 로봇 활용이 확대되고 있다."
        ),
        "category": "tech",
    },
    # -- culture (5) --
    {
        "title": "한국 전통 음악 국악",
        "content": (
            "국악은 한국 고유의 전통 음악으로 판소리, 가야금, 사물놀이 등이 "
            "대표적이다. 판소리는 유네스코 인류무형문화유산으로 등재되어 있으며 "
            "한 명의 소리꾼이 고수의 북 장단에 맞춰 이야기를 노래하는 "
            "독특한 공연 예술이다."
        ),
        "category": "culture",
    },
    {
        "title": "K-POP과 한류의 세계화",
        "content": (
            "BTS, 블랙핑크 등 K-POP 아이돌 그룹은 빌보드 차트를 석권하며 "
            "전 세계적인 한류 열풍을 이끌고 있다. K-드라마와 K-영화도 "
            "넷플릭스를 통해 글로벌 시장에 진출하였으며, "
            "한국 문화 콘텐츠 수출은 매년 급성장하고 있다."
        ),
        "category": "culture",
    },
    {
        "title": "한국 도자기와 공예 예술",
        "content": (
            "한국 도자기는 고려청자의 비색과 조선백자의 순백미로 유명하다. "
            "현대에도 이천, 여주 등 도자기 마을에서 전통 기법을 "
            "계승 발전시키고 있으며, 한지 공예와 나전칠기 등 "
            "한국 전통 공예는 세계적으로 예술적 가치를 인정받고 있다."
        ),
        "category": "culture",
    },
    {
        "title": "한국 영화 산업의 성장",
        "content": (
            "봉준호 감독의 기생충은 2020년 아카데미 작품상을 수상하며 "
            "한국 영화의 위상을 세계에 알렸다. 한국 영화 산업은 "
            "연간 관객 수 2억 명을 넘기며 아시아 최대 규모이고 "
            "넷플릭스와 디즈니플러스 등 OTT 플랫폼 투자도 활발하다."
        ),
        "category": "culture",
    },
    {
        "title": "한국 전통 의복 한복",
        "content": (
            "한복은 한국의 전통 의복으로 저고리, 치마, 바지가 기본 구성이다. "
            "현대에는 생활한복과 개량한복이 일상에서 착용되며 "
            "한복 입고 경복궁 방문은 외국인 관광객에게 인기 있는 "
            "한국 문화 체험 프로그램이다."
        ),
        "category": "culture",
    },
    # -- economy (5) --
    {
        "title": "한국 수출 경제 구조",
        "content": (
            "한국은 수출 주도형 경제로 반도체, 자동차, 석유화학, 철강이 "
            "주요 수출 품목이다. GDP 대비 수출 비중은 40%를 넘으며 "
            "미국, 중국, EU가 3대 수출 시장이다. "
            "최근 K-콘텐츠와 K-뷰티 등 소비재 수출도 급증하고 있다."
        ),
        "category": "economy",
    },
    {
        "title": "한국 부동산 시장 동향",
        "content": (
            "서울 아파트 가격은 2020년 이후 급등하여 "
            "평균 매매가가 10억 원을 넘어섰다. 정부는 다주택자 규제, "
            "대출 규제, 공급 확대 등 다양한 정책을 시행하고 있으며 "
            "전세 사기 문제가 사회적 이슈로 떠올랐다."
        ),
        "category": "economy",
    },
    {
        "title": "한국 스타트업 투자 현황",
        "content": (
            "한국 스타트업 생태계는 유니콘 기업 20개 이상을 배출하며 "
            "아시아에서 중요한 창업 허브로 성장하였다. "
            "정부의 K-스타트업 펀드와 민간 VC 투자가 활발하며 "
            "핀테크, 바이오, AI 분야가 주요 투자 대상이다."
        ),
        "category": "economy",
    },
    {
        "title": "한국은행 금리 정책",
        "content": (
            "한국은행은 물가 안정과 경제 성장의 균형을 위해 "
            "기준금리를 조정하고 있다. 2022년 이후 긴축 기조로 "
            "기준금리가 3.5%까지 인상되었으며, 가계부채 관리와 "
            "경기 연착륙이 통화정책의 핵심 과제이다."
        ),
        "category": "economy",
    },
    {
        "title": "한국 반도체 수출과 경기 회복",
        "content": (
            "2024년 반도체 수출이 역대 최고를 기록하며 "
            "한국 경제 회복을 견인하고 있다. AI 반도체와 HBM 수요 급증으로 "
            "삼성전자와 SK하이닉스의 실적이 크게 개선되었으며 "
            "반도체 슈퍼사이클 진입 기대감이 높아지고 있다."
        ),
        "category": "economy",
    },
    # -- sports (5) --
    {
        "title": "한국 프로야구 인기와 발전",
        "content": (
            "KBO 리그는 1982년 출범 이후 한국 최고 인기 프로 스포츠로 "
            "자리잡았다. 연간 관중 수 800만 명을 넘기며 "
            "류현진, 김하성 등 MLB 진출 선수들이 한국 야구의 "
            "위상을 세계에 알리고 있다."
        ),
        "category": "sports",
    },
    {
        "title": "한국 축구 국가대표팀",
        "content": (
            "한국 축구 국가대표팀은 2002년 월드컵 4강 신화를 달성하며 "
            "아시아 축구의 역사를 새로 썼다. 손흥민은 토트넘에서 "
            "프리미어리그 득점왕을 차지하며 아시아 최고의 축구 선수로 "
            "세계적인 명성을 얻고 있다."
        ),
        "category": "sports",
    },
    {
        "title": "한국 태권도의 세계화",
        "content": (
            "태권도는 한국이 종주국인 무술로 2000년 시드니 올림픽부터 "
            "정식 종목으로 채택되었다. 세계태권도연맹(WT) 가맹국은 "
            "210개국 이상이며, 전 세계 수련생은 8천만 명에 달한다. "
            "한국의 무예 정신과 문화를 세계에 전파하는 역할을 하고 있다."
        ),
        "category": "sports",
    },
    {
        "title": "한국 e스포츠 산업",
        "content": (
            "한국은 e스포츠 종주국으로 리그 오브 레전드, 스타크래프트 등에서 "
            "세계 최강의 실력을 보유하고 있다. T1의 페이커는 "
            "e스포츠 역사상 가장 위대한 선수로 평가받으며 "
            "한국 e스포츠 산업 규모는 연간 2천억 원을 넘어섰다."
        ),
        "category": "sports",
    },
    {
        "title": "2024 파리 올림픽 한국 성적",
        "content": (
            "한국은 2024년 파리 올림픽에서 양궁, 사격, 펜싱 등 "
            "전통 강세 종목에서 금메달을 획득하였다. 특히 양궁은 "
            "남녀 단체전과 개인전을 석권하며 올림픽 양궁 최강국의 "
            "위상을 다시 한번 입증하였다."
        ),
        "category": "sports",
    },
    # -- medicine (5) --
    {
        "title": "한국 의료 시스템과 건강보험",
        "content": (
            "한국의 국민건강보험은 전 국민을 대상으로 하는 "
            "단일 보험자 체계로 OECD 국가 중 의료비 부담이 낮은 편이다. "
            "CT, MRI 등 고가 검사의 보험 적용이 확대되고 있으며 "
            "비급여 항목 축소를 통한 보장성 강화가 진행 중이다."
        ),
        "category": "medicine",
    },
    {
        "title": "한국 바이오 제약 산업",
        "content": (
            "삼성바이오로직스, 셀트리온은 바이오시밀러 분야에서 "
            "글로벌 1, 2위를 차지하고 있다. 한국의 바이오 제약 산업은 "
            "연간 수출 100억 달러를 넘기며, 자체 개발 신약과 "
            "세포유전자 치료제 분야에서도 빠르게 성장하고 있다."
        ),
        "category": "medicine",
    },
    {
        "title": "한국 의료 관광 현황",
        "content": (
            "한국은 성형외과, 피부과, 건강검진 분야에서 "
            "세계적인 의료 관광 목적지이다. 연간 60만 명 이상의 "
            "외국인 환자가 방문하며, 첨단 의료 기술과 합리적 비용이 "
            "한국 의료 관광의 핵심 경쟁력이다."
        ),
        "category": "medicine",
    },
    {
        "title": "한국 한의학과 전통 의료",
        "content": (
            "한의학은 침, 뜸, 한약을 활용하는 한국 전통 의료 체계이다. "
            "양방과 한방의 협진 시스템이 발달하여 근골격계 질환, "
            "소화기 질환 치료에 효과적이다. 최근 한의학의 과학화를 위한 "
            "임상 연구와 AI 진단 시스템 개발이 활발하다."
        ),
        "category": "medicine",
    },
    {
        "title": "코로나19 대응과 K-방역",
        "content": (
            "한국은 코로나19 팬데믹에서 3T 전략(검사, 추적, 치료)으로 "
            "세계적인 방역 모범 사례로 꼽혔다. 드라이브스루 검사소, "
            "자가격리 앱, 역학조사 시스템 등 IT 기반 방역 체계가 "
            "K-방역의 핵심이었으며 글로벌 표준으로 확산되었다."
        ),
        "category": "medicine",
    },
    # -- education (5) --
    {
        "title": "한국 교육 시스템과 수능",
        "content": (
            "대학수학능력시험(수능)은 한국 대학 입시의 핵심으로 "
            "매년 11월 전국적으로 시행된다. 한국의 교육열은 세계 최고 수준으로 "
            "사교육 시장 규모가 연간 26조 원에 달하며, "
            "OECD 국가 중 대학 진학률이 가장 높다."
        ),
        "category": "education",
    },
    {
        "title": "한국 대학의 세계 랭킹",
        "content": (
            "서울대, KAIST, 포항공대, 연세대, 고려대 등 한국 주요 대학은 "
            "QS 세계대학 랭킹에서 상위권에 위치하고 있다. "
            "특히 공학, 컴퓨터과학 분야에서 강점을 보이며 "
            "외국인 유학생 유치도 매년 증가하고 있다."
        ),
        "category": "education",
    },
    {
        "title": "한국 온라인 교육 시장",
        "content": (
            "코로나19 이후 한국의 에듀테크 시장이 급성장하여 "
            "연간 10조 원 규모에 달한다. 메가스터디, 대성마이맥 등 "
            "온라인 강의 플랫폼과 AI 기반 맞춤형 학습 서비스가 "
            "전통적인 학원 교육을 빠르게 대체하고 있다."
        ),
        "category": "education",
    },
    {
        "title": "한국 직업 교육과 마이스터고",
        "content": (
            "마이스터고는 산업 수요에 맞춘 직업 교육 특성화 고등학교로 "
            "졸업 후 취업률이 90%를 넘는다. 반도체, 로봇, 에너지 등 "
            "첨단 산업 분야의 기능 인재를 양성하며 "
            "독일의 듀얼 교육 시스템을 한국식으로 발전시킨 모델이다."
        ),
        "category": "education",
    },
    {
        "title": "한국어 교육과 세종학당",
        "content": (
            "세종학당은 전 세계 82개국에 설치된 한국어 교육 기관으로 "
            "한류 영향으로 수강생이 급증하고 있다. "
            "한국어능력시험(TOPIK) 응시자는 연간 40만 명을 넘었으며 "
            "한국어는 세계에서 가장 빠르게 학습자가 증가하는 언어이다."
        ),
        "category": "education",
    },
    # -- environment (5) --
    {
        "title": "한국 탄소중립 2050 전략",
        "content": (
            "한국 정부는 2050 탄소중립을 선언하고 "
            "온실가스 감축 로드맵을 수립하였다. 석탄 발전 단계적 폐지, "
            "신재생에너지 비중 확대, 탄소 포집 기술 개발 등이 "
            "핵심 전략이며 2030년까지 40% 감축을 목표로 하고 있다."
        ),
        "category": "environment",
    },
    {
        "title": "한국 재생에너지 현황",
        "content": (
            "한국의 재생에너지 발전 비중은 약 9%로 OECD 평균 대비 낮지만 "
            "태양광과 풍력 발전 설비가 빠르게 증가하고 있다. "
            "새만금 태양광 단지, 서남해 해상풍력 등 대규모 프로젝트가 "
            "추진되며 그린 수소 생산 기술 개발도 활발하다."
        ),
        "category": "environment",
    },
    {
        "title": "한국 미세먼지 문제와 대응",
        "content": (
            "한국은 봄철 미세먼지 농도가 높아 사회적 문제가 되고 있다. "
            "국내 배출원 관리와 중국발 미세먼지 공동 대응이 과제이며 "
            "미세먼지 저감 특별법 시행, 노후 경유차 운행 제한, "
            "공기청정기 보급 확대 등 다양한 대책이 시행되고 있다."
        ),
        "category": "environment",
    },
    {
        "title": "한국 해양 환경 보전",
        "content": (
            "한국 연안의 해양 쓰레기와 적조 현상은 심각한 환경 문제이다. "
            "정부는 해양 플라스틱 저감 대책, 갯벌 복원 사업, "
            "해양 보호구역 확대를 추진하고 있으며 "
            "시민 참여형 연안 정화 활동도 활발하게 이루어지고 있다."
        ),
        "category": "environment",
    },
    {
        "title": "한국 전기차 보급과 친환경 교통",
        "content": (
            "한국의 전기차 등록 대수는 50만 대를 넘어섰으며 "
            "충전 인프라 확충이 빠르게 진행되고 있다. "
            "서울시는 친환경 버스 전면 전환을 추진하고 있으며 "
            "수소 연료전지 버스, 자전거 도로 확대 등 "
            "친환경 교통 체계 구축에 힘쓰고 있다."
        ),
        "category": "environment",
    },
]

# ---------------------------------------------------------------------------
# Test Queries (15 queries)
# ---------------------------------------------------------------------------

TEST_QUERIES: List[Dict[str, str]] = [
    # Existing 8 from demo
    {"text": "한국 전쟁의 원인과 결과", "label": "Korean War"},
    {"text": "김치 만드는 방법", "label": "Kimchi recipe"},
    {"text": "서울 여행 추천 장소", "label": "Seoul travel"},
    {"text": "삼성전자 반도체 기술", "label": "Samsung semi"},
    {"text": "한국 전통 음악의 특징", "label": "Korean music"},
    {"text": "부산 해운대 맛집", "label": "Busan food"},
    {"text": "조선 시대 과학 발전", "label": "Joseon science"},
    {"text": "인공지능 스타트업 현황", "label": "AI startups"},
    # New 5 covering new domains
    {"text": "한국 경제 수출 현황", "label": "Korea exports"},
    {"text": "손흥민 축구 성적", "label": "Son football"},
    {"text": "한국 건강보험 제도", "label": "Health insurance"},
    {"text": "수능 교육 시스템", "label": "CSAT education"},
    {"text": "탄소중립 재생에너지", "label": "Carbon neutral"},
    # 2 cross-domain queries
    {
        "text": "한국 기술 혁신이 경제에 미치는 영향",
        "label": "Tech-economy cross",
    },
    {
        "text": "전통 문화와 관광 산업의 관계",
        "label": "Culture-travel cross",
    },
]


# ===========================================================================
# Data loading from training shards
# ===========================================================================


def load_documents_from_shards(
    data_dir: str = DATA_DIR,
    num_docs: int = DEFAULT_NUM_DOCS,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Load unique documents and query-doc pairs from training shards.

    Returns (documents, queries) where:
    - documents: list of {"title": ..., "content": ..., "category": "corpus"}
    - queries: list of {"text": ..., "label": ..., "expected_doc_idx": int}
    """
    data_path = Path(data_dir)
    shard_files = sorted(data_path.glob("train_shard_*.jsonl"))
    if not shard_files:
        raise FileNotFoundError(f"No shards in {data_dir}")

    seen_docs: Dict[str, int] = {}  # content -> doc_index
    documents: List[Dict[str, str]] = []
    query_pairs: List[Dict[str, Any]] = []

    for shard_file in shard_files:
        with open(shard_file) as f:
            for line in f:
                if len(documents) >= num_docs:
                    break
                row = json.loads(line)
                content = row["positive"]
                query = row["query"]

                if content not in seen_docs:
                    doc_idx = len(documents)
                    seen_docs[content] = doc_idx
                    # Use query as pseudo-title (truncated)
                    title = query[:50] if len(query) <= 50 else query[:47] + "..."
                    documents.append({
                        "title": title,
                        "content": content,
                        "category": row.get("source", "corpus"),
                    })
                    query_pairs.append({
                        "text": query,
                        "label": f"q{doc_idx}",
                        "expected_doc_idx": doc_idx,
                    })
        if len(documents) >= num_docs:
            break

    print(f"Loaded {len(documents)} unique documents from {data_dir}")

    # Select diverse test queries (every N-th for coverage)
    num_test = min(50, len(query_pairs))
    step = max(1, len(query_pairs) // num_test)
    test_queries = [query_pairs[i * step] for i in range(num_test)]
    print(f"Selected {len(test_queries)} test queries")

    return documents, test_queries


# ===========================================================================
# Encoder
# ===========================================================================


class SparseEncoder:
    """Client-side SPLADE-max encoder for Korean neural sparse search."""

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        device: str = "auto",
    ) -> None:
        """Initialize SPLADE encoder from a HuggingFace model directory."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"Loading model from {model_dir} on {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForMaskedLM.from_pretrained(model_dir)
        self.model.to(device)
        self.model.eval()

        self.vocab_size = self.tokenizer.vocab_size
        self._token_lookup: List[str] = (
            self.tokenizer.convert_ids_to_tokens(
                list(range(self.vocab_size))
            )
        )

        # Build reverse vocab: token string -> integer ID
        self._token_to_id: Dict[str, int] = {}
        for tid in range(self.vocab_size):
            token_str = self._token_lookup[tid]
            if token_str:
                self._token_to_id[token_str] = tid

        self.special_token_ids = {
            tid
            for tid in (
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.pad_token_id,
                self.tokenizer.unk_token_id,
                self.tokenizer.bos_token_id,
                self.tokenizer.eos_token_id,
            )
            if tid is not None
        }

        print(
            f"Model loaded: {self.model.config.architectures}, "
            f"vocab={self.vocab_size}, device={device}"
        )

    @torch.no_grad()
    def _encode_raw(
        self,
        text: str,
        max_length: int = DOC_MAX_LENGTH,
        top_k: Optional[int] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Encode text, returning (token_str_dict, token_id_dict)."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits  # [1, seq, vocab]

        # SPLADE-max: log(1 + ReLU(logits)), max-pool over sequence
        sparse_scores = torch.log1p(torch.relu(logits))  # [1, seq, V]
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        sparse_scores = sparse_scores * mask
        sparse_repr = sparse_scores.max(dim=1).values.squeeze(0)  # [V]

        # Extract non-zero entries
        nonzero = (sparse_repr > 0).nonzero(as_tuple=True)[0]
        str_dict: Dict[str, float] = {}
        id_dict: Dict[str, float] = {}

        for idx in nonzero.tolist():
            if idx in self.special_token_ids:
                continue
            token = self._token_lookup[idx]
            if token and not token.startswith(("[", "<")):
                weight = round(sparse_repr[idx].item(), 4)
                if weight <= 0.0:
                    continue
                str_dict[token] = weight
                id_dict[str(idx)] = weight

        if top_k and len(str_dict) > top_k:
            sorted_items = sorted(
                str_dict.items(), key=lambda x: x[1], reverse=True
            )
            str_dict = dict(sorted_items[:top_k])
            # Rebuild id_dict to match top_k filtering
            id_dict = {}
            for token, weight in str_dict.items():
                tid = self._token_to_id.get(token)
                if tid is not None:
                    id_dict[str(tid)] = weight

        return str_dict, id_dict

    def encode(
        self,
        text: str,
        max_length: int = DOC_MAX_LENGTH,
        top_k: Optional[int] = None,
    ) -> Dict[str, float]:
        """Encode text to sparse {token_string: weight} dict."""
        str_dict, _ = self._encode_raw(text, max_length, top_k)
        return str_dict

    def encode_for_sparse_vector(
        self,
        text: str,
        max_length: int = DOC_MAX_LENGTH,
        top_k: Optional[int] = None,
    ) -> Dict[str, float]:
        """Encode text to sparse {token_id_string: weight} dict.

        Keys are integer token IDs as strings for sparse_vector fields.
        Example: {"31380": 2.5, "8923": 1.2}
        """
        _, id_dict = self._encode_raw(text, max_length, top_k)
        return id_dict

    def encode_query(
        self, text: str, top_k: int = 100
    ) -> Dict[str, float]:
        """Encode a query (shorter max_length, optional top-k)."""
        return self.encode(text, max_length=QUERY_MAX_LENGTH, top_k=top_k)

    def encode_query_for_sparse_vector(
        self, text: str, top_k: int = 100
    ) -> Dict[str, float]:
        """Encode query to {token_id_string: weight} for neural_sparse."""
        return self.encode_for_sparse_vector(
            text, max_length=QUERY_MAX_LENGTH, top_k=top_k
        )

    def encode_document(self, text: str) -> Dict[str, float]:
        """Encode a document (longer max_length)."""
        return self.encode(text, max_length=DOC_MAX_LENGTH)

    def encode_document_for_sparse_vector(
        self, text: str
    ) -> Dict[str, float]:
        """Encode document to {token_id_string: weight} for sparse_vector."""
        return self.encode_for_sparse_vector(
            text, max_length=DOC_MAX_LENGTH
        )


# ===========================================================================
# AWS OpenSearch connection
# ===========================================================================


def get_aws_opensearch_client(
    endpoint: str,
    region: str = DEFAULT_REGION,
) -> Any:
    """Create OpenSearch client with AWS SigV4 authentication."""
    import boto3
    from opensearchpy import OpenSearch, AWSV4SignerAuth, RequestsHttpConnection

    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, region, "es")

    # Strip protocol for hosts if full URL given
    host = endpoint.replace("https://", "").replace("http://", "")

    client = OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60,
    )

    info = client.info()
    version = info.get("version", {}).get("number", "unknown")
    dist = info.get("version", {}).get("distribution", "unknown")
    print(f"Connected to {dist} {version} at {endpoint}")
    return client


# ===========================================================================
# Index management
# ===========================================================================


def _base_settings() -> Dict[str, Any]:
    """Shared index settings with nori analyzer."""
    return {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "korean_nori": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "filter": ["nori_readingform", "lowercase"],
                }
            }
        },
    }


def _base_properties() -> Dict[str, Any]:
    """Shared field mappings (title, content, category)."""
    return {
        "title": {"type": "text", "analyzer": "korean_nori"},
        "content": {"type": "text", "analyzer": "korean_nori"},
        "category": {"type": "keyword"},
    }


def create_ann_index(
    client: Any,
    index_name: str = INDEX_ANN,
    n_postings: int = 300,
    cluster_ratio: float = 0.1,
    summary_prune_ratio: float = 0.4,
) -> None:
    """Create sparse_vector index with SEISMIC ANN method."""
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)

    props = _base_properties()
    props["sparse_embedding"] = {
        "type": "sparse_vector",
        "index": True,
        "method": {
            "name": "seismic",
            "parameters": {
                "n_postings": n_postings,
                "cluster_ratio": cluster_ratio,
                "summary_prune_ratio": summary_prune_ratio,
            },
        },
    }

    body = {
        "settings": _base_settings(),
        "mappings": {"properties": props},
    }
    client.indices.create(index=index_name, body=body)
    print(
        f"Created ANN index '{index_name}' "
        f"(n_postings={n_postings}, cluster_ratio={cluster_ratio}, "
        f"summary_prune_ratio={summary_prune_ratio})"
    )


def create_exact_index(
    client: Any,
    index_name: str = INDEX_EXACT,
) -> None:
    """Create sparse_vector index for near-exact search.

    Uses SEISMIC with very high n_postings (1000) and low prune ratio
    to maximize recall as reference baseline.
    """
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)

    props = _base_properties()
    props["sparse_embedding"] = {
        "type": "sparse_vector",
        "index": True,
        "method": {
            "name": "seismic",
            "parameters": {
                "n_postings": 1000,
                "cluster_ratio": 0.05,
                "summary_prune_ratio": 0.1,
            },
        },
    }

    body = {
        "settings": _base_settings(),
        "mappings": {"properties": props},
    }
    client.indices.create(index=index_name, body=body)
    print(f"Created exact index '{index_name}' (SEISMIC, exact via query params)")


def create_rank_features_index(
    client: Any,
    index_name: str = INDEX_RANK,
) -> None:
    """Create rank_features index (legacy comparison)."""
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)

    props = _base_properties()
    props["sparse_embedding"] = {"type": "rank_features"}

    body = {
        "settings": _base_settings(),
        "mappings": {"properties": props},
    }
    client.indices.create(index=index_name, body=body)
    print(f"Created rank_features index '{index_name}'")


def bulk_index_documents(
    client: Any,
    index_name: str,
    documents: List[Dict[str, str]],
    sparse_vectors: List[Dict[str, float]],
    chunk_size: int = 500,
) -> int:
    """Bulk-index documents with their sparse embeddings."""
    from opensearchpy.helpers import bulk

    total_success = 0
    total_errors = 0

    for start in range(0, len(documents), chunk_size):
        end = min(start + chunk_size, len(documents))
        actions = []
        for i in range(start, end):
            doc = documents[i]
            vec = sparse_vectors[i]
            # Filter out any zero/negative weights
            clean_vec = {
                k: v for k, v in vec.items() if v > 0
            }
            if not clean_vec:
                continue
            actions.append(
                {
                    "_index": index_name,
                    "_id": str(i),
                    "_source": {
                        "title": doc["title"],
                        "content": doc["content"],
                        "category": doc["category"],
                        "sparse_embedding": clean_vec,
                    },
                }
            )

        if actions:
            success, errors = bulk(
                client, actions,
                raise_on_error=False,
                refresh=False,
            )
            total_success += success
            total_errors += len(errors)

    # Final refresh
    client.indices.refresh(index=index_name)

    if total_errors:
        print(
            f"WARNING: {total_errors} indexing errors in "
            f"'{index_name}'"
        )
    print(f"Indexed {total_success} documents into '{index_name}'")
    return total_success


# ===========================================================================
# Search functions
# ===========================================================================


def neural_sparse_search(
    client: Any,
    index_name: str,
    query_vec_int: Dict[str, float],
    top_k: int = 5,
    method_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Search using neural_sparse query on sparse_vector field."""
    neural_sparse_body: Dict[str, Any] = {
        "query_tokens": query_vec_int,
    }
    if method_params:
        neural_sparse_body["method_parameters"] = method_params

    body: Dict[str, Any] = {
        "size": top_k,
        "query": {
            "neural_sparse": {
                "sparse_embedding": neural_sparse_body,
            }
        },
    }
    return client.search(index=index_name, body=body)


def rank_feature_search(
    client: Any,
    index_name: str,
    query_vec_str: Dict[str, float],
    top_k: int = 5,
) -> Dict[str, Any]:
    """Search using rank_feature queries (for rank_features field)."""
    should_clauses = []
    for token, weight in query_vec_str.items():
        should_clauses.append(
            {
                "rank_feature": {
                    "field": f"sparse_embedding.{token}",
                    "boost": weight,
                }
            }
        )

    if not should_clauses:
        return {"hits": {"hits": [], "total": {"value": 0}}}

    body = {
        "size": top_k,
        "query": {"bool": {"should": should_clauses}},
    }
    return client.search(index=index_name, body=body)


def bm25_search(
    client: Any,
    index_name: str,
    query_text: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Search using BM25 with nori analyzer."""
    body = {
        "size": top_k,
        "query": {
            "match": {
                "content": {
                    "query": query_text,
                    "analyzer": "korean_nori",
                }
            }
        },
    }
    return client.search(index=index_name, body=body)


def hybrid_search(
    client: Any,
    index_name: str,
    query_text: str,
    query_vec_int: Dict[str, float],
    top_k: int = 5,
    bm25_weight: float = 0.3,
    sparse_weight: float = 0.7,
) -> List[Dict[str, Any]]:
    """Hybrid BM25 + neural_sparse with client-side score normalization."""
    bm25_resp = bm25_search(client, index_name, query_text, top_k * 2)
    bm25_hits = bm25_resp["hits"]["hits"]

    sparse_resp = neural_sparse_search(
        client, index_name, query_vec_int, top_k * 2
    )
    sparse_hits = sparse_resp["hits"]["hits"]

    bm25_max = max((h["_score"] for h in bm25_hits), default=1.0)
    sparse_max = max((h["_score"] for h in sparse_hits), default=1.0)

    combined: Dict[str, Dict[str, Any]] = {}

    for hit in bm25_hits:
        doc_id = hit["_id"]
        norm = hit["_score"] / bm25_max if bm25_max > 0 else 0.0
        combined[doc_id] = {
            "source": hit["_source"],
            "bm25_score": norm,
            "sparse_score": 0.0,
        }

    for hit in sparse_hits:
        doc_id = hit["_id"]
        norm = hit["_score"] / sparse_max if sparse_max > 0 else 0.0
        if doc_id in combined:
            combined[doc_id]["sparse_score"] = norm
        else:
            combined[doc_id] = {
                "source": hit["_source"],
                "bm25_score": 0.0,
                "sparse_score": norm,
            }

    for entry in combined.values():
        entry["hybrid_score"] = (
            bm25_weight * entry["bm25_score"]
            + sparse_weight * entry["sparse_score"]
        )

    ranked = sorted(
        combined.items(),
        key=lambda x: x[1]["hybrid_score"],
        reverse=True,
    )
    return [
        {
            "_id": doc_id,
            "_source": entry["source"],
            "_score": entry["hybrid_score"],
            "bm25_score": entry["bm25_score"],
            "sparse_score": entry["sparse_score"],
        }
        for doc_id, entry in ranked[:top_k]
    ]


# ===========================================================================
# Experiment functions
# ===========================================================================


def _timed_search(
    search_fn, *args: Any, **kwargs: Any
) -> Tuple[Dict[str, Any], float]:
    """Execute search and return (response, latency_ms)."""
    t0 = time.perf_counter()
    resp = search_fn(*args, **kwargs)
    latency = (time.perf_counter() - t0) * 1000
    return resp, latency


def _extract_doc_ids(resp: Dict[str, Any], top_k: int = 5) -> List[str]:
    """Extract document IDs from search response."""
    return [h["_id"] for h in resp["hits"]["hits"][:top_k]]


def _recall_at_k(
    retrieved: List[str], reference: List[str], k: int = 5
) -> float:
    """Compute recall@k: fraction of reference in retrieved[:k]."""
    if not reference:
        return 0.0
    ref_set = set(reference[:k])
    ret_set = set(retrieved[:k])
    return len(ref_set & ret_set) / len(ref_set)


def run_baseline_comparison(
    client: Any,
    encoder: SparseEncoder,
    query_vecs_int: List[Dict[str, float]],
    query_vecs_str: List[Dict[str, float]],
    queries: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Compare exact vs ANN vs rank_features on all queries."""
    if queries is None:
        queries = TEST_QUERIES
    print("\n" + "=" * 90)
    print("  EXPERIMENT: Baseline Comparison (Exact vs ANN vs Rank)")
    print("=" * 90)

    results: List[Dict[str, Any]] = []

    for i, q in enumerate(queries):
        qv_int = query_vecs_int[i]
        qv_str = query_vecs_str[i]

        # Exact search (reference) - same SEISMIC method, separate index
        exact_resp, exact_ms = _timed_search(
            neural_sparse_search, client, INDEX_EXACT, qv_int, 5,
        )
        exact_ids = _extract_doc_ids(exact_resp)

        # ANN search
        ann_resp, ann_ms = _timed_search(
            neural_sparse_search, client, INDEX_ANN, qv_int, 5
        )
        ann_ids = _extract_doc_ids(ann_resp)

        # Rank features search
        rank_resp, rank_ms = _timed_search(
            rank_feature_search, client, INDEX_RANK, qv_str, 5
        )
        rank_ids = _extract_doc_ids(rank_resp)

        recall_ann = _recall_at_k(ann_ids, exact_ids)
        recall_rank = _recall_at_k(rank_ids, exact_ids)

        row = {
            "query": q["text"],
            "label": q["label"],
            "exact_ms": round(exact_ms, 1),
            "ann_ms": round(ann_ms, 1),
            "rank_ms": round(rank_ms, 1),
            "ann_recall@5": round(recall_ann, 3),
            "rank_recall@5": round(recall_rank, 3),
            "exact_top1": exact_ids[0] if exact_ids else "",
            "ann_top1": ann_ids[0] if ann_ids else "",
            "rank_top1": rank_ids[0] if rank_ids else "",
        }
        results.append(row)

    # Print table
    print(
        f"\n  {'Label':<22}{'Exact(ms)':<11}{'ANN(ms)':<10}"
        f"{'Rank(ms)':<10}{'ANN R@5':<10}{'Rank R@5':<10}"
    )
    print("  " + "-" * 73)
    for r in results:
        print(
            f"  {r['label']:<22}{r['exact_ms']:<11.1f}"
            f"{r['ann_ms']:<10.1f}{r['rank_ms']:<10.1f}"
            f"{r['ann_recall@5']:<10.3f}{r['rank_recall@5']:<10.3f}"
        )

    # Averages
    avg_exact = sum(r["exact_ms"] for r in results) / len(results)
    avg_ann = sum(r["ann_ms"] for r in results) / len(results)
    avg_rank = sum(r["rank_ms"] for r in results) / len(results)
    avg_ann_recall = (
        sum(r["ann_recall@5"] for r in results) / len(results)
    )
    avg_rank_recall = (
        sum(r["rank_recall@5"] for r in results) / len(results)
    )
    print("  " + "-" * 73)
    print(
        f"  {'AVERAGE':<22}{avg_exact:<11.1f}"
        f"{avg_ann:<10.1f}{avg_rank:<10.1f}"
        f"{avg_ann_recall:<10.3f}{avg_rank_recall:<10.3f}"
    )

    return {
        "experiment": "baseline_comparison",
        "results": results,
        "averages": {
            "exact_ms": round(avg_exact, 1),
            "ann_ms": round(avg_ann, 1),
            "rank_ms": round(avg_rank, 1),
            "ann_recall@5": round(avg_ann_recall, 3),
            "rank_recall@5": round(avg_rank_recall, 3),
        },
    }


def run_seismic_parameter_experiment(
    client: Any,
    encoder: SparseEncoder,
    documents: List[Dict[str, str]],
    doc_vecs_int: List[Dict[str, float]],
    query_vecs_int: List[Dict[str, float]],
) -> Dict[str, Any]:
    """Test different SEISMIC index parameters."""
    print("\n" + "=" * 90)
    print("  EXPERIMENT: SEISMIC Index Parameters")
    print("=" * 90)

    # Default values
    default_n = 300
    default_cr = 0.1
    default_spr = 0.4

    param_configs: List[Dict[str, Any]] = []

    # Vary n_postings (wider range for recall variation)
    for n in [10, 50, 100, 300, 500, 1000]:
        param_configs.append({
            "name": f"n_postings={n}",
            "n_postings": n,
            "cluster_ratio": default_cr,
            "summary_prune_ratio": default_spr,
        })

    # Vary cluster_ratio (wider range)
    for cr in [0.01, 0.05, 0.1, 0.2, 0.5]:
        if cr == default_cr:
            continue
        param_configs.append({
            "name": f"cluster_ratio={cr}",
            "n_postings": default_n,
            "cluster_ratio": cr,
            "summary_prune_ratio": default_spr,
        })

    # Vary summary_prune_ratio (wider range)
    for spr in [0.1, 0.2, 0.4, 0.6, 0.8]:
        if spr == default_spr:
            continue
        param_configs.append({
            "name": f"summary_prune_ratio={spr}",
            "n_postings": default_n,
            "cluster_ratio": default_cr,
            "summary_prune_ratio": spr,
        })

    # Get exact reference results
    exact_results: Dict[int, List[str]] = {}
    num_q = len(query_vecs_int)
    for i in range(num_q):
        resp = neural_sparse_search(
            client, INDEX_EXACT, query_vecs_int[i], 5
        )
        exact_results[i] = _extract_doc_ids(resp)

    all_results: List[Dict[str, Any]] = []
    temp_idx = "sparse-ann-seismic-exp"

    for cfg in param_configs:
        print(f"\n  Testing: {cfg['name']}")

        # Create index with these params
        create_ann_index(
            client,
            index_name=temp_idx,
            n_postings=cfg["n_postings"],
            cluster_ratio=cfg["cluster_ratio"],
            summary_prune_ratio=cfg["summary_prune_ratio"],
        )
        bulk_index_documents(client, temp_idx, documents, doc_vecs_int)

        recalls = []
        latencies = []
        for i in range(num_q):
            resp, ms = _timed_search(
                neural_sparse_search, client, temp_idx,
                query_vecs_int[i], 5,
            )
            ids = _extract_doc_ids(resp)
            recall = _recall_at_k(ids, exact_results[i])
            recalls.append(recall)
            latencies.append(ms)

        avg_recall = sum(recalls) / len(recalls)
        avg_latency = sum(latencies) / len(latencies)

        row = {
            "config": cfg["name"],
            "avg_recall@5": round(avg_recall, 3),
            "avg_latency_ms": round(avg_latency, 1),
            "params": {
                "n_postings": cfg["n_postings"],
                "cluster_ratio": cfg["cluster_ratio"],
                "summary_prune_ratio": cfg["summary_prune_ratio"],
            },
        }
        all_results.append(row)
        print(
            f"    recall@5={avg_recall:.3f}  "
            f"latency={avg_latency:.1f}ms"
        )

        # Cleanup temp index
        if client.indices.exists(index=temp_idx):
            client.indices.delete(index=temp_idx)

    # Print summary table
    print(f"\n  {'Config':<30}{'Recall@5':<12}{'Latency(ms)':<12}")
    print("  " + "-" * 54)
    for r in all_results:
        print(
            f"  {r['config']:<30}{r['avg_recall@5']:<12.3f}"
            f"{r['avg_latency_ms']:<12.1f}"
        )

    return {
        "experiment": "seismic_parameters",
        "results": all_results,
    }


def run_query_parameter_experiment(
    client: Any,
    query_vecs_int: List[Dict[str, float]],
) -> Dict[str, Any]:
    """Test different method_parameters in neural_sparse query."""
    print("\n" + "=" * 90)
    print("  EXPERIMENT: Query-Time Parameters")
    print("=" * 90)

    # Get exact reference
    exact_results: Dict[int, List[str]] = {}
    num_q = len(query_vecs_int)
    for i in range(num_q):
        resp = neural_sparse_search(
            client, INDEX_EXACT, query_vecs_int[i], 5
        )
        exact_results[i] = _extract_doc_ids(resp)

    param_sets: List[Dict[str, Any]] = [
        {"name": "default (no params)", "params": None},
    ]

    # heap_factor variations
    for hf in [0.5, 1.0, 2.0]:
        param_sets.append({
            "name": f"heap_factor={hf}",
            "params": {"heap_factor": hf},
        })

    # top_n variations
    for tn in [5, 10, 20]:
        param_sets.append({
            "name": f"top_n={tn}",
            "params": {"top_n": tn},
        })

    all_results: List[Dict[str, Any]] = []

    for ps in param_sets:
        recalls = []
        latencies = []
        for i in range(num_q):
            resp, ms = _timed_search(
                neural_sparse_search,
                client,
                INDEX_ANN,
                query_vecs_int[i],
                5,
                ps["params"],
            )
            ids = _extract_doc_ids(resp)
            recall = _recall_at_k(ids, exact_results[i])
            recalls.append(recall)
            latencies.append(ms)

        avg_recall = sum(recalls) / len(recalls)
        avg_latency = sum(latencies) / len(latencies)

        row = {
            "config": ps["name"],
            "avg_recall@5": round(avg_recall, 3),
            "avg_latency_ms": round(avg_latency, 1),
            "method_parameters": ps["params"],
        }
        all_results.append(row)

    # Print table
    print(f"\n  {'Config':<28}{'Recall@5':<12}{'Latency(ms)':<12}")
    print("  " + "-" * 52)
    for r in all_results:
        print(
            f"  {r['config']:<28}{r['avg_recall@5']:<12.3f}"
            f"{r['avg_latency_ms']:<12.1f}"
        )

    return {
        "experiment": "query_parameters",
        "results": all_results,
    }


def run_hybrid_weight_experiment(
    client: Any,
    encoder: SparseEncoder,
    query_vecs_int: List[Dict[str, float]],
    queries: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Test different BM25:Sparse weight ratios."""
    if queries is None:
        queries = TEST_QUERIES
    print("\n" + "=" * 90)
    print("  EXPERIMENT: Hybrid Weight Ratios (BM25 : Sparse)")
    print("=" * 90)

    weight_ratios = [
        (0.0, 1.0),
        (0.2, 0.8),
        (0.3, 0.7),
        (0.5, 0.5),
        (0.7, 0.3),
        (1.0, 0.0),
    ]

    # Use exact index for reference (pure sparse)
    exact_results: Dict[int, List[str]] = {}
    num_q = len(query_vecs_int)
    for i in range(num_q):
        resp = neural_sparse_search(
            client, INDEX_EXACT, query_vecs_int[i], 5
        )
        exact_results[i] = _extract_doc_ids(resp)

    all_results: List[Dict[str, Any]] = []

    for bm25_w, sparse_w in weight_ratios:
        name = f"BM25={bm25_w:.1f}:Sparse={sparse_w:.1f}"
        recalls = []
        latencies = []

        for i, q in enumerate(queries):
            t0 = time.perf_counter()
            hits = hybrid_search(
                client,
                INDEX_ANN,
                q["text"],
                query_vecs_int[i],
                top_k=5,
                bm25_weight=bm25_w,
                sparse_weight=sparse_w,
            )
            ms = (time.perf_counter() - t0) * 1000
            latencies.append(ms)

            hybrid_ids = [h["_id"] for h in hits]
            recall = _recall_at_k(hybrid_ids, exact_results[i])
            recalls.append(recall)

        avg_recall = sum(recalls) / len(recalls)
        avg_latency = sum(latencies) / len(latencies)

        row = {
            "config": name,
            "bm25_weight": bm25_w,
            "sparse_weight": sparse_w,
            "avg_recall@5": round(avg_recall, 3),
            "avg_latency_ms": round(avg_latency, 1),
        }
        all_results.append(row)

    # Print table
    print(f"\n  {'Weight Ratio':<30}{'Recall@5':<12}{'Latency(ms)':<12}")
    print("  " + "-" * 54)
    for r in all_results:
        print(
            f"  {r['config']:<30}{r['avg_recall@5']:<12.3f}"
            f"{r['avg_latency_ms']:<12.1f}"
        )

    return {
        "experiment": "hybrid_weights",
        "results": all_results,
    }


# ===========================================================================
# Output formatting
# ===========================================================================


def format_sparse_tokens(
    vec: Dict[str, float], top_n: int = 10
) -> str:
    """Format top-N sparse tokens for display."""
    sorted_tokens = sorted(
        vec.items(), key=lambda x: x[1], reverse=True
    )
    parts = [f"{tok}:{w:.3f}" for tok, w in sorted_tokens[:top_n]]
    return "  ".join(parts)


def truncate(text: str, max_len: int = 60) -> str:
    """Truncate text for table display."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def save_results(
    all_experiments: Dict[str, Any],
    output_dir: str = RESULTS_DIR,
    filename: str = RESULTS_FILE,
) -> str:
    """Save experiment results to JSON file."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filepath = out_path / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(all_experiments, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {filepath}")
    return str(filepath)


def cleanup_indices(client: Any) -> None:
    """Delete all experiment indices."""
    indices = [INDEX_ANN, INDEX_EXACT, INDEX_RANK]
    for idx in indices:
        if client.indices.exists(index=idx):
            client.indices.delete(index=idx)
            print(f"Deleted index '{idx}'")
        else:
            print(f"Index '{idx}' does not exist")

    # Also clean up any temp experiment indices
    temp_idx = "sparse-ann-seismic-exp"
    if client.indices.exists(index=temp_idx):
        client.indices.delete(index=temp_idx)
        print(f"Deleted temp index '{temp_idx}'")


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    """Run neural sparse search experiments on AWS OpenSearch."""
    parser = argparse.ArgumentParser(
        description=(
            "Neural Sparse Search experiments on "
            "Amazon OpenSearch Service"
        )
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"OpenSearch endpoint (default: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"HF model directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION})",
    )
    parser.add_argument(
        "--experiment",
        default="all",
        choices=["all", "baseline", "seismic", "query", "hybrid"],
        help="Which experiment to run (default: all)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete all experiment indices and exit",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for model inference (default: auto)",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=DEFAULT_NUM_DOCS,
        help=f"Number of docs to load from shards (default: {DEFAULT_NUM_DOCS})",
    )
    parser.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help=f"Training data directory (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use built-in 50 sample documents instead of shard data",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Connect to AWS OpenSearch
    # ------------------------------------------------------------------
    print("=" * 90)
    print("  Neural Sparse Search - AWS OpenSearch Experiments")
    print("  Model: SPLADEModernBERT (skt/A.X-Encoder-base, 50K vocab)")
    print(f"  Endpoint: {args.endpoint}")
    print("=" * 90)
    print()

    client = get_aws_opensearch_client(args.endpoint, args.region)

    # ------------------------------------------------------------------
    # Cleanup mode
    # ------------------------------------------------------------------
    if args.cleanup:
        cleanup_indices(client)
        return

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    if args.use_sample:
        documents = SAMPLE_DOCUMENTS
        test_queries = TEST_QUERIES
        print(f"Using {len(documents)} built-in sample documents")
    else:
        documents, test_queries = load_documents_from_shards(
            args.data_dir, args.num_docs
        )

    # ------------------------------------------------------------------
    # Load or encode vectors (with caching)
    # ------------------------------------------------------------------
    cache_dir = Path(RESULTS_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"vectors_cache_{len(documents)}.json"

    if cache_file.exists():
        print(f"Loading cached vectors from {cache_file} ...")
        with open(cache_file) as f:
            cache = json.load(f)
        doc_vecs_str = cache["doc_vecs_str"]
        doc_vecs_int = cache["doc_vecs_int"]
        query_vecs_str = cache["query_vecs_str"]
        query_vecs_int = cache["query_vecs_int"]
        print(
            f"Loaded {len(doc_vecs_str)} doc vectors, "
            f"{len(query_vecs_str)} query vectors from cache"
        )
        # Still need encoder for seismic experiment
        encoder = SparseEncoder(
            model_dir=args.model_dir, device=args.device
        )
    else:
        t0 = time.time()
        encoder = SparseEncoder(
            model_dir=args.model_dir, device=args.device
        )
        print(f"Model loaded in {time.time() - t0:.1f}s\n")

        print(f"Encoding {len(documents)} documents ...")
        t0 = time.time()
        doc_vecs_str: List[Dict[str, float]] = []
        doc_vecs_int: List[Dict[str, float]] = []
        for i, doc in enumerate(documents):
            str_vec, int_vec = encoder._encode_raw(
                doc["content"], max_length=DOC_MAX_LENGTH
            )
            doc_vecs_str.append(str_vec)
            doc_vecs_int.append(int_vec)
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(documents) - i - 1) / rate
                print(
                    f"  {i + 1}/{len(documents)} "
                    f"({rate:.0f} docs/s, ETA {eta:.0f}s)"
                )
        print(
            f"Encoded {len(documents)} documents "
            f"in {time.time() - t0:.1f}s"
        )

        print(f"Encoding {len(test_queries)} queries ...")
        t0 = time.time()
        query_vecs_str: List[Dict[str, float]] = []
        query_vecs_int: List[Dict[str, float]] = []
        for q in test_queries:
            qv_str = encoder.encode_query(q["text"])
            qv_int = encoder.encode_query_for_sparse_vector(
                q["text"]
            )
            query_vecs_str.append(qv_str)
            query_vecs_int.append(qv_int)
        print(
            f"Encoded {len(test_queries)} queries "
            f"in {time.time() - t0:.1f}s"
        )

        # Save cache
        print(f"Saving vector cache to {cache_file} ...")
        with open(cache_file, "w") as f:
            json.dump(
                {
                    "doc_vecs_str": doc_vecs_str,
                    "doc_vecs_int": doc_vecs_int,
                    "query_vecs_str": query_vecs_str,
                    "query_vecs_int": query_vecs_int,
                },
                f,
            )
        print("Cache saved.")

    # Show stats
    avg_doc_nz = sum(len(v) for v in doc_vecs_str) / len(doc_vecs_str)
    avg_q_nz = sum(len(v) for v in query_vecs_str) / len(query_vecs_str)
    print(f"\nAvg non-zero tokens: docs={avg_doc_nz:.0f}, queries={avg_q_nz:.0f}")

    # ------------------------------------------------------------------
    # Create indices and index documents
    # ------------------------------------------------------------------
    print("\n" + "-" * 90)
    print("Creating indices and indexing documents ...")
    print("-" * 90)

    create_ann_index(client)
    create_exact_index(client)
    create_rank_features_index(client)

    bulk_index_documents(client, INDEX_ANN, documents, doc_vecs_int)
    bulk_index_documents(
        client, INDEX_EXACT, documents, doc_vecs_int
    )
    bulk_index_documents(
        client, INDEX_RANK, documents, doc_vecs_str
    )

    # ------------------------------------------------------------------
    # Run experiments
    # ------------------------------------------------------------------
    all_experiments: Dict[str, Any] = {
        "endpoint": args.endpoint,
        "model_dir": args.model_dir,
        "num_documents": len(documents),
        "num_queries": len(test_queries),
        "avg_doc_nonzero": round(avg_doc_nz, 1),
        "avg_query_nonzero": round(avg_q_nz, 1),
        "experiments": {},
    }

    exp = args.experiment

    if exp in ("all", "baseline"):
        result = run_baseline_comparison(
            client, encoder, query_vecs_int, query_vecs_str,
            queries=test_queries,
        )
        all_experiments["experiments"]["baseline"] = result

    if exp in ("all", "seismic"):
        result = run_seismic_parameter_experiment(
            client, encoder, documents,
            doc_vecs_int, query_vecs_int,
        )
        all_experiments["experiments"]["seismic"] = result

    if exp in ("all", "query"):
        result = run_query_parameter_experiment(
            client, query_vecs_int
        )
        all_experiments["experiments"]["query"] = result

    if exp in ("all", "hybrid"):
        result = run_hybrid_weight_experiment(
            client, encoder, query_vecs_int,
            queries=test_queries,
        )
        all_experiments["experiments"]["hybrid"] = result

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    save_results(all_experiments)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  SUMMARY")
    print("=" * 90)
    print(f"  Endpoint:          {args.endpoint}")
    print(f"  Documents:         {len(documents)}")
    print(f"  Queries:           {len(test_queries)}")
    print(f"  Experiments run:   {list(all_experiments['experiments'].keys())}")
    print(f"  Results saved:     {RESULTS_DIR}/{RESULTS_FILE}")
    print()
    print("  Indices created:")
    print(f"    - {INDEX_ANN} (SEISMIC ANN)")
    print(f"    - {INDEX_EXACT} (exact search)")
    print(f"    - {INDEX_RANK} (rank_features)")
    print()
    print("  To cleanup: python scripts/neural_sparse_search_aws.py --cleanup")
    print()


if __name__ == "__main__":
    main()
