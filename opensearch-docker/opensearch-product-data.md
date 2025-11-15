# OpenSearch 한국어 상품 데이터 색인

이 문서는 OpenSearch DevTool에서 바로 복사하여 사용할 수 있는 한국어 상품 데이터 색인 스크립트입니다.

## 1. 인덱스 생성 (노리 형태소 분석기 사용)


```json
PUT /_ingest/pipeline/bedrock-v2-ingest-pipeline
{
  "description": "Bedrock Titan v2 Embedding pipeline",
  "processors": [
    {
      "text_embedding": {
        "model_id": "{model id}",
        "field_map": {
          "description": "description_embedding"
        }
      }
    }
  ]
}
```

```json
PUT /products
{
  "settings": {
    "index": {
      "number_of_shards": 3,
      "number_of_replicas": 1,
      "default_pipeline": "bedrock-v2-ingest-pipeline",
      "knn": true,
      "analysis": {
        "tokenizer": {
          "nori_user_dict": {
            "type": "nori_tokenizer",
            "decompound_mode": "mixed",
            "user_dictionary_rules": [
              "갤럭시",
              "아이폰",
              "맥북"
            ]
          }
        },
        "analyzer": {
          "nori_analyzer": {
            "type": "custom",
            "tokenizer": "nori_user_dict",
            "filter": [
              "nori_posfilter"
            ]
          }
        },
        "filter": {
          "nori_posfilter": {
            "type": "nori_part_of_speech",
            "stoptags": [
              "IC", "MAG", "MAJ",
              "MM", "SP", "SSC", "SSO", "SC",
              "SE", "XPN", "XSA", "XSN", "XSV",
              "UNA", "NA", "VSV"
            ]
          }
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "product_id": {
        "type": "keyword"
      },
      "name": {
        "type": "text",
        "analyzer": "nori_analyzer",
        "fields": {
          "keyword": {
            "type": "keyword"
          }
        }
      },
      "category": {
        "type": "keyword"
      },
      "brand": {
        "type": "keyword"
      },
      "description": {
        "type": "text",
        "analyzer": "nori_analyzer"
      },
      "description_embedding":{
        "type": "knn_vector",
        "dimension": 1024,
        "method": {
          "engine": "faiss",
          "space_type": "innerproduct",
          "name": "hnsw",
          "parameters": {
            "ef_construction": 512,
            "ef_search": 512,
            "m": 32
          }
        }
      },
      "price": {
        "type": "integer"
      },
      "stock": {
        "type": "integer"
      },
      "rating": {
        "type": "float"
      },
      "review_count": {
        "type": "integer"
      },
      "tags": {
        "type": "keyword"
      },
      "created_at": {
        "type": "date"
      }
    }
  }
}
```

## 2. 더미 데이터 색인 (100개)

```json
POST /products/_bulk
{"index":{"_id":"1"}}
{"product_id":"PROD001","name":"삼성 갤럭시 S23 울트라 256GB","category":"전자제품","brand":"삼성","description":"최신 갤럭시 플래그십 스마트폰. 200MP 카메라와 강력한 성능을 자랑합니다.","price":1398000,"stock":45,"rating":4.8,"review_count":1523,"tags":["스마트폰","5G","안드로이드"],"created_at":"2024-01-15"}
{"index":{"_id":"2"}}
{"product_id":"PROD002","name":"애플 아이폰 15 Pro 128GB","category":"전자제품","brand":"애플","description":"티타늄 디자인의 프리미엄 아이폰. A17 Pro 칩 탑재.","price":1550000,"stock":32,"rating":4.9,"review_count":2104,"tags":["스마트폰","5G","iOS"],"created_at":"2024-01-20"}
{"index":{"_id":"3"}}
{"product_id":"PROD003","name":"LG 그램 17인치 노트북","category":"전자제품","brand":"LG","description":"초경량 17인치 대화면 노트북. 무게 1.35kg의 혁신.","price":2190000,"stock":28,"rating":4.7,"review_count":856,"tags":["노트북","경량","업무용"],"created_at":"2024-02-01"}
{"index":{"_id":"4"}}
{"product_id":"PROD004","name":"나이키 에어맥스 270 운동화","category":"패션","brand":"나이키","description":"편안한 쿠셔닝의 데일리 운동화. 다양한 컬러 옵션.","price":189000,"stock":120,"rating":4.6,"review_count":3421,"tags":["운동화","스포츠","캐주얼"],"created_at":"2024-02-05"}
{"index":{"_id":"5"}}
{"product_id":"PROD005","name":"아디다스 울트라부스트 22","category":"패션","brand":"아디다스","description":"최고의 러닝화. 부스트 폼 쿠셔닝 기술 적용.","price":229000,"stock":85,"rating":4.7,"review_count":1876,"tags":["러닝화","스포츠","운동"],"created_at":"2024-02-10"}
{"index":{"_id":"6"}}
{"product_id":"PROD006","name":"다이슨 V15 무선청소기","category":"가전제품","brand":"다이슨","description":"레이저 먼지 감지 기술. 강력한 흡입력의 무선청소기.","price":899000,"stock":15,"rating":4.8,"review_count":743,"tags":["청소기","무선","프리미엄"],"created_at":"2024-02-12"}
{"index":{"_id":"7"}}
{"product_id":"PROD007","name":"삼성 비스포크 냉장고 4도어","category":"가전제품","brand":"삼성","description":"맞춤형 컬러 패널의 프리미엄 냉장고. 870L 대용량.","price":3290000,"stock":8,"rating":4.6,"review_count":412,"tags":["냉장고","대용량","비스포크"],"created_at":"2024-02-15"}
{"index":{"_id":"8"}}
{"product_id":"PROD008","name":"소니 WH-1000XM5 노이즈캔슬링 헤드폰","category":"전자제품","brand":"소니","description":"업계 최고 수준의 노이즈 캔슬링. 30시간 재생.","price":449000,"stock":67,"rating":4.9,"review_count":2876,"tags":["헤드폰","무선","노이즈캔슬링"],"created_at":"2024-02-18"}
{"index":{"_id":"9"}}
{"product_id":"PROD009","name":"로지텍 MX Master 3S 무선마우스","category":"전자제품","brand":"로지텍","description":"프로페셔널 무선마우스. 조용한 클릭과 정밀한 트래킹.","price":139000,"stock":156,"rating":4.7,"review_count":1234,"tags":["마우스","무선","업무용"],"created_at":"2024-02-20"}
{"index":{"_id":"10"}}
{"product_id":"PROD010","name":"코카콜라 제로 355ml 24캔","category":"식품","brand":"코카콜라","description":"제로 칼로리 탄산음료. 상쾌한 맛은 그대로.","price":18900,"stock":450,"rating":4.5,"review_count":8932,"tags":["음료","탄산","제로칼로리"],"created_at":"2024-02-22"}
{"index":{"_id":"11"}}
{"product_id":"PROD011","name":"농심 신라면 5입","category":"식품","brand":"농심","description":"한국을 대표하는 매운 라면. 얼큰한 국물맛.","price":4500,"stock":890,"rating":4.8,"review_count":15234,"tags":["라면","즉석식품","매운맛"],"created_at":"2024-02-25"}
{"index":{"_id":"12"}}
{"product_id":"PROD012","name":"스타벅스 하우스 블렌드 원두 250g","category":"식품","brand":"스타벅스","description":"밸런스 좋은 미디엄 로스트 원두. 부드러운 맛.","price":13900,"stock":234,"rating":4.6,"review_count":3421,"tags":["커피","원두","홈카페"],"created_at":"2024-03-01"}
{"index":{"_id":"13"}}
{"product_id":"PROD013","name":"에어팟 프로 2세대","category":"전자제품","brand":"애플","description":"향상된 노이즈 캔슬링. 공간 오디오 지원.","price":359000,"stock":78,"rating":4.8,"review_count":4567,"tags":["이어폰","무선","노이즈캔슬링"],"created_at":"2024-03-05"}
{"index":{"_id":"14"}}
{"product_id":"PROD014","name":"무인양품 초음파 아로마 디퓨저","category":"생활용품","brand":"무인양품","description":"심플한 디자인의 아로마 디퓨저. 조용한 작동.","price":49000,"stock":142,"rating":4.5,"review_count":892,"tags":["디퓨저","아로마","인테리어"],"created_at":"2024-03-08"}
{"index":{"_id":"15"}}
{"product_id":"PROD015","name":"이케아 POÄNG 안락의자","category":"가구","brand":"이케아","description":"자작나무 프레임의 편안한 안락의자. 다양한 커버.","price":129000,"stock":45,"rating":4.7,"review_count":2341,"tags":["의자","안락의자","북유럽스타일"],"created_at":"2024-03-10"}
{"index":{"_id":"16"}}
{"product_id":"PROD016","name":"한샘 모던 책상 1200","category":"가구","brand":"한샘","description":"심플한 디자인의 원목 책상. 재택근무에 최적.","price":289000,"stock":34,"rating":4.6,"review_count":678,"tags":["책상","원목","재택근무"],"created_at":"2024-03-12"}
{"index":{"_id":"17"}}
{"product_id":"PROD017","name":"시디즈 T50 메쉬 의자","category":"가구","brand":"시디즈","description":"허리를 편안하게 받쳐주는 인체공학 의자.","price":549000,"stock":23,"rating":4.8,"review_count":1567,"tags":["의자","메쉬","인체공학"],"created_at":"2024-03-15"}
{"index":{"_id":"18"}}
{"product_id":"PROD018","name":"필립스 에어프라이어 XXL","category":"가전제품","brand":"필립스","description":"7.3L 대용량 에어프라이어. 기름 없이 바삭하게.","price":429000,"stock":56,"rating":4.7,"review_count":3892,"tags":["에어프라이어","주방가전","건강요리"],"created_at":"2024-03-18"}
{"index":{"_id":"19"}}
{"product_id":"PROD019","name":"쿠쿠 10인용 압력밥솥","category":"가전제품","brand":"쿠쿠","description":"스테인리스 내솥의 IH 압력밥솥. 맛있는 밥.","price":389000,"stock":42,"rating":4.8,"review_count":5234,"tags":["밥솥","압력밥솥","주방가전"],"created_at":"2024-03-20"}
{"index":{"_id":"20"}}
{"product_id":"PROD020","name":"테팔 인덕션 프라이팬 28cm","category":"주방용품","brand":"테팔","description":"티타늄 프로 코팅. 인덕션 겸용 프라이팬.","price":79000,"stock":189,"rating":4.5,"review_count":2876,"tags":["프라이팬","논스틱","인덕션"],"created_at":"2024-03-22"}
{"index":{"_id":"21"}}
{"product_id":"PROD021","name":"락앤락 밀폐용기 10종 세트","category":"주방용품","brand":"락앤락","description":"4면 잠금 밀폐용기. 다양한 크기 구성.","price":35900,"stock":267,"rating":4.6,"review_count":4123,"tags":["밀폐용기","보관용기","주방"],"created_at":"2024-03-25"}
{"index":{"_id":"22"}}
{"product_id":"PROD022","name":"유니클로 히트텍 라운드넥 티셔츠","category":"패션","brand":"유니클로","description":"발열 기능의 이너웨어. 겨울 필수 아이템.","price":19900,"stock":534,"rating":4.7,"review_count":8765,"tags":["히트텍","이너웨어","겨울"],"created_at":"2024-03-28"}
{"index":{"_id":"23"}}
{"product_id":"PROD023","name":"자라 오버사이즈 코트","category":"패션","brand":"자라","description":"트렌디한 오버핏 코트. 울 혼방 소재.","price":159000,"stock":78,"rating":4.5,"review_count":1234,"tags":["코트","아우터","겨울"],"created_at":"2024-04-01"}
{"index":{"_id":"24"}}
{"product_id":"PROD024","name":"리바이스 501 오리지널 청바지","category":"패션","brand":"리바이스","description":"클래식한 정통 청바지. 스트레이트 핏.","price":129000,"stock":156,"rating":4.8,"review_count":3421,"tags":["청바지","데님","클래식"],"created_at":"2024-04-03"}
{"index":{"_id":"25"}}
{"product_id":"PROD025","name":"닌텐도 스위치 OLED 모델","category":"전자제품","brand":"닌텐도","description":"7인치 OLED 화면. 향상된 오디오와 스탠드.","price":429000,"stock":67,"rating":4.9,"review_count":2345,"tags":["게임기","닌텐도","휴대용"],"created_at":"2024-04-05"}
{"index":{"_id":"26"}}
{"product_id":"PROD026","name":"플레이스테이션 5 디지털 에디션","category":"전자제품","brand":"소니","description":"차세대 게임 콘솔. 초고속 SSD 탑재.","price":519000,"stock":23,"rating":4.8,"review_count":1876,"tags":["게임기","플스5","콘솔"],"created_at":"2024-04-08"}
{"index":{"_id":"27"}}
{"product_id":"PROD027","name":"삼성 갤럭시 탭 S9 11인치","category":"전자제품","brand":"삼성","description":"강력한 성능의 안드로이드 태블릿. S펜 포함.","price":999000,"stock":45,"rating":4.7,"review_count":987,"tags":["태블릿","안드로이드","S펜"],"created_at":"2024-04-10"}
{"index":{"_id":"28"}}
{"product_id":"PROD028","name":"애플 아이패드 에어 5세대","category":"전자제품","brand":"애플","description":"M1 칩 탑재 태블릿. 10.9인치 Liquid Retina.","price":929000,"stock":38,"rating":4.8,"review_count":1543,"tags":["태블릿","iOS","M1"],"created_at":"2024-04-12"}
{"index":{"_id":"29"}}
{"product_id":"PROD029","name":"샤오미 공기청정기 4 Pro","category":"가전제품","brand":"샤오미","description":"500m³/h CADR. 스마트 미세먼지 센서.","price":329000,"stock":89,"rating":4.6,"review_count":2134,"tags":["공기청정기","미세먼지","스마트가전"],"created_at":"2024-04-15"}
{"index":{"_id":"30"}}
{"product_id":"PROD030","name":"코웨이 정수기 렌탈형","category":"가전제품","brand":"코웨이","description":"직수형 정수기. 얼음 정수 온수 모두 가능.","price":49900,"stock":0,"rating":4.7,"review_count":5678,"tags":["정수기","렌탈","냉온수"],"created_at":"2024-04-18"}
{"index":{"_id":"31"}}
{"product_id":"PROD031","name":"버버리 체크 머플러","category":"패션","brand":"버버리","description":"시그니처 체크 패턴. 100% 캐시미어.","price":589000,"stock":12,"rating":4.9,"review_count":234,"tags":["머플러","명품","캐시미어"],"created_at":"2024-04-20"}
{"index":{"_id":"32"}}
{"product_id":"PROD032","name":"구찌 마몬트 숄더백","category":"패션","brand":"구찌","description":"퀼팅 레더 숄더백. 골드 체인 스트랩.","price":2890000,"stock":5,"rating":4.8,"review_count":156,"tags":["가방","명품","숄더백"],"created_at":"2024-04-22"}
{"index":{"_id":"33"}}
{"product_id":"PROD033","name":"루이비통 모노그램 지갑","category":"패션","brand":"루이비통","description":"클래식 모노그램 캔버스. 장지갑.","price":890000,"stock":8,"rating":4.7,"review_count":345,"tags":["지갑","명품","모노그램"],"created_at":"2024-04-25"}
{"index":{"_id":"34"}}
{"product_id":"PROD034","name":"에르메스 H 벨트","category":"패션","brand":"에르메스","description":"리버서블 레더 벨트. 시그니처 H 버클.","price":1290000,"stock":3,"rating":4.9,"review_count":89,"tags":["벨트","명품","레더"],"created_at":"2024-04-28"}
{"index":{"_id":"35"}}
{"product_id":"PROD035","name":"맥북 에어 M2 13인치","category":"전자제품","brand":"애플","description":"M2 칩 탑재. 얇고 가벼운 노트북.","price":1590000,"stock":45,"rating":4.9,"review_count":3421,"tags":["노트북","맥북","M2"],"created_at":"2024-05-01"}
{"index":{"_id":"36"}}
{"product_id":"PROD036","name":"맥북 프로 14인치 M3 Pro","category":"전자제품","brand":"애플","description":"프로페셔널을 위한 노트북. 18GB 통합메모리.","price":2990000,"stock":18,"rating":4.9,"review_count":876,"tags":["노트북","맥북","M3"],"created_at":"2024-05-03"}
{"index":{"_id":"37"}}
{"product_id":"PROD037","name":"델 울트라샤프 27인치 모니터","category":"전자제품","brand":"델","description":"4K UHD IPS 모니터. USB-C 허브 기능.","price":789000,"stock":34,"rating":4.7,"review_count":1234,"tags":["모니터","4K","USB-C"],"created_at":"2024-05-05"}
{"index":{"_id":"38"}}
{"product_id":"PROD038","name":"LG 울트라기어 게이밍 모니터 32인치","category":"전자제품","brand":"LG","description":"165Hz 주사율. 1ms 응답속도. QHD 해상도.","price":599000,"stock":56,"rating":4.8,"review_count":2341,"tags":["모니터","게이밍","고주사율"],"created_at":"2024-05-08"}
{"index":{"_id":"39"}}
{"product_id":"PROD039","name":"레이저 블랙위도우 V4 키보드","category":"전자제품","brand":"레이저","description":"기계식 게이밍 키보드. RGB 백라이트.","price":219000,"stock":78,"rating":4.6,"review_count":1567,"tags":["키보드","기계식","게이밍"],"created_at":"2024-05-10"}
{"index":{"_id":"40"}}
{"product_id":"PROD040","name":"코르세어 K70 RGB 키보드","category":"전자제품","brand":"코르세어","description":"체리 MX 스위치. 알루미늄 프레임.","price":189000,"stock":92,"rating":4.7,"review_count":987,"tags":["키보드","기계식","RGB"],"created_at":"2024-05-12"}
{"index":{"_id":"41"}}
{"product_id":"PROD041","name":"한성 모니터암 듀얼","category":"사무용품","brand":"한성","description":"2개 모니터 거치 가능. 가스스프링 방식.","price":89000,"stock":123,"rating":4.5,"review_count":2134,"tags":["모니터암","듀얼","사무용품"],"created_at":"2024-05-15"}
{"index":{"_id":"42"}}
{"product_id":"PROD042","name":"3M 손목보호 마우스패드","category":"사무용품","brand":"3M","description":"젤 손목받침대. 편안한 마우스 사용.","price":19900,"stock":456,"rating":4.4,"review_count":3421,"tags":["마우스패드","손목보호","사무용품"],"created_at":"2024-05-18"}
{"index":{"_id":"43"}}
{"product_id":"PROD043","name":"모나미 153 볼펜 12자루","category":"사무용품","brand":"모나미","description":"한국의 국민 볼펜. 부드러운 필기감.","price":6000,"stock":789,"rating":4.8,"review_count":15234,"tags":["볼펜","필기구","사무용품"],"created_at":"2024-05-20"}
{"index":{"_id":"44"}}
{"product_id":"PROD044","name":"라미 사파리 만년필","category":"사무용품","brand":"라미","description":"경량 ABS 바디. 입문용 만년필로 최적.","price":35000,"stock":234,"rating":4.7,"review_count":876,"tags":["만년필","필기구","프리미엄"],"created_at":"2024-05-22"}
{"index":{"_id":"45"}}
{"product_id":"PROD045","name":"몰스킨 클래식 노트 Large","category":"사무용품","brand":"몰스킨","description":"하드커버 룰드 노트. 192페이지.","price":29000,"stock":345,"rating":4.6,"review_count":1234,"tags":["노트","다이어리","프리미엄"],"created_at":"2024-05-25"}
{"index":{"_id":"46"}}
{"product_id":"PROD046","name":"CJ 햇반 즉석밥 210g 24개","category":"식품","brand":"CJ","description":"전자레인지 2분이면 갓 지은 밥맛.","price":23900,"stock":678,"rating":4.6,"review_count":12345,"tags":["즉석밥","간편식","쌀"],"created_at":"2024-05-28"}
{"index":{"_id":"47"}}
{"product_id":"PROD047","name":"풀무원 탱탱쫄면 4개입","category":"식품","brand":"풀무원","description":"쫄깃한 면발의 비빔쫄면. 매콤달콤한 소스.","price":5900,"stock":890,"rating":4.7,"review_count":8765,"tags":["쫄면","즉석식품","비빔"],"created_at":"2024-06-01"}
{"index":{"_id":"48"}}
{"product_id":"PROD048","name":"오뚜기 진라면 매운맛 5개입","category":"식품","brand":"오뚜기","description":"깔끔한 매운맛. 쫄깃한 면발의 라면.","price":4200,"stock":1234,"rating":4.5,"review_count":23456,"tags":["라면","즉석식품","매운맛"],"created_at":"2024-06-03"}
{"index":{"_id":"49"}}
{"product_id":"PROD049","name":"동원 참치캔 살코기 100g 10개","category":"식품","brand":"동원","description":"순살 참치. 샐러드나 김밥에 활용.","price":17900,"stock":567,"rating":4.6,"review_count":5678,"tags":["참치","캔","단백질"],"created_at":"2024-06-05"}
{"index":{"_id":"50"}}
{"product_id":"PROD050","name":"오뚜기 케찹 500g","category":"식품","brand":"오뚜기","description":"토마토의 새콤달콤한 맛. 다양한 요리에.","price":3500,"stock":890,"rating":4.5,"review_count":4321,"tags":["케찹","소스","조미료"],"created_at":"2024-06-08"}
{"index":{"_id":"51"}}
{"product_id":"PROD051","name":"청정원 순창 고추장 1kg","category":"식품","brand":"청정원","description":"전통 방식 발효. 깊은 맛의 고추장.","price":8900,"stock":456,"rating":4.7,"review_count":3421,"tags":["고추장","장류","전통"],"created_at":"2024-06-10"}
{"index":{"_id":"52"}}
{"product_id":"PROD052","name":"샘표 진간장 골드 860ml","category":"식품","brand":"샘표","description":"천연양조 간장. 은은한 단맛.","price":7500,"stock":678,"rating":4.6,"review_count":5678,"tags":["간장","양조간장","조미료"],"created_at":"2024-06-12"}
{"index":{"_id":"53"}}
{"product_id":"PROD053","name":"백설 요리올리고당 1.2kg","category":"식품","brand":"백설","description":"설탕 대체 천연 감미료. 건강한 단맛.","price":9900,"stock":345,"rating":4.5,"review_count":2345,"tags":["올리고당","감미료","건강"],"created_at":"2024-06-15"}
{"index":{"_id":"54"}}
{"product_id":"PROD054","name":"종가집 포기김치 3kg","category":"식품","brand":"종가집","description":"국내산 배추. 전통 방식으로 담근 김치.","price":23900,"stock":234,"rating":4.7,"review_count":6789,"tags":["김치","발효식품","한식"],"created_at":"2024-06-18"}
{"index":{"_id":"55"}}
{"product_id":"PROD055","name":"곰곰 유기농 우유 1L","category":"식품","brand":"곰곰","description":"국내산 유기농 원유 100%. 신선한 우유.","price":4500,"stock":567,"rating":4.8,"review_count":4567,"tags":["우유","유기농","유제품"],"created_at":"2024-06-20"}
{"index":{"_id":"56"}}
{"product_id":"PROD056","name":"서울우유 바나나맛 우유 200ml 6개","category":"식품","brand":"서울우유","description":"달콤한 바나나향. 어린이가 좋아하는 맛.","price":5400,"stock":789,"rating":4.7,"review_count":8901,"tags":["우유","가공유","바나나"],"created_at":"2024-06-22"}
{"index":{"_id":"57"}}
{"product_id":"PROD057","name":"매일 상하목장 유기농 요구르트","category":"식품","brand":"매일","description":"유산균이 살아있는 유기농 요구르트.","price":6900,"stock":456,"rating":4.6,"review_count":3456,"tags":["요구르트","유기농","유산균"],"created_at":"2024-06-25"}
{"index":{"_id":"58"}}
{"product_id":"PROD058","name":"빙그레 바나나맛 우유 240ml","category":"식품","brand":"빙그레","description":"50년 전통의 국민 음료. 달콤한 바나나맛.","price":1500,"stock":1234,"rating":4.8,"review_count":23456,"tags":["우유","가공유","국민음료"],"created_at":"2024-06-28"}
{"index":{"_id":"59"}}
{"product_id":"PROD059","name":"남양 프렌치카페 카페믹스 100개입","category":"식품","brand":"남양","description":"부드러운 믹스커피. 대용량 구성.","price":19900,"stock":345,"rating":4.5,"review_count":5678,"tags":["커피","믹스커피","대용량"],"created_at":"2024-07-01"}
{"index":{"_id":"60"}}
{"product_id":"PROD060","name":"맥심 모카골드 커피믹스 200개입","category":"식품","brand":"동서식품","description":"고급 아라비카 원두. 부드러운 맛.","price":35900,"stock":267,"rating":4.6,"review_count":12345,"tags":["커피","믹스커피","선물세트"],"created_at":"2024-07-03"}
{"index":{"_id":"61"}}
{"product_id":"PROD061","name":"네스카페 돌체구스토 캡슐 아메리카노","category":"식품","brand":"네스카페","description":"캡슐 커피머신용. 16개입 아메리카노.","price":8900,"stock":456,"rating":4.5,"review_count":2345,"tags":["커피","캡슐","아메리카노"],"created_at":"2024-07-05"}
{"index":{"_id":"62"}}
{"product_id":"PROD062","name":"일리 에스프레소 원두 250g","category":"식품","brand":"일리","description":"이탈리아 프리미엄 원두. 다크 로스트.","price":17900,"stock":189,"rating":4.7,"review_count":876,"tags":["커피","원두","에스프레소"],"created_at":"2024-07-08"}
{"index":{"_id":"63"}}
{"product_id":"PROD063","name":"테팔 매직핸즈 냄비세트 5종","category":"주방용품","brand":"테팔","description":"인덕션 겸용 냄비. 손잡이 분리형.","price":149000,"stock":78,"rating":4.6,"review_count":1234,"tags":["냄비","주방용품","세트"],"created_at":"2024-07-10"}
{"index":{"_id":"64"}}
{"product_id":"PROD064","name":"쿠진아트 핸드블렌더","category":"주방가전","brand":"쿠진아트","description":"200W 파워풀한 믹서. 다양한 악세서리.","price":89000,"stock":123,"rating":4.5,"review_count":987,"tags":["믹서","블렌더","주방가전"],"created_at":"2024-07-12"}
{"index":{"_id":"65"}}
{"product_id":"PROD065","name":"키친에이드 스탠드믹서","category":"주방가전","brand":"키친에이드","description":"300W 강력한 베이킹 믹서. 10단 조절.","price":589000,"stock":34,"rating":4.8,"review_count":543,"tags":["믹서","베이킹","프리미엄"],"created_at":"2024-07-15"}
{"index":{"_id":"66"}}
{"product_id":"PROD066","name":"브레빌 에스프레소 머신","category":"주방가전","brand":"브레빌","description":"15바 펌프 압력. 바리스타급 에스프레소.","price":899000,"stock":23,"rating":4.9,"review_count":234,"tags":["커피머신","에스프레소","프리미엄"],"created_at":"2024-07-18"}
{"index":{"_id":"67"}}
{"product_id":"PROD067","name":"드롱기 전자동 커피머신","category":"주방가전","brand":"드롱기","description":"원두부터 추출까지 원터치. 우유 거품기 내장.","price":1290000,"stock":15,"rating":4.8,"review_count":432,"tags":["커피머신","전자동","라떼"],"created_at":"2024-07-20"}
{"index":{"_id":"68"}}
{"product_id":"PROD068","name":"비타믹스 블렌더 프로페셔널","category":"주방가전","brand":"비타믹스","description":"2HP 모터. 스무디부터 수프까지.","price":789000,"stock":28,"rating":4.7,"review_count":654,"tags":["블렌더","스무디","프리미엄"],"created_at":"2024-07-22"}
{"index":{"_id":"69"}}
{"product_id":"PROD069","name":"필립스 전기면도기 9000 시리즈","category":"생활가전","brand":"필립스","description":"AI 센서 탑재. 습식 건식 겸용.","price":389000,"stock":67,"rating":4.6,"review_count":1234,"tags":["면도기","전기면도기","습식"],"created_at":"2024-07-25"}
{"index":{"_id":"70"}}
{"product_id":"PROD070","name":"브라운 전기면도기 시리즈 9","category":"생활가전","brand":"브라운","description":"5중날 시스템. 피부 보호 기술.","price":429000,"stock":45,"rating":4.7,"review_count":876,"tags":["면도기","전기면도기","프리미엄"],"created_at":"2024-07-28"}
{"index":{"_id":"71"}}
{"product_id":"PROD071","name":"파나소닉 전동칫솔 돌츠","category":"생활가전","brand":"파나소닉","description":"초음파 진동. 치아와 잇몸 케어.","price":159000,"stock":89,"rating":4.5,"review_count":2345,"tags":["전동칫솔","구강케어","초음파"],"created_at":"2024-08-01"}
{"index":{"_id":"72"}}
{"product_id":"PROD072","name":"오랄비 전동칫솔 지니어스 X","category":"생활가전","brand":"오랄비","description":"AI 위치 감지. 블루투스 연결.","price":189000,"stock":78,"rating":4.6,"review_count":1567,"tags":["전동칫솔","AI","스마트"],"created_at":"2024-08-03"}
{"index":{"_id":"73"}}
{"product_id":"PROD073","name":"샤오미 전동칫솔 T500","category":"생활가전","brand":"샤오미","description":"31,000회/분 진동. 무선충전.","price":49000,"stock":234,"rating":4.4,"review_count":3421,"tags":["전동칫솔","무선충전","가성비"],"created_at":"2024-08-05"}
{"index":{"_id":"74"}}
{"product_id":"PROD074","name":"일룸 책상 린백 1500","category":"가구","brand":"일룸","description":"고급 원목 책상. 넓은 작업 공간.","price":489000,"stock":23,"rating":4.7,"review_count":456,"tags":["책상","원목","프리미엄"],"created_at":"2024-08-08"}
{"index":{"_id":"75"}}
{"product_id":"PROD075","name":"에이스침대 BMA 1033-N 매트리스","category":"가구","brand":"에이스침대","description":"7존 독립 포켓스프링. Q사이즈.","price":1890000,"stock":12,"rating":4.8,"review_count":234,"tags":["매트리스","침대","독립스프링"],"created_at":"2024-08-10"}
{"index":{"_id":"76"}}
{"product_id":"PROD076","name":"시몬스 뷰티레스트 블랙 매트리스","category":"가구","brand":"시몬스","description":"마이크로 포켓코일. 최고급 매트리스.","price":3290000,"stock":8,"rating":4.9,"review_count":156,"tags":["매트리스","침대","프리미엄"],"created_at":"2024-08-12"}
{"index":{"_id":"77"}}
{"product_id":"PROD077","name":"템퍼 오리지널 베개","category":"침구","brand":"템퍼","description":"NASA 기술 메모리폼. 목 받침 최적화.","price":189000,"stock":67,"rating":4.6,"review_count":987,"tags":["베개","메모리폼","수면"],"created_at":"2024-08-15"}
{"index":{"_id":"78"}}
{"product_id":"PROD078","name":"마이크로화이버 침구세트 퀸","category":"침구","brand":"한샘","description":"부드러운 촉감. 사계절 사용 가능.","price":89000,"stock":145,"rating":4.5,"review_count":2345,"tags":["침구","이불","세트"],"created_at":"2024-08-18"}
{"index":{"_id":"79"}}
{"product_id":"PROD079","name":"구스다운 거위털 이불 킹","category":"침구","brand":"프리미엄침구","description":"95% 거위털. 초경량 따뜻한 겨울이불.","price":459000,"stock":34,"rating":4.8,"review_count":543,"tags":["이불","거위털","겨울"],"created_at":"2024-08-20"}
{"index":{"_id":"80"}}
{"product_id":"PROD080","name":"대림바스 욕실장 800","category":"가구","brand":"대림바스","description":"방수 MDF. 세면대 수납장 일체형.","price":389000,"stock":28,"rating":4.5,"review_count":678,"tags":["욕실장","수납","방수"],"created_at":"2024-08-22"}
{"index":{"_id":"81"}}
{"product_id":"PROD081","name":"한샘 시스템 선반 5단","category":"가구","brand":"한샘","description":"조립식 철제 선반. 높이 조절 가능.","price":79000,"stock":189,"rating":4.6,"review_count":3421,"tags":["선반","수납","철제"],"created_at":"2024-08-25"}
{"index":{"_id":"82"}}
{"product_id":"PROD082","name":"이케아 KALLAX 선반유닛 4x4","category":"가구","brand":"이케아","description":"16칸 큐브 선반. 다용도 수납.","price":149000,"stock":56,"rating":4.7,"review_count":2876,"tags":["선반","수납","큐브"],"created_at":"2024-08-28"}
{"index":{"_id":"83"}}
{"product_id":"PROD083","name":"무인양품 PP 수납박스 딥형","category":"생활용품","brand":"무인양품","description":"반투명 플라스틱. 쌓아서 사용 가능.","price":15900,"stock":456,"rating":4.5,"review_count":4567,"tags":["수납박스","정리","플라스틱"],"created_at":"2024-09-01"}
{"index":{"_id":"84"}}
{"product_id":"PROD084","name":"다이소 정리함 3개 세트","category":"생활용품","brand":"다이소","description":"투명 아크릴 정리함. 서랍형.","price":5000,"stock":890,"rating":4.3,"review_count":12345,"tags":["정리함","수납","아크릴"],"created_at":"2024-09-03"}
{"index":{"_id":"85"}}
{"product_id":"PROD085","name":"3M 스카치브라이트 막대걸레","category":"생활용품","brand":"3M","description":"360도 회전. 극세사 패드.","price":29900,"stock":234,"rating":4.6,"review_count":5678,"tags":["걸레","청소","회전"],"created_at":"2024-09-05"}
{"index":{"_id":"86"}}
{"product_id":"PROD086","name":"유한킴벌리 크리넥스 화장지 30롤","category":"생활용품","brand":"유한킴벌리","description":"3겹 두루마리 화장지. 부드러운 촉감.","price":23900,"stock":567,"rating":4.5,"review_count":8901,"tags":["화장지","생필품","3겹"],"created_at":"2024-09-08"}
{"index":{"_id":"87"}}
{"product_id":"PROD087","name":"깨끗한나라 미용티슈 180매 6개","category":"생활용품","brand":"깨끗한나라","description":"보습 로션 티슈. 피부 자극 최소화.","price":9900,"stock":789,"rating":4.4,"review_count":6789,"tags":["티슈","미용티슈","보습"],"created_at":"2024-09-10"}
{"index":{"_id":"88"}}
{"product_id":"PROD088","name":"페브리즈 섬유탈취제 800ml","category":"생활용품","brand":"P&G","description":"옷감 침구 소파 탈취. 무향.","price":12900,"stock":345,"rating":4.6,"review_count":4321,"tags":["탈취제","섬유","스프레이"],"created_at":"2024-09-12"}
{"index":{"_id":"89"}}
{"product_id":"PROD089","name":"리셋 섬유유연제 리필 2L","category":"생활용품","brand":"LG생활건강","description":"은은한 향. 정전기 방지.","price":8900,"stock":456,"rating":4.5,"review_count":7890,"tags":["유연제","세탁","리필"],"created_at":"2024-09-15"}
{"index":{"_id":"90"}}
{"product_id":"PROD090","name":"테크 고농축 세탁세제 3.5L","category":"생활용품","brand":"LG생활건강","description":"소량으로 강력세척. 액체세제.","price":15900,"stock":567,"rating":4.6,"review_count":9876,"tags":["세제","세탁","고농축"],"created_at":"2024-09-18"}
{"index":{"_id":"91"}}
{"product_id":"PROD091","name":"샤프란 주방세제 1L 3개","category":"생활용품","brand":"애경산업","description":"기름때 제거. 손 보호 성분.","price":9900,"stock":678,"rating":4.4,"review_count":5432,"tags":["주방세제","설거지","세제"],"created_at":"2024-09-20"}
{"index":{"_id":"92"}}
{"product_id":"PROD092","name":"락스 표백제 4L","category":"생활용품","brand":"유한양행","description":"살균 표백 소독. 욕실 주방 청소.","price":6900,"stock":890,"rating":4.5,"review_count":6543,"tags":["표백제","살균","청소"],"created_at":"2024-09-22"}
{"index":{"_id":"93"}}
{"product_id":"PROD093","name":"크린랩 위생장갑 100매","category":"생활용품","brand":"크린랩","description":"일회용 PE장갑. 주방 청소용.","price":3900,"stock":1234,"rating":4.3,"review_count":8765,"tags":["장갑","일회용","위생"],"created_at":"2024-09-25"}
{"index":{"_id":"94"}}
{"product_id":"PROD094","name":"지퍼락 냉동 보관백 대형 25매","category":"생활용품","brand":"지퍼락","description":"이중 지퍼. 냉동실 보관용.","price":7900,"stock":567,"rating":4.6,"review_count":4321,"tags":["지퍼백","보관","냉동"],"created_at":"2024-09-28"}
{"index":{"_id":"95"}}
{"product_id":"PROD095","name":"크린랩 비닐봉투 검정 50L 50매","category":"생활용품","brand":"크린랩","description":"두꺼운 재질. 음식물쓰레기용.","price":5900,"stock":789,"rating":4.4,"review_count":7654,"tags":["비닐봉투","쓰레기봉투","대용량"],"created_at":"2024-10-01"}
{"index":{"_id":"96"}}
{"product_id":"PROD096","name":"바세린 핸드크림 200ml","category":"화장품","brand":"바세린","description":"집중 보습 핸드크림. 건조한 손 케어.","price":8900,"stock":345,"rating":4.5,"review_count":3456,"tags":["핸드크림","보습","바디케어"],"created_at":"2024-10-03"}
{"index":{"_id":"97"}}
{"product_id":"PROD097","name":"니베아 크림 400ml 대용량","category":"화장품","brand":"니베아","description":"온가족 보습크림. 얼굴 바디 겸용.","price":12900,"stock":456,"rating":4.6,"review_count":8901,"tags":["크림","보습","바디"],"created_at":"2024-10-05"}
{"index":{"_id":"98"}}
{"product_id":"PROD098","name":"설화수 윤조에센스 60ml","category":"화장품","brand":"설화수","description":"한방 명품 에센스. 피부 탄력 개선.","price":189000,"stock":78,"rating":4.8,"review_count":1234,"tags":["에센스","한방","프리미엄"],"created_at":"2024-10-08"}
{"index":{"_id":"99"}}
{"product_id":"PROD099","name":"후 천기단 크림 60ml","category":"화장품","brand":"후","description":"궁중 비방 크림. 안티에이징.","price":329000,"stock":45,"rating":4.9,"review_count":876,"tags":["크림","한방","안티에이징"],"created_at":"2024-10-10"}
{"index":{"_id":"100"}}
{"product_id":"PROD100","name":"라네즈 워터뱅크 세트","category":"화장품","brand":"라네즈","description":"수분 공급 4종 세트. 촉촉한 피부.","price":89000,"stock":123,"rating":4.7,"review_count":5678,"tags":["스킨케어","세트","수분"],"created_at":"2024-10-12"}
```

## 3. 인덱스 확인

```json
GET /products/_mapping
```

## 4. 데이터 검색 예제

### 4.1 기본 검색 (전체 조회)
```json
GET /products/_search
{
  "query": {
    "match_all": {}
  }
}
```

### 4.2 한국어 상품명 검색
```json
GET /products/_search
{
  "query": {
    "match": {
      "name": "갤럭시 스마트폰"
    }
  }
}
```

### 4.3 카테고리별 검색
```json
GET /products/_search
{
  "query": {
    "term": {
      "category": "전자제품"
    }
  }
}
```

### 4.4 가격 범위 검색
```json
GET /products/_search
{
  "query": {
    "range": {
      "price": {
        "gte": 100000,
        "lte": 500000
      }
    }
  }
}
```

### 4.5 복합 검색 (한국어 + 가격 + 평점)
```json
GET /products/_search
{
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "description": "무선"
          }
        }
      ],
      "filter": [
        {
          "range": {
            "price": {
              "lte": 500000
            }
          }
        },
        {
          "range": {
            "rating": {
              "gte": 4.5
            }
          }
        }
      ]
    }
  }
}
```

### 4.6 노리 분석기 테스트
```json
GET /products/_analyze
{
  "analyzer": "nori_analyzer",
  "text": "삼성 갤럭시 스마트폰"
}
```

## 5. 인덱스 삭제 (필요시)

```json
DELETE /products
```

## 참고사항

- 노리(Nori) 형태소 분석기는 한국어 텍스트를 효과적으로 분석합니다
- `decompound_mode: "mixed"`는 복합명사를 원형과 분해형 모두로 분석합니다
- 100개의 더미 데이터는 다양한 카테고리(전자제품, 패션, 가전제품, 식품, 가구, 생활용품, 화장품 등)를 포함합니다
- 각 상품에는 ID, 이름, 카테고리, 브랜드, 설명, 가격, 재고, 평점, 리뷰 수, 태그, 생성일이 포함됩니다
- `_bulk` API를 사용하여 한 번에 모든 데이터를 색인할 수 있습니다
