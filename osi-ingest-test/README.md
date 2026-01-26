  사용법:                                                                                                                       
  ./osi-periodic-ingest.sh [interval_seconds] [batch_size]                                                                      
                                                                                                                                
  # 기본값: 1초 간격, 3건씩                                                                                                    ─
  ./osi-periodic-ingest.sh                                                                                                     ─
                                                                                                                                
  # 2초 간격, 5건씩                                                                                                             
  ./osi-periodic-ingest.sh 2 5                                                                                                  
                                                                                                                                
  # 백그라운드 실행                                                                                                             
  nohup ./osi-periodic-ingest.sh 1 3 > periodic.log 2>&1 &   