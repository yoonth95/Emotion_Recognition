# Emotion_Recognition

[Data-Analysis_Object-Detection](https://github.com/yoonth95/Data-Analysis_Object-Detection) 의 프로젝트에서 사용한 Emotion Recognition 입니다.

![result_image](https://user-images.githubusercontent.com/78673090/140088202-5a77b496-f7a7-4137-af25-cd9f253f0f8e.jpg)

![Emotion_Recognition](https://user-images.githubusercontent.com/78673090/140088236-1c2aa280-8374-4f2c-92f6-85828af96b59.gif)


## 1. [FER_CNN](https://github.com/yoonth95/Emotion_Recognition/blob/master/FER_CNN.ipynb)
- Dataset : [Facial-Expression-Recognition](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- Kaggle의 FER2013 데이터를 사용
- CNN 과정 진행
- Augmentation을 사용하여 정확도 향상
- callback으로 모델 저장
<br>

![제목 없음](https://user-images.githubusercontent.com/78673090/140089300-30ae28fb-7f49-4cd7-a2c8-cfb0f1e34b08.png)
<br>

![제목 없음2](https://user-images.githubusercontent.com/78673090/140089339-ca037491-a92f-4796-983d-f75a74d9f36a.png)
<br>

## 2. [MTCNN_Image_Video](https://github.com/yoonth95/Emotion_Recognition/blob/master/MTCNN_Image_Video.ipynb)
- Haar Cascade Classifier은 속도는 빠르지만 정확도는 부족
- MTCNN 모듈은 속도는 조금 느리지만 정확도는 높음
- 얼굴 인식의 정확도를 높이고자 MTCNN 모듈 사용
- 학습시킨 모델로 얼굴의 표정 감지

## 3. [Mediapipe_Webcam](https://github.com/yoonth95/Emotion_Recognition/blob/master/Mediapipe_Webcam.py)
- Google에서 제공하는 Mediapipe 모델로 경량 모델이라 CPU 및 모바일 장치에서도 사용가능
- 다른 얼굴 인식 모듈에 비해 속도가 빠르고 얼굴 인식 정확도도 높음

**웹캠 인식 속도 비교**
- MTCNN으로 실행했을 시 평균 1fps의 속도
- Mediapipe로 실행했을 시 평균 15fps의 속도가 나옴

## 3. 개선점
- 학습용 데이터셋의 7가지 표정이 사람이 봐도 표정을 모르는 경우가 있어 정확도가 떨어지기 때문에 기쁨과 슬픔의 표정을 눈과 입의 포인트로 잡아 정확도를 높일 예정
- 인식 속도가 낮아 웹캠에 적용하기 힘듬 (Mediapipe로 해결)
