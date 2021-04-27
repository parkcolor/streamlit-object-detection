import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
from ssdfile import ssd
from yolofile import yolo
from segmentation import ssg
import webbrowser

def main():
    st.title('Object Dectection Project')

    menu = ['Home', 'SSD', 'YOLO','Semantic Segmentation']
    choice = st.sidebar.selectbox('Menu',menu)

    if choice == 'Home':
        st.header('본 프로젝트에서는 SSD, YOLO, Semantic Segmentation을 활용하여 Object Detection을 실습하였습니다.')
        st.write('각 페이지를 통해 model을 선택하여 이미지파일을 input하면 결과를 보실 수 있습니다.')
        st.image('ssd.png')

    if choice == 'SSD':
        st.header('SSD : Single Shot Multibox Detetor')
        st.write('2015년 빠른 속도를 앞세운 YOLO모델이 탄생하였지만, 정확도 측면에서 다소 한계점이 있었고, 작은 물체를 잘 잡아내지 못하는 단점을 보완하는 목적으로 SSD가 탄생하였습니다.'        )
        st.image('ssd1.png')
        ssd()


    if choice == 'YOLO':
        st.header('YOLO : You Only Look Once')
        st.image('yy1.png')
        st.write('입력받은 이미지를 S X S 그리드 영역으로 나누고 물체가 있을 영역에 Bounding Box를 예측하는 방식의 모델입니다.')
        st.image('yy.png')
        url1 = 'https://pjreddie.com/darknet/yolo/'
        if st.button('홈페이지 바로가기'):
            webbrowser.open_new_tab(url1)

        yolo()
        
    if choice == 'Semantic Segmentation':
        ssg()

if __name__ == '__main__':
    main()