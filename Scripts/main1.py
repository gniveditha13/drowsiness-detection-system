
from keras.models import load_model
import cv2
import numpy as np
import streamlit as st


st.set_page_config(page_title="Drowsyness Detection System",page_icon="https://play-lh.googleusercontent.com/IAeHzhpURcSVysmYeqX08Qi4Y_lT9KdvUrFGjRG9vYC7xfkh6RK_ttqLgpOlRDapxS8=w240-h480-rw")

st.markdown(
    """
    <style>
    .h1 {
        font-family: 'Courier New', Monospace  ;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        color: purple
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<h1 class="h1">DROWSYNESS DETECTION SYSTEM</h1>', unsafe_allow_html=True)

#---------------------

st.markdown(
    """
    <style>
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: lightblue;
        color:purple;
    }
    </style>
    """,
    unsafe_allow_html=True
)
#---------------
#button color code

st.markdown(
    """
    <style>
    .stButton > button {
        background-color:lightblue;
        color: purple;
        border-radius: 12px;
        border: none;
        margin-top:10px;
        padding: 8px 14px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color:purple;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

#----------------------------
#text_input label text color code

st.markdown(
    """
    <style>
    .stTextInput label {
        color: purple;
        font-style:oblique;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#----------------------------
#text_input box color code

st.markdown(
    """
    <style>
    .stTextInput>div>div>input {
        background-color: lightblue;
        color: purple;
        border: 5px solid purple; 
        border-radius: 15px;  
    }
    .stTextInput>div>div>input:focus {
        background-color: purple; 
        border-color: lightblue;
        color:white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


#------------------------------

choice=st.sidebar.selectbox("MENU",("HOME","IP CAMERA","CAMERA","IMAGES THAT ARE DETECTED"))

if(choice=="HOME"):
    image_url = "https://miro.medium.com/v2/resize:fit:914/0*9iyN78vkR9gUuY7u"
    st.markdown(
    """
    <style>
    .image-container {
        margin-bottom:20px;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.markdown(
    f'<div class="image-container"><img src="{image_url}" width="707"></div>',
    unsafe_allow_html=True
    )
    
    st.markdown( """
    <style>
    .para {
        font-family: 'Courier New', Monospace  ;
        font-size: 16px;
        line-height: 1.5;
        color:purple;
        margin-bottom: 20px;
    }
    </style> <div class="para"> A Drowsiness Detection System is a software application or platform designed to help in detecting drowsiness in humans. This sytem offers a proactive approach to monitor and prevent drowsiness-related accidents. By leveraging computer vision, machine learning, and user-friendly interface design, the application aims to enhance safety and awareness in various settings, including driving, working, and studying.This system keep tracks the facial features like eyes closed,yawning and nodding of head and if all of these are captured in the video then it alerts.</div>""", unsafe_allow_html=True)

elif(choice=="IP CAMERA"):
    url=st.text_input("Enter IP Camera URL: ")
    window=st.empty()
    ipcambtn=st.button("Start Detection")
    if(ipcambtn):
        vid=cv2.VideoCapture(url)
        facexml=cv2.CascadeClassifier("face.xml")
        eyesxml=cv2.CascadeClassifier("eyes.xml")
        drowsymodel=load_model("final.h5",compile=False)
        ipcamstopbtn=st.button("Stop Detection")
        if(ipcamstopbtn):
            vid.release()
            st.experimental_rerun()
        i=1
        while True:
            flag,frame=vid.read()
            if flag:
                faces=facexml.detectMultiScale(frame)
                eyes=eyesxml.detectMultiScale(frame)
                for (x,y,l,w) in faces:
                    faceimg=frame[y:y+w,x:x+l]
                    faceimg=cv2.resize(faceimg, (224, 224), interpolation=cv2.INTER_AREA)
                    faceimg=np.asarray(faceimg, dtype=np.float32).reshape(1, 224, 224, 3)
                    faceimg=(faceimg / 127.5) - 1
                    pred=drowsymodel.predict(faceimg)[0][0]
                    print(pred)
                    if(pred>=0.9):
                        for (ex,ey,el,ew) in eyes:
                            cv2.rectangle(frame,(ex,ey),(ex+el,ey+ew),(0,0,255),2)
                            cv2.putText(frame, "Wake Up!!", (100,80),cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 255),5)
                            path="facesdetected/"+str(i)+".png"  
                            cv2.imwrite(path,frame[y:y+w,x:x+l])
                            i=i+1
                            cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),5)
                window.image(frame,channels="BGR")
        
elif(choice=="CAMERA"):
    cam=st.selectbox("Choose any camera to capture the video",("--Select here--","Webcam","USB Camera"))
    window=st.empty()
    btn=st.button("Start Detection")
    if(btn):
        c=0
        if(cam=="USB Camera"):
            c=1
        vid=cv2.VideoCapture(c)
        facexml=cv2.CascadeClassifier("face.xml")
        eyesxml=cv2.CascadeClassifier("eyes.xml")
        drowsymodel=load_model("final.h5",compile=False)
        btn2=st.button("Stop Detection")
        if(btn2):
            vid.release()
            st.experimental_rerun()
        i=1
        while True:
            flag,frame=vid.read()
            if flag:
                faces=facexml.detectMultiScale(frame)
                eyes=eyesxml.detectMultiScale(frame)
                for (x,y,l,w) in faces:
                    faceimg=frame[y:y+w,x:x+l]
                    faceimg=cv2.resize(faceimg, (224, 224), interpolation=cv2.INTER_AREA)
                    faceimg=np.asarray(faceimg, dtype=np.float32).reshape(1, 224, 224, 3)
                    faceimg=(faceimg / 127.5) - 1
                    pred=drowsymodel.predict(faceimg)[0][0]
                    print(pred)
                
                    if(pred>=0.9):
                        for (ex,ey,el,ew) in eyes:
                            cv2.rectangle(frame,(ex,ey),(ex+el,ey+ew),(0,0,255),2) #red
                            cv2.putText(frame, "Drowsiness Detected!!", (60,60),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255),2)
                            path="facesdetected/"+str(i)+".png"  
                            cv2.imwrite(path,frame[y:y+w,x:x+l])
                            i=i+1
                            cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),5) #green
                window.image(frame,channels="BGR")

elif(choice=="IMAGES THAT ARE DETECTED"):
    if "savedimg" not in st.session_state:
        st.session_state["savedimg"]=1
    
    nextbtn=st.button("Next Image")
    if nextbtn:
        st.session_state["savedimg"]=st.session_state["savedimg"]+1
        
    prevbtn=st.button("Previous Image")
    if prevbtn:
        st.session_state["savedimg"]=st.session_state["savedimg"]-1
        
    st.image("facesdetected/"+str(st.session_state["savedimg"])+".png")




#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    

