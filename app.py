from matplotlib import container, image
import streamlit as st
from actualpage import show_actual_page
from predictpage import show_predict_page,accuracy_score_page
from actualpage import graph
from PIL import Image
from streamlit_option_menu import option_menu


#Old Code
#img = Image.open('sm.png')

#PAGE_CONFIG = {'page_title': 'StockPrediction','page_icon':':smiley:','layout':'centered'}

#st.set_page_config(page_title='Stock_Prediction', page_icon = 'sm.png', layout='centered', initial_sidebar_state='auto')

st.title('Stock Market')
#st.write(a,b)
selected = option_menu(
    menu_title=None,
    options=['Home','Prediction','Accuracy','Contact us'],
    icons=['house','Book','envolpe','envelope'],
    menu_icon='cast',
    default_index=0,
    orientation='horizontal'
    

    )

if selected == 'Home':
    show_actual_page()
if selected == 'Prediction':
    #show_predict_page()
    result = show_predict_page()

    show_predict_page.a = result[0]
    show_predict_page.b = result[1]
if selected == 'Accuracy':
    accuracy_score_page(show_predict_page.a,show_predict_page.b)
        
if selected == 'Contact us':
    st.write('')
    st.write('')
    #st.subheader('Comming Soon')
    img = Image.open("stm.jpeg")
    st.write('Developer:   Abdul Qadoos')
    st.write('occupation: Python Developer')
    st.write('Location: 31 Avenons, E13 8HU')
    st.write('company: Self Empolyed')
    st.image(img)





hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

